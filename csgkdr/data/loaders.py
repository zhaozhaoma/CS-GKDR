from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .clinical import CLIN_DIM, MASK_DIM, VALUE_DIM, ClinicalPreprocessor
from .datasets import RESTfMRIDataset, RESTsMRIDataset, set_cache_root


@dataclass
class GraphBatch:
    x: torch.Tensor           # [N_total, F]
    edge_index: torch.Tensor  # [2, E_total]
    batch: torch.Tensor       # [N_total]
    batch_size: int


class RestMetaCSGKDRDataset(Dataset):
    def __init__(self, joint_dataset, clinical_processor: Optional[ClinicalPreprocessor] = None):
        super().__init__()
        self.joint_dataset = joint_dataset
        self.clinical_processor = clinical_processor

    def __len__(self):
        return len(self.joint_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # joint_dataset now returns 5-tuple: (fmri, smri, age, subid, site)
        fmri_data, smri_data, age, subid, site = self.joint_dataset[idx]
        x          = fmri_data.x           # [num_nodes, num_features]
        edge_index = fmri_data.edge_index  # [2, num_edges]
        if isinstance(smri_data, tuple):
            smri = smri_data[0]
        else:
            smri = smri_data
        if self.clinical_processor is not None:
            clin_vec = torch.tensor(
                self.clinical_processor.get_vector(subid), dtype=torch.float32
            )
        else:
            clin_vec = torch.zeros(CLIN_DIM, dtype=torch.float32)
            clin_vec[CLIN_DIM // 2:] = 1.0   # all values masked

        return {
            "x":          x,
            "edge_index": edge_index,
            "smri":       smri,
            "clin":       clin_vec,           # [39]
            "age":        age.view(1),
            "key":        subid,
            "site":       int(site),
        }

def collate_restmeta_batch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch_list) == 0:
        raise ValueError("Empty batch")

    device = torch.device("cpu")
    smri_list = [b["smri"] for b in batch_list]
    C      = smri_list[0].shape[0]
    D_max  = max(s.shape[1] for s in smri_list)
    H_max  = max(s.shape[2] for s in smri_list)
    W_max  = max(s.shape[3] for s in smri_list)

    padded_smri = []
    for smri in smri_list:
        c, d, h, w = smri.shape
        pad_d = D_max - d
        pad_h = H_max - h
        pad_w = W_max - w
        # F.pad: (W_left, W_right, H_left, H_right, D_left, D_right, ...)
        padded = F.pad(smri, (0, pad_w, 0, pad_h, 0, pad_d, 0, 0),
                       mode="constant", value=0.0)
        padded_smri.append(padded.unsqueeze(0))

    smri_batch = torch.cat(padded_smri, dim=0)   # [B, C, D_max, H_max, W_max]
    xs, edge_indices, batches = [], [], []
    node_offset = 0
    for i, b in enumerate(batch_list):
        x          = b["x"]           # [num_nodes, num_features]
        edge_index = b["edge_index"]  # [2, num_edges]
        n_nodes    = x.shape[0]

        xs.append(x)
        batches.append(torch.full((n_nodes,), i, dtype=torch.long, device=x.device))
        edge_indices.append(edge_index + node_offset)
        node_offset += n_nodes

    x_cat     = torch.cat(xs, dim=0).to(device)
    batch_vec = torch.cat(batches, dim=0).to(device)
    edge_cat  = torch.cat(edge_indices, dim=1).to(device)

    fmri_batch = GraphBatch(
        x=x_cat,
        edge_index=edge_cat.long(),
        batch=batch_vec.long(),
        batch_size=len(batch_list),
    )

    # ── Clinical vectors ─────────────────────────────────────────────────────
    clin_batch = torch.stack([b["clin"] for b in batch_list], dim=0)  # [B, CLIN_DIM]

    # ── Age labels and metadata ───────────────────────────────────────────────
    ages  = torch.stack([b["age"] for b in batch_list], dim=0)   # [B, 1]
    keys  = [b["key"]  for b in batch_list]
    sites = [b["site"] for b in batch_list]

    return {
        "fmri":  fmri_batch,
        "smri":  smri_batch,
        "clin":  clin_batch,
        "age":   ages,
        "keys":  keys,
        "sites": sites,
    }


def create_restmeta_dataloaders(
    fmri_data_dir: str,
    smri_data_dir: str,
    metadata_path: str,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    batch_size: int = 8,
    num_workers: int = 0,
    smri_imgtypes: Optional[List[str]] = None,
    smri_normalize: bool = True,
    pin_memory: bool = True,
    clinical_excel_path: Optional[str] = None,
    clinical_processor: Optional[ClinicalPreprocessor] = None,
) -> Tuple:
    """
    Create train/val/test DataLoaders for REST-meta-MDD in CS-GKDR format.

    Parameters
    ----------
    fmri_data_dir       : Path to fMRI data directory
    smri_data_dir       : Path to sMRI data directory
    metadata_path       : Path to metadata CSV
    train_indices       : List of training sample indices
    val_indices         : List of validation sample indices
    test_indices        : List of test sample indices
    batch_size          : Batch size
    num_workers         : Number of DataLoader workers
    smri_imgtypes       : sMRI image types (default: ["wc1"])
    smri_normalize      : Whether to normalize sMRI volumes
    pin_memory          : Whether to pin memory for CUDA
    clinical_excel_path : Path to the phenotypic Excel file.  If provided and
                          clinical_processor is None, a ClinicalPreprocessor
                          will be fitted on the training subjects.
    clinical_processor  : Pre-fitted ClinicalPreprocessor to reuse across
                          experiments.  When provided, clinical_excel_path is
                          only used to rebuild the per-subject lookup if the
                          processor has not been fitted yet.

    Returns
    -------
    (train_loader, val_loader, test_loader, age_scaler, clinical_processor)
      age_scaler          : (mean, std) tensors from training-set ages
      clinical_processor  : The fitted ClinicalPreprocessor (or None)
    """
    from torch.utils.data import DataLoader, Subset

    if smri_imgtypes is None:
        smri_imgtypes = ["wc1"]

    # ── Import base datasets ─────────────────────────────────────────────────

    # ── Base datasets (full, then subset for memory efficiency) ──────────────
    fmri_full = RESTfMRIDataset(
        data_dir=fmri_data_dir,
        metadata_path=metadata_path,
        split="full",
    )
    smri_full = RESTsMRIDataset(
        data_dir=smri_data_dir,
        metadata_path=metadata_path,
        imgtypes=smri_imgtypes,
        split="full",
        normalize=smri_normalize,
        inmemory=False,
    )

    fmri_train = Subset(fmri_full, train_indices)
    fmri_val   = Subset(fmri_full, val_indices)
    fmri_test  = Subset(fmri_full, test_indices)

    smri_train = Subset(smri_full, train_indices)
    smri_val   = Subset(smri_full, val_indices)
    smri_test  = Subset(smri_full, test_indices)

    # ── Load metadata ─────────────────────────────────────────────────────────
    metadata_df = pd.read_csv(metadata_path)

    # ── ClinicalPreprocessor ─────────────────────────────────────────────────
    if clinical_processor is None and clinical_excel_path is not None:
        train_subids = metadata_df.iloc[train_indices]["subID"].astype(str).tolist()
        clinical_processor = ClinicalPreprocessor()
        clinical_processor.fit_transform(clinical_excel_path, train_subids)
        print(
            f"[ClinicalPreprocessor] fitted on {len(train_subids)} training subjects. "
            f"Lookup size: {clinical_processor.known_count()}"
        )

    # ── Joint dataset wrapper (defined locally) ───────────────────────────────
    class JointDatasetWrapper(torch.utils.data.Dataset):
        """Combines fMRI Subset + sMRI Subset + normalised age."""

        def __init__(self, fmri_ds, smri_ds, indices, metadata, age_scaler=None):
            self.fmri_dataset = fmri_ds
            self.smri_dataset = smri_ds
            self.metadata     = metadata.iloc[indices].reset_index(drop=True)

            ages_raw = torch.tensor(self.metadata["age"].values, dtype=torch.float32)

            if age_scaler is not None:
                self.age_mean = age_scaler[0]
                self.age_std  = age_scaler[1]
            else:
                self.age_mean = ages_raw.mean()
                self.age_std  = ages_raw.std()

            self.ages = (ages_raw - self.age_mean) / (self.age_std + 1e-8)

        def __len__(self):
            return len(self.fmri_dataset)

        def __getitem__(self, idx):
            fmri_data = self.fmri_dataset[idx]
            smri_data = self.smri_dataset[idx]
            if isinstance(smri_data, tuple):
                smri_data = smri_data[0]
            age    = self.ages[idx]
            subid  = str(self.metadata["subID"].iloc[idx]).strip()
            site   = (
                int(self.metadata["site"].iloc[idx])
                if "site" in self.metadata.columns
                else -1
            )
            return fmri_data, smri_data, age, subid, site

    # ── Create joint datasets with age normalisation ──────────────────────────
    joint_train = JointDatasetWrapper(fmri_train, smri_train, train_indices, metadata_df)
    age_scaler  = (joint_train.age_mean, joint_train.age_std)

    joint_val  = JointDatasetWrapper(fmri_val,  smri_val,  val_indices,  metadata_df, age_scaler)
    joint_test = JointDatasetWrapper(fmri_test, smri_test, test_indices, metadata_df, age_scaler)

    # ── Wrap in CS-GKDR adapter ───────────────────────────────────────────────
    train_dataset = RestMetaCSGKDRDataset(joint_train, clinical_processor)
    val_dataset   = RestMetaCSGKDRDataset(joint_val,   clinical_processor)
    test_dataset  = RestMetaCSGKDRDataset(joint_test,  clinical_processor)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_restmeta_batch,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_restmeta_batch,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_restmeta_batch,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, age_scaler, clinical_processor


# ----------------------------------------------------------------------------
# Clinical column subsets of the 39-d vector, splits, diagnosis, subject table
# ----------------------------------------------------------------------------
assert CLIN_DIM == 39 and VALUE_DIM == 20 and MASK_DIM == 19, "unexpected clinical vector layout"

# Leave-two-sites-out splits used in the paper (test sites / validation sites).
PAPER_SPLITS = {
    "exp_01": {"test": ["S20"], "val": ["S15", "S17"]},
    "exp_02": {"test": ["S7", "S9"], "val": ["S15", "S11"]},
    "exp_03": {"test": ["S14", "S19"], "val": ["S15", "S11"]},
    "exp_04": {"test": ["S1", "S8"], "val": ["S15", "S17"]},
    "exp_05": {"test": ["S17", "S10"], "val": ["S4", "S9"]},
    "exp_06": {"test": ["S23", "S15"], "val": ["S14", "S11"]},
    "exp_07": {"test": ["S22", "S13"], "val": ["S23", "S17"]},
    "exp_08": {"test": ["S21", "S11"], "val": ["S2", "S19"]},
    "exp_09": {"test": ["S2", "S4"], "val": ["S23", "S1"]},
}


def resolve_sites(cfg: dict) -> dict:
    """Split preset (cfg["split"]["preset"]) or explicit test/validation sites."""
    s = cfg["split"]
    if s.get("test_sites") and s.get("val_sites"):
        return {"test_sites": list(s["test_sites"]), "val_sites": list(s["val_sites"])}
    p = PAPER_SPLITS[s["preset"]]
    return {"test_sites": p["test"], "val_sites": p["val"]}


def _site_int(s) -> int:
    return int(s[1:]) if isinstance(s, str) and s.upper().startswith("S") else int(s)


def site_split(metadata: pd.DataFrame, test_sites, val_sites):
    """Same rule as create_site_based_splits() in train_restmeta_multi_exp.py."""
    t, v = {_site_int(s) for s in test_sites}, {_site_int(s) for s in val_sites}
    tr, va, te = [], [], []
    for site in metadata["site"].unique():
        idx = metadata[metadata["site"] == site].index.tolist()
        (te if int(site) in t else va if int(site) in v else tr).extend(idx)
    return [int(i) for i in tr], [int(i) for i in va], [int(i) for i in te]


def make_split(metadata: pd.DataFrame, cfg: dict) -> dict:
    sites = resolve_sites(cfg)
    tr, va, te = site_split(metadata, sites["test_sites"], sites["val_sites"])
    ids = metadata["subID"].astype(str)
    sets = {k: set(ids.iloc[i]) for k, i in (("train", tr), ("val", va), ("test", te))}
    dup = ids[ids.duplicated()].unique().tolist()
    checks = {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_duplicated_subIDs": len(dup),
              "overlap_train_val": len(sets["train"] & sets["val"]), "overlap_train_test": len(sets["train"] & sets["test"]),
              "overlap_val_test": len(sets["val"] & sets["test"]),
              "test_sites_found": sorted({int(s) for s in metadata.loc[te, "site"]}),
              "val_sites_found": sorted({int(s) for s in metadata.loc[va, "site"]})}
    checks["ok"] = bool(len(te) and len(va) and not dup and not checks["overlap_train_val"]
                        and not checks["overlap_train_test"] and not checks["overlap_val_test"])
    return {"train_indices": tr, "val_indices": va, "test_indices": te, "checks": checks,
            "test_sites": sites["test_sites"], "val_sites": sites["val_sites"]}


def infer_diagnosis(metadata: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, dict]:
    d = cfg["diagnosis"]
    lab = metadata[d.get("column", "label")]
    info = {"column": d.get("column", "label")}
    mdd, hc = d.get("mdd_values"), d.get("hc_values")
    if mdd is None or hc is None:
        grp = metadata["subID"].astype(str).str.extract(r"^S\d+-(\d)-")[0]
        g1, g2 = lab[grp == "1"].mode(), lab[grp == "2"].mode()
        if len(g1) and len(g2) and g1.iloc[0] != g2.iloc[0]:
            mdd, hc = [g1.iloc[0]], [g2.iloc[0]]
            info["inferred_from"] = "subID group digit (REST-meta-MDD naming: 1 = MDD, 2 = NC) matched to the label column"
        else:
            mdd, hc = [1], [0]
            info["inferred_from"] = "default 1 = MDD, 0 = HC (please confirm in the config)"
    py = lambda v: v.item() if hasattr(v, "item") else v
    info["mdd_values"], info["hc_values"] = [py(v) for v in mdd], [py(v) for v in hc]
    dx = pd.Series("other", index=metadata.index, dtype=object)
    dx[lab.isin(mdd)] = "MDD"
    dx[lab.isin(hc)] = "HC"
    info["counts"] = dx.value_counts().to_dict()
    return dx, info


def make_loaders(cfg: dict, split: dict, device: str):
    """Train/val/test loaders + a non-shuffled training loader (prediction export, BN recalibration)."""
    d, t = cfg["data"], cfg["training"]
    set_cache_root(d["cache_root"])
    excel = None
    if "clin" in cfg["modalities"]:  # the phenotype file is read only for the trimodal model
        excel = d.get("clinical_excel_path")
        if not excel or not os.path.isfile(excel):
            raise RuntimeError(f"The clinical modality needs the phenotype Excel file "
                               f"(data.clinical_excel_path); not found: {excel}")
    tr, va, te, age_scaler, proc = create_restmeta_dataloaders(
        fmri_data_dir=d["fmri_data_dir"], smri_data_dir=d["smri_data_dir"], metadata_path=d["metadata_csv"],
        train_indices=split["train_indices"], val_indices=split["val_indices"], test_indices=split["test_indices"],
        batch_size=t["batch_size"], num_workers=d["num_workers"], smri_imgtypes=d["smri_imgtypes"],
        smri_normalize=d["smri_normalize"], pin_memory=(device == "cuda"), clinical_excel_path=excel)
    tr_eval = DataLoader(tr.dataset, batch_size=t["batch_size"], shuffle=False, num_workers=d["num_workers"],
                         collate_fn=collate_restmeta_batch, pin_memory=(device == "cuda"))
    loaders = {"train": tr, "val": va, "test": te, "train_eval": tr_eval}
    return loaders, (float(age_scaler[0]), float(age_scaler[1])), proc


def subject_table(metadata: pd.DataFrame, split: dict, dx: pd.Series, dx_column: str) -> pd.DataFrame:
    """Per-subject site / diagnosis / age / sex / split (index = subject ID)."""
    lab = np.array([""] * len(metadata), dtype=object)
    for k in ("train", "val", "test"):
        lab[split[f"{k}_indices"]] = k
    sex = pd.to_numeric(metadata["sex"], errors="coerce") if "sex" in metadata.columns else pd.Series(np.nan, index=metadata.index)
    return pd.DataFrame({"site": metadata["site"].values, "label_raw": metadata[dx_column].values,
                         "diagnosis": dx.values, "age": metadata["age"].values.astype(float),
                         "sex_male": sex.map({1: 1.0, 2: 0.0}).values, "split": lab},
                        index=metadata["subID"].astype(str).str.strip().values)


def clinical_coverage(proc: ClinicalPreprocessor, metadata: pd.DataFrame) -> int:
    """Number of subjects missing from the phenotype file (they receive an all-missing vector)."""
    return int(sum(str(s).strip() not in proc for s in metadata["subID"]))
