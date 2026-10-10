from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import List, Optional

from .utils import deep_update

DEFAULT_CONFIG = {
    "run_dir": "./runs",
    "seed": 42,
    "device": "auto",          # auto | cuda | cpu
    "progress_bar": "auto",    # auto | on | off
    "data": {
        "fmri_data_dir": "data/REST-meta-MDD/fMRI/AAL",      # <subID>.npy, AAL time series [116, T]
        "smri_data_dir": "data/REST-meta-MDD/VBM",           # <imgtype>/<subID>.nii.gz
        "metadata_csv": "data/REST-meta-MDD/metadata.csv",   # subID, sex, age, site, label, ...
        # phenotype file; read only by the trimodal model (--trimodal)
        "clinical_excel_path": "data/REST-meta-MDD/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx",
        "cache_root": "data/cache",                          # only used for relative data paths
        "smri_imgtypes": ["wc1"],
        "smri_normalize": True,
        "num_workers": 0,
    },
    # Leave-two-sites-out split
    # or explicit "test_sites" / "val_sites" lists, which take precedence over the preset.
    "split": {"preset": "exp_05", "test_sites": None, "val_sites": None},
    "diagnosis": {"column": "label", "mdd_values": None, "hc_values": None},
    "modalities": ["fmri", "smri"],   # default bimodal; --trimodal sets ["fmri", "smri", "clin"]
    "clinical_columns": None,         # trimodal only: subset of the 39-d clinical vector; None = all 39
    "model": {
        "fmri_in_channels": 116, "fmri_hidden": 256, "fmri_num_layers": 3, "fmri_out_dim": 128,
        "fmri_dropout": 0.3, "fmri_num_nodes": 116, "fmri_emb_style": "none", "fmri_emb_size": 128,
        "fmri_use_attention": True, "fmri_use_hierarchical_pooling": True,
        "smri_channels": [32, 64, 128, 256, 256, 64], "smri_use_attention": True, "smri_use_depthwise_conv": True,
        "clin_hidden": 64, "clin_out_dim": 32,
        "common_dim": 32, "unique_dim": 32, "dropout": 0.3,
        "enc_activation": "leaky_relu",
    },
    "training": {
        "batch_size": 10, "grad_accum_steps": 2,
        "n_epochs": 200, "patience": 50, "warmup_epochs": 5,
        "learning_rate": 1e-3, "backbone_lr": 5e-4, "weight_decay": 1e-3,
        "finetune_backbone": True, "grad_clip": 1.0,
        "sanity_batches": 3,
    },
    "loss": {
        "task": 1.0, "rec": 0.1,
        "tc": 0.05,                 # alpha: CS total correlation among the unique representations
        "beta": 1.0,                # common-information term I_CS(C^m; h^m)
        "gamma": 0.01,              # mean(C^2), proxy for H(C)
        "var": 1.0, "var_tau": 1.0,  # per-dimension variance floor (anti-collapse)
        "var_on_common": True,      # apply the anti-collapse terms to C^m as well as U^m
        "cov": 1.0,                 # mean squared off-diagonal correlation (anti-collapse)
        "reg_nodrop": True,         # anti-collapse terms on dropout-free encoder outputs
        "mi_estimator": "eq31",     # eq31 (paper Eq. 31) | legacy (earlier cs_qmi)
    },
    "evaluation": {
        "bn_recalibration": False,        # --bn-recal: recalibrate BatchNorm statistics on the training subjects
        "bias_correction_fit": "train",   # train (paper Eq. 32) | val ; both are always exported
    },
}


# Settings that define a trained model; a run directory refuses to mix different values.
TRAINING_KEYS = ["data", "split", "diagnosis", "modalities", "clinical_columns", "model", "training", "loss", "seed"]


def _strip_comments(d):
    if isinstance(d, dict):
        return {k: _strip_comments(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


def load_config(path: Optional[str]) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = deep_update(cfg, _strip_comments(json.load(f)))
    return cfg


def apply_overrides(cfg: dict, sets: List[str]) -> dict:
    """--set key.sub=value (value parsed as JSON when possible, e.g. 0.1, true, [\"fmri\",\"smri\"])."""
    for item in sets or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise ValueError(f"--set expects key=value, got {item!r}")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"unknown config key {key!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = val
    return cfg


def default_run_name(cfg: dict) -> str:
    s = cfg["split"]
    split = s["preset"] if not (s.get("test_sites") and s.get("val_sites")) else "test-" + "-".join(s["test_sites"])
    mods = "tri" if "clin" in cfg["modalities"] else "bi"
    return f"{split}_{mods}_seed{cfg['seed']}"


def training_signature(cfg: dict) -> dict:
    out = {k: copy.deepcopy(cfg.get(k)) for k in TRAINING_KEYS}
    out["data"] = {k: v for k, v in out["data"].items() if k != "num_workers"}
    if "clin" not in (out["modalities"] or []):  # clinical settings are unused by the bimodal model
        out["data"].pop("clinical_excel_path", None)
        out["clinical_columns"] = None
    out["training"] = {k: v for k, v in out["training"].items() if k != "sanity_batches"}
    return out


def save_config(cfg: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
