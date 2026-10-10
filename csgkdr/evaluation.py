from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .trainer import move_batch
from .utils import apply_bias_correction, fit_bias_correction, jdump, log, regression_metrics, warn


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader, device: str, forward: Optional[Callable] = None, max_batches: int = 0) -> int:
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    saved = [m.momentum for m in bns]
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all batches
        m.train()          # only BN layers use/collect batch statistics
    fwd = forward or model
    n = 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if b["age"].shape[0] < 2:
            continue
        fwd(move_batch(b, device))
        n += 1
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.eval()
    return n


@torch.no_grad()
def run_inference(model, loader, device, want_latents=False):
    model.eval()
    keys, preds = [], []
    lat = defaultdict(list)
    for b in loader:
        keys += [str(k) for k in b["keys"]]
        out = model(move_batch(b, device))
        preds.append(out["pred"].view(-1).float().cpu())
        if want_latents:
            for k, v in out.items():
                if torch.is_tensor(v) and v.dim() == 2 and k != "pred":
                    lat[k].append(v.float().cpu())
    return keys, torch.cat(preds).numpy(), {k: torch.cat(v).numpy() for k, v in lat.items()}


def export_model(tag: str, model, loaders, age_scaler, subj: pd.DataFrame, out_dir, device, primary_fit="val") -> dict:
    out_dir = Path(out_dir)
    mean, std = age_scaler
    frames, lat_test = [], None
    for split, ld in (("train", loaders["train_eval"]), ("val", loaders["val"]), ("test", loaders["test"])):
        keys, p, lat = run_inference(model, ld, device, want_latents=(split == "test"))
        frames.append(pd.DataFrame({"subject_id": keys, "split": split, "pred_raw": p * std + mean}))
        if split == "test":
            lat_test = (keys, lat)
    df = pd.concat(frames, ignore_index=True).merge(subj.drop(columns=["split"], errors="ignore"), how="left",
                                                    left_on="subject_id", right_index=True)
    bad = ~np.isfinite(df["pred_raw"].values)
    if bad.any():
        raise FloatingPointError(f"[{tag}] {int(bad.sum())} subjects have non-finite predictions; they are not silently dropped.")
    fits = {}
    for src in ("val", "train"):
        s = df[df["split"] == src]
        fits[src] = fit_bias_correction(s["pred_raw"], s["age"])
        if abs(fits[src]["b"]) < 0.2:
            warn(f"[{tag}] bias-correction slope fitted on {src} is small (b={fits[src]['b']:.3f}); corrected values unstable.")
    for src, col in (("val", "pred_corr_val"), ("train", "pred_corr_train")):
        df[col] = apply_bias_correction(df["pred_raw"], fits[src]["a"], fits[src]["b"])
    prim = "pred_corr_val" if primary_fit == "val" else "pred_corr_train"
    df["pred_corrected"] = df[prim]
    df["PAD"] = df["pred_corrected"] - df["age"]
    df["PAD_raw"] = df["pred_raw"] - df["age"]
    df = df[["subject_id", "split", "site", "diagnosis", "label_raw", "age", "sex_male", "pred_raw", "pred_corr_val",
             "pred_corr_train", "pred_corrected", "PAD", "PAD_raw"]]
    df.to_csv(out_dir / "predictions.csv", index=False)
    metrics = {sp: {"raw": regression_metrics(g["pred_raw"], g["age"]),
                    "corr_val": regression_metrics(g["pred_corr_val"], g["age"]),
                    "corr_train": regression_metrics(g["pred_corr_train"], g["age"])}
               for sp, g in df.groupby("split")}
    jdump({"primary_fit": primary_fit, "fits": fits, "age_scaler": {"mean": mean, "std": std}}, out_dir / "debias.json")
    jdump({"model": tag, "metrics": metrics}, out_dir / "metrics.json")
    keys, lat = lat_test
    np.savez_compressed(out_dir / "latents_test.npz", keys=np.array(keys), **lat)
    t = metrics["test"]
    log(f"[{tag}] exported: test MAE raw={t['raw']['MAE']:.3f} corr(val-fit)={t['corr_val']['MAE']:.3f} "
        f"corr(train-fit)={t['corr_train']['MAE']:.3f}")
    return metrics
