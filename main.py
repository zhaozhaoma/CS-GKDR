"""
Interpretable Multimodal Brain Age Prediction via Cauchy-Schwarz Divergence-Induced Gács-Körner Common Information

"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 3-D CNN memory fragmentation

import pandas as pd
import torch

from csgkdr.config import apply_overrides, default_run_name, load_config, save_config, training_signature
from csgkdr.data.loaders import clinical_coverage, infer_diagnosis, make_loaders, make_split, subject_table
from csgkdr.evaluation import export_model, recalibrate_bn
from csgkdr.models import CSGKDR
from csgkdr.trainer import Trainer, sanity_check
from csgkdr.utils import (configure_progress, env_info, free_gpu, get_device, jdump, jload, log, now_str, set_seed,
                          tee_stdout, torch_load)


def resolve(args):
    cfg = load_config(args.config)
    if args.split:
        cfg["split"].update({"preset": args.split, "test_sites": None, "val_sites": None})
    if args.trimodal:
        cfg["modalities"] = ["fmri", "smri", "clin"]
    if args.bn_recal:
        cfg["evaluation"]["bn_recalibration"] = True
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    apply_overrides(cfg, args.set)
    bad = set(cfg["modalities"]) - {"fmri", "smri", "clin"}
    if bad or not {"fmri", "smri"} <= set(cfg["modalities"]):
        raise ValueError(f"modalities must be fmri,smri[,clin]; got {cfg['modalities']}")
    return cfg, Path(cfg["run_dir"]) / (args.name or default_run_name(cfg))


def check_run_dir(cfg: dict, run_dir: Path):
    """A run directory holds exactly one training configuration (results are never mixed)."""
    old = jload(run_dir / "config.json")
    if old is not None and training_signature(old) != training_signature(cfg):
        new_sig, old_sig = training_signature(cfg), training_signature(old)
        diff = [k for k in new_sig if old_sig.get(k) != new_sig.get(k)]
        raise ValueError(f"{run_dir} already contains a run with different settings ({', '.join(diff)}); "
                         f"use another --name or delete the directory")


def build_model(cfg: dict) -> CSGKDR:
    return CSGKDR(cfg["model"], use_clin="clin" in cfg["modalities"], clin_indices=cfg["clinical_columns"],
                  reg_nodrop=bool(cfg["loss"].get("reg_nodrop", False)))


def run(cfg: dict, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    log.add_file(run_dir / "log.txt")
    configure_progress(cfg.get("progress_bar", "auto"))
    check_run_dir(cfg, run_dir)
    save_config(cfg, run_dir / "config.json")
    device = get_device(cfg)

    # ---- data
    metadata = pd.read_csv(cfg["data"]["metadata_csv"])
    split = make_split(metadata, cfg)
    if not split["checks"]["ok"]:
        raise RuntimeError(f"split integrity check failed: {split['checks']}")
    dx, dx_info = infer_diagnosis(metadata, cfg)
    subj = subject_table(metadata, split, dx, cfg["diagnosis"]["column"])
    L = cfg["loss"]
    log(f"run: {run_dir}   device: {device}   seed: {cfg['seed']}   modalities: {cfg['modalities']}")
    log(f"split: test {split['test_sites']}  val {split['val_sites']}  N train/val/test = "
        f"{len(split['train_indices'])}/{len(split['val_indices'])}/{len(split['test_indices'])}   diagnosis: {dx_info['counts']}")
    log(f"d_C = {cfg['model']['common_dim']}, d_U = {cfg['model']['unique_dim']}   loss: task {L['task']}  rec {L['rec']}  "
        f"alpha(TC) {L['tc']}  beta {L['beta']} ({L['mi_estimator']})  gamma {L['gamma']}  var {L['var']} (tau {L['var_tau']})  cov {L['cov']}")
    free_gpu()
    loaders, age_scaler, proc = make_loaders(cfg, split, device)
    if proc is not None and clinical_coverage(proc, metadata):
        log(f"{clinical_coverage(proc, metadata)} subjects are missing from the phenotype file (all-missing clinical vector).", level="WARN")

    # ---- training
    if not ((run_dir / "train_done.json").exists() and (run_dir / "best.pt").exists()):
        set_seed(cfg["seed"])
        model = build_model(cfg).to(device)
        tr = Trainer(model, device, run_dir, cfg["training"], dict(L), age_std=age_scaler[1])
        if not (run_dir / "last.pt").exists():
            sc = sanity_check(tr, loaders["train"], cfg["training"]["sanity_batches"])
            jdump(sc, run_dir / "sanity_check.json")
            if not sc.get("ok"):
                raise RuntimeError(f"sanity check failed (non-finite loss or gradient): {sc}")
            log("sanity check ok; loss-term shares at initialisation: "
                + ", ".join(f"{k[6:]} {v:.2f}" for k, v in sc.items() if k.startswith("share_")))
        jdump({"age_scaler": {"mean": age_scaler[0], "std": age_scaler[1]},
               "split": {k: split[k] for k in ("test_sites", "val_sites")}, "diagnosis": dx_info,
               "env": env_info(), "created": now_str()}, run_dir / "run_info.json")
        with tee_stdout(run_dir / "train_log.txt"):
            info = tr.fit(loaders["train"], loaders["val"], cfg["training"]["n_epochs"])
        log(f"training finished: best epoch {info['best_epoch']} of {info['epochs_run']}, "
            f"best validation MAE {info['best_val_mae_norm'] * age_scaler[1]:.2f} y")
        del model, tr
        free_gpu()
    else:
        log("training already finished; evaluating the selected model")

    # ---- evaluation of the selected model (optionally after BN recalibration on the training subjects)
    ev = cfg["evaluation"]
    out = run_dir / ("eval_bnrecal" if ev["bn_recalibration"] else "eval")
    out.mkdir(exist_ok=True)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch_load(run_dir / "best.pt", map_location="cpu")["model"], strict=True)
    if ev["bn_recalibration"]:
        t0 = time.time()
        log(f"BN recalibration: recomputing BatchNorm statistics on the {len(split['train_indices'])} training subjects ...")
        nb = recalibrate_bn(model, loaders["train_eval"], device)
        log(f"BN recalibration: done ({nb} batches, {time.time() - t0:.0f} s)")
        torch.save({"model": model.state_dict(), "note": "BN statistics recalibrated on the training subjects"}, out / "model_recal.pt")
        jdump({"bn_recalibration": True, "n_batches": nb,
               "method": "eval-mode graph; BN layers collect cumulative statistics on the training subjects"}, out / "recal_info.json")
    else:
        jdump({"bn_recalibration": False}, out / "recal_info.json")
    t0 = time.time()
    log(f"evaluation: predicting train/val/test ({len(split['train_indices'])}/{len(split['val_indices'])}/"
        f"{len(split['test_indices'])} subjects) and exporting ...")
    metrics = export_model(run_dir.name, model, loaders, age_scaler, subj, out, device, ev["bias_correction_fit"])
    log(f"evaluation: done ({time.time() - t0:.0f} s)")
    t = metrics["test"]
    log(f"test: MAE {t['raw']['MAE']:.3f} y (bias-corrected {t['corr_train']['MAE']:.3f})  r {t['raw']['Corr']:.3f}  "
        f"R2 {t['raw']['R2']:.3f}   -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--trimodal", action="store_true", help="add the clinical modality (default: bimodal fMRI + sMRI)")
    ap.add_argument("--bn-recal", action="store_true", help="recalibrate BatchNorm statistics on the training subjects before evaluation")
    ap.add_argument("--split", default=None, help="paper split preset exp_01 ... exp_09 (default: exp_05)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", default=None, help="run name (default: <split>_<bi|tri>_seed<seed>)")
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. --set training.n_epochs=100")
    a = ap.parse_args(argv)
    try:
        cfg, run_dir = resolve(a)
        run(cfg, run_dir)
    except KeyboardInterrupt:
        log("interrupted; re-run the same command to resume", level="WARN")
        return 130
    except (KeyError, ValueError, RuntimeError) as e:
        log(f"{type(e).__name__}: {e}", level="ERROR")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
