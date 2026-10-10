from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .divergence import cs_total_correlation
from .objective import compute_loss, term_magnitudes
from .utils import SHOW_PROGRESS, atomic_torch_save, get_rng_state, jdump, jload, log, set_rng_state, torch_load


def move_batch(batch: dict, device: str) -> dict:
    d = torch.device(device)
    out = {}
    for k, v in batch.items():
        if hasattr(v, "edge_index") and hasattr(v, "batch_size"):
            out[k] = type(v)(x=v.x.to(d, non_blocking=True), edge_index=v.edge_index.to(d, non_blocking=True),
                             batch=v.batch.to(d, non_blocking=True), batch_size=v.batch_size)
        elif torch.is_tensor(v):
            out[k] = v.to(d, non_blocking=True)
        else:
            out[k] = v
    return out


def _iter(loader, desc):
    if SHOW_PROGRESS["on"]:
        from tqdm import tqdm
        return tqdm(loader, desc=desc, ncols=110, leave=False)
    return loader


class Trainer:
    def __init__(self, model, device: str, out_dir, tcfg: dict, lw: dict, age_std: float = 1.0):
        self.model, self.device, self.lw = model, device, lw
        self.age_std = float(age_std)  # only used to report the validation MAE in years
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.accum = max(1, int(tcfg["grad_accum_steps"]))
        self.clip = float(tcfg.get("grad_clip", 1.0))
        self.patience = int(tcfg["patience"])
        self.warmup = int(tcfg["warmup_epochs"])
        bb, head = [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                (bb if ("fmri_backbone" in name or "smri_backbone" in name) else head).append(p)
        groups = []
        if bb:
            lr_bb = tcfg["backbone_lr"] if tcfg.get("finetune_backbone", True) else 0.0
            groups.append({"params": bb, "lr": lr_bb, "initial_lr": lr_bb})
        groups.append({"params": head, "lr": tcfg["learning_rate"], "initial_lr": tcfg["learning_rate"]})
        self.optim = AdamW(groups, weight_decay=tcfg["weight_decay"])
        self.initial_lrs = [g["initial_lr"] for g in self.optim.param_groups]
        self.sched = CosineAnnealingWarmRestarts(self.optim, T_0=20, T_mult=2)
        self.best_val_mae, self.patience_counter, self.should_stop = float("inf"), 0, False
        self.modalities = list(model.modalities)

    def _set_lr(self, epoch: int):  # linear warmup, then cosine annealing with warm restarts
        if epoch <= self.warmup:
            f = epoch / max(1, self.warmup)
            for i, g in enumerate(self.optim.param_groups):
                g["lr"] = self.initial_lrs[i] * f
        else:
            self.sched.step()

    def train_epoch(self, loader, epoch: int):
        self.model.train()
        run = defaultdict(float)
        n_skip = 0
        self.optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(_iter(loader, f"Epoch {epoch:03d}"), start=1):
            batch = move_batch(batch, self.device)
            y = batch["age"]
            if y.shape[0] < 2:  # BatchNorm / batch statistics need >= 2 samples
                n_skip += 1
                continue
            out = self.model(batch)
            loss, logs = compute_loss(out, y, self.modalities, self.lw)
            if not torch.isfinite(loss):
                n_skip += 1
                self.optim.zero_grad(set_to_none=True)
                continue
            (loss / self.accum).backward()
            if step % self.accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)
            for k, v in logs.items():
                run[k] += v
        if len(loader) % self.accum != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
        d = max(1, len(loader))
        res = {k: v / d for k, v in run.items()}
        res["skipped_batches"] = n_skip
        return res

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        P, Y, L = [], [], []
        lat = defaultdict(list)
        for batch in _iter(loader, "Eval"):
            batch = move_batch(batch, self.device)
            out = self.model(batch)
            loss, _ = compute_loss(out, batch["age"], self.modalities, self.lw)
            if torch.isfinite(loss):
                L.append(float(loss))
            P.append(out["pred"].view(-1).float().cpu())
            Y.append(batch["age"].view(-1).float().cpu())
            for k in ["c"] + [f"u_{m}" for m in self.modalities] + [f"c_{m}" for m in self.modalities]:
                lat[k].append(out[k].float().cpu())
        p, y = torch.cat(P), torch.cat(Y)
        m = torch.isfinite(p)
        res = {"mae": float((p[m] - y[m]).abs().mean()) if m.any() else float("nan"),
               "loss": float(np.mean(L)) if L else float("nan"),
               "r": float(np.corrcoef(p[m].numpy(), y[m].numpy())[0, 1]) if int(m.sum()) > 2 else float("nan")}
        # representation monitor (validation set; Eq. (31) estimator in float64)
        yy = y.double().view(-1, 1)
        for k, v in lat.items():
            x = torch.cat(v).double()
            sd = x.std(0)
            res[f"minsd_{k}"] = float(sd.min())
            ev = torch.linalg.eigvalsh(torch.cov(x.t())).clamp(min=0)
            res[f"effdim_{k}"] = float(ev.sum() ** 2 / (ev ** 2).sum()) if float(ev.sum()) > 0 else 0.0
            if k == "c" or k.startswith("u_"):
                try:
                    res[f"I_{k};Y"] = float(cs_total_correlation(x, yy, eps=1e-10))
                except Exception:
                    res[f"I_{k};Y"] = float("nan")
        return res

    def fit(self, train_loader, val_loader, n_epochs: int) -> dict:
        done_p, last_p = self.out_dir / "train_done.json", self.out_dir / "last.pt"
        if done_p.exists():
            return jload(done_p)
        history, start = [], 1
        if last_p.exists():
            st = torch_load(last_p, map_location=self.device)
            self.model.load_state_dict(st["model"])
            self.optim.load_state_dict(st["optimizer"])
            self.sched.load_state_dict(st["scheduler"])
            self.best_val_mae, self.patience_counter, self.should_stop = st["best_val_mae"], st["patience_counter"], st["should_stop"]
            history, start = st["history"], st["epoch"] + 1
            set_rng_state(st["rng"])
            log(f"[{self.out_dir.name}] resuming from epoch {start}")
        best_epoch = (jload(self.out_dir / "best_val.json") or {}).get("epoch")
        for epoch in range(start, n_epochs + 1):
            if self.should_stop:
                break
            t0 = time.time()
            self._set_lr(epoch)
            tr = self.train_epoch(train_loader, epoch)
            va = self.evaluate(val_loader)
            line = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()},
                    "lr_head": float(self.optim.param_groups[-1]["lr"])}
            history.append(line)
            improved = False
            if np.isfinite(va["mae"]):
                if va["mae"] < self.best_val_mae:
                    self.best_val_mae, self.patience_counter, best_epoch = va["mae"], 0, epoch
                    improved = True
                    atomic_torch_save({"model": self.model.state_dict(), "epoch": epoch}, self.out_dir / "best.pt")
                    jdump({"val_mae": va["mae"], "epoch": epoch}, self.out_dir / "best_val.json")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.should_stop = True
            line["epoch_time_s"] = time.time() - t0
            self._report(epoch, n_epochs, tr, va, line, best_epoch, improved)
            jdump(history, self.out_dir / "history.json")
            atomic_torch_save({"model": self.model.state_dict(), "optimizer": self.optim.state_dict(),
                               "scheduler": self.sched.state_dict(), "epoch": epoch, "best_val_mae": self.best_val_mae,
                               "patience_counter": self.patience_counter, "should_stop": self.should_stop,
                               "history": history, "rng": get_rng_state()}, last_p)
        info = {"best_val_mae_norm": float(self.best_val_mae), "best_epoch": best_epoch, "epochs_run": len(history),
                "stopped_early": bool(self.should_stop),
                "selection": "min validation MAE (normalized age); test set never used"}
        jdump(info, done_p)
        if last_p.exists():
            last_p.unlink()
        return info

    def _report(self, epoch, n_epochs, tr, va, line, best_epoch, improved):
        f = lambda v, nd=3: "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"
        mods = self.modalities
        print("-" * 104)
        print(f"Epoch {epoch:3d}/{n_epochs}   lr {line['lr_head']:.2e}   {line['epoch_time_s']:.1f} s"
              + ("   * new best" if improved else ""))
        print(f"  train | loss {f(tr.get('loss'))}  task {f(tr.get('task'))}  rec {f(tr.get('rec'))}  "
              f"TC {f(tr.get('tc'), 4)}  MI(C;h) {f(tr.get('mi_common'), 4)}  C^2 {f(tr.get('entropy'))}  "
              f"L_var {f(tr.get('var_floor'))}  L_cov {f(tr.get('cov_pen'))}"
              + (f"  skipped {tr['skipped_batches']}" if tr.get("skipped_batches") else ""))
        print(f"  val   | MAE {f(va['mae'] * self.age_std, 2)} y  r {f(va.get('r'))}  loss {f(va.get('loss'))}  "
              f"best {f(self.best_val_mae * self.age_std, 2)} y (epoch {best_epoch})  "
              f"patience {self.patience_counter}/{self.patience}")
        print("  repr  | min sd  " + "  ".join(f"U_{m} {f(va.get(f'minsd_u_{m}'), 2)}" for m in mods)
              + f"  C {f(va.get('minsd_c'), 2)}   |  d_eff  "
              + "  ".join(f"U_{m} {f(va.get(f'effdim_u_{m}'), 1)}" for m in mods) + f"  C {f(va.get('effdim_c'), 1)}")
        print("  info  | " + f"I(C;Y) {f(va.get('I_c;Y'), 4)}  " + "  ".join(f"I(U_{m};Y) {f(va.get(f'I_u_{m};Y'), 4)}" for m in mods),
              flush=True)



def sanity_check(trainer: Trainer, loader, n_batches: int = 3) -> dict:
    """Forward/backward on a few training batches; all side effects rolled back."""
    model = trainer.model
    snap = copy.deepcopy(model.state_dict())
    rng = get_rng_state()
    model.train()
    recs = []
    try:
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            batch = move_batch(batch, trainer.device)
            if batch["age"].shape[0] < 2:
                continue
            out = model(batch)
            loss, logs = compute_loss(out, batch["age"], trainer.modalities, trainer.lw)
            trainer.optim.zero_grad(set_to_none=True)
            fin, gn = bool(torch.isfinite(loss)), 0.0
            fg = True
            if fin:
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        fg &= bool(torch.isfinite(p.grad).all())
                        gn += float(p.grad.detach().float().norm() ** 2)
            recs.append({**logs, **term_magnitudes(logs, trainer.lw), "finite_loss": fin, "finite_grad": fg, "grad_norm": math.sqrt(gn)})
    finally:
        trainer.optim.zero_grad(set_to_none=True)
        model.load_state_dict(snap)
        set_rng_state(rng)
    if not recs:
        return {"ok": False, "reason": "no batch evaluated"}
    agg = {k: float(np.mean([r[k] for r in recs])) for k in recs[0] if not isinstance(recs[0][k], bool)}
    agg["ok"] = all(r["finite_loss"] and r["finite_grad"] for r in recs)
    agg["n_batches"] = len(recs)
    return agg
