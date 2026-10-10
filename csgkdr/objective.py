from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from .divergence import cs_qmi, cs_total_correlation


def variance_floor(x: torch.Tensor, tau: float) -> torch.Tensor:
    std = torch.sqrt(x.var(dim=0, unbiased=True) + 1e-4)
    return F.relu(tau - std).mean()


def correlation_penalty(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    xc = x - x.mean(0, keepdim=True)
    xs = xc / torch.sqrt((xc ** 2).sum(0, keepdim=True) / (n - 1) + 1e-4)
    c = xs.t() @ xs / (n - 1)
    off = c - torch.diag(torch.diag(c))
    return (off ** 2).sum() / (d * (d - 1))


def cs_mi(x, y, estimator="eq31"):
    if estimator == "legacy":
        return cs_qmi(x, y, eps=1e-10)
    return cs_total_correlation(x, y, eps=1e-10)  # paper Eq.(31) / Theorem 1 with M = 2


def compute_loss(out: dict, y: torch.Tensor, modalities: List[str], lw: dict):
    dev = y.device
    pred = out["pred"].view(-1, 1)
    y = y.view(-1, 1)
    task = F.mse_loss(pred, y)
    rec = sum(F.mse_loss(out[f"rec_{m}"].clamp(-10, 10), out[f"h_{m}"].detach().clamp(-10, 10)) for m in modalities) / len(modalities)
    try:  # non-finite or exploding TC estimates are skipped for this batch
        tc = cs_total_correlation(*[out[f"u_{m}"] for m in modalities], eps=1e-10)
        if not torch.isfinite(tc) or tc > 50.0:
            tc = torch.zeros((), device=dev)
    except Exception:
        tc = torch.zeros((), device=dev)
    mi = torch.zeros((), device=dev)
    for m in modalities:
        try:
            v = cs_mi(out[f"c_{m}"], out[f"h_{m}"].detach(), lw.get("mi_estimator", "eq31"))
            if torch.isfinite(v):
                mi = mi + v.clamp(max=20.0)
        except Exception:
            pass
    ent = sum(torch.mean(out[f"c_{m}"] ** 2) for m in modalities) / len(modalities)
    U = lambda m: out.get(f"ureg_{m}", out[f"u_{m}"])
    Cm = lambda m: out.get(f"creg_{m}", out[f"c_{m}"])
    reps = [U(m) for m in modalities] + ([Cm(m) for m in modalities] if lw.get("var_on_common", True) else [])
    lvar = sum(variance_floor(r, lw.get("var_tau", 1.0)) for r in reps) / len(reps)
    lcov = sum(correlation_penalty(r) for r in reps) / len(reps) if lw.get("cov", 0.0) > 0 else torch.zeros((), device=dev)
    total = (lw["task"] * task + lw["rec"] * rec + lw["tc"] * tc - lw["beta"] * mi
             + lw["gamma"] * ent + lw.get("var", 0.0) * lvar + lw.get("cov", 0.0) * lcov)
    logs = {"loss": total, "task": task, "rec": rec, "tc": tc, "mi_common": mi, "entropy": ent, "var_floor": lvar, "cov_pen": lcov}
    return total, {k: float(v.detach().cpu()) for k, v in logs.items()}


def term_magnitudes(logs: dict, lw: dict) -> dict:
    parts = {"task": lw["task"] * logs["task"], "rec": lw["rec"] * logs["rec"], "tc": lw["tc"] * logs["tc"],
             "mi": lw["beta"] * logs["mi_common"], "entropy": lw["gamma"] * logs["entropy"],
             "var_floor": lw.get("var", 0.0) * logs["var_floor"], "cov": lw.get("cov", 0.0) * logs.get("cov_pen", 0.0)}
    tot = sum(abs(v) for v in parts.values()) or 1.0
    return {f"share_{k}": abs(v) / tot for k, v in parts.items()}
