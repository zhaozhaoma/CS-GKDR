from __future__ import annotations
import contextlib
import copy
import datetime as _dt
import io
import json
import os
import platform
import random
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return str(o)


def jdump(obj: Any, path, indent: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def jload(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_torch_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(tmp, path)


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:  # very old torch
        return torch.load(str(path), map_location=map_location)


def now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _Logger:
    def __init__(self):
        self.files: List[Path] = []

    def add_file(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.files.append(path)

    def __call__(self, *msg, level: str = "INFO"):
        text = " ".join(str(m) for m in msg)
        line = f"[{now_str()}] [{level}] {text}"
        print(line, flush=True)
        for p in self.files:
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


log = _Logger()


def warn(*msg):
    log(*msg, level="WARN")


class Tee(io.TextIOBase):
    """Duplicate writes to several streams (used to keep per-model train logs)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


@contextlib.contextmanager
def tee_stdout(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old


SHOW_PROGRESS = {"on": False}


def configure_progress(mode):
    SHOW_PROGRESS["on"] = sys.stdout.isatty() if mode == "auto" else bool(mode is True or mode == "on")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    st = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    if not st:
        return
    try:
        random.setstate(st["python"])
        np.random.set_state(st["numpy"])
        torch.set_rng_state(st["torch"])
        if torch.cuda.is_available() and "cuda" in st:
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as e:  # pragma: no cover
        warn(f"Could not restore RNG state: {e}")


def free_gpu():
    """Release cached GPU blocks between models (no effect on computations)."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device(cfg) -> str:
    d = cfg.get("device", "auto")
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def regression_metrics(pred, age) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    age = np.asarray(age, dtype=np.float64).reshape(-1)
    m = np.isfinite(pred) & np.isfinite(age)
    pred, age = pred[m], age[m]
    n = int(len(pred))
    if n < 2:
        return {"N": n, "MAE": float("nan"), "Corr": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    err = pred - age
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((age - age.mean()) ** 2))
    if np.std(pred) > 0 and np.std(age) > 0:
        corr = float(np.corrcoef(pred, age)[0, 1])
    else:
        corr = float("nan")
    return {
        "N": n,
        "MAE": float(np.mean(np.abs(err))),
        "Corr": corr,
        "RMSE": float(np.sqrt(sse / n)),
        "R2": float(1.0 - sse / sst) if sst > 0 else float("nan"),
    }


def fit_bias_correction(train_pred_raw, train_age) -> Dict[str, float]:
    """Fit pred_raw = a + b * age on the given (non-test) subjects (paper Eq. 32)."""
    p = np.asarray(train_pred_raw, dtype=np.float64).reshape(-1)
    y = np.asarray(train_age, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(y)
    b, a = np.polyfit(y[m], p[m], 1)
    return {"a": float(a), "b": float(b), "n_fit": int(m.sum()),
            "formula": "pred_corrected = (pred_raw - a) / b"}


def apply_bias_correction(pred_raw, a: float, b: float):
    if abs(b) < 1e-8:
        raise ValueError(f"Bias-correction slope b={b} is ~0; correction undefined.")
    return (np.asarray(pred_raw, dtype=np.float64) - a) / b


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "host": socket.gethostname(),
    }
    try:
        import torch_geometric
        info["torch_geometric"] = torch_geometric.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
