from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd

# ── HAMD-17 maximum scores (standard Hamilton Depression Rating Scale) ──────
HAMD_MAX_SCORES: Dict[str, int] = {
    "*HAMD1":  4,   # Depressed mood
    "*HAMD2":  4,   # Guilt
    "*HAMD3":  4,   # Suicide
    "*HAMD4":  2,   # Early insomnia
    "*HAMD5":  2,   # Middle insomnia
    "*HAMD6":  2,   # Late insomnia
    "*HAMD7":  4,   # Work and activities
    "*HAMD8":  4,   # Psychomotor retardation
    "*HAMD9":  4,   # Agitation   (value 10 → out of range → NaN)
    "*HAMD10": 4,   # Anxiety (psychic)
    "*HAMD11": 4,   # Anxiety (somatic)
    "*HAMD12": 2,   # Gastrointestinal
    "*HAMD13": 2,   # General somatic
    "*HAMD14": 2,   # Genital symptoms
    "*HAMD15": 4,   # Hypochondriasis
    "*HAMD16": 3,   # Loss of weight
    "*HAMD17": 2,   # Insight
}
HAMD_COLS: List[str] = list(HAMD_MAX_SCORES.keys())

# ── Dimension layout ─────────────────────────────────────────────────────────
SEX_ONEHOT_DIM   = 2          # Sex one-hot
EDU_DIM          = 1          # Education (z-score)
HAMD_DIM         = 17         # HAMD sub-items
VALUE_DIM        = SEX_ONEHOT_DIM + EDU_DIM + HAMD_DIM   # 20
MASK_DIM         = 1 + 1 + HAMD_DIM                       # 19 (one per original variable)
CLIN_DIM         = VALUE_DIM + MASK_DIM                   # 39  ← exported constant


class ClinicalPreprocessor:
    def __init__(self) -> None:
        self.edu_mean: float = 0.0
        self.edu_std:  float = 1.0
        self.hamd_medians: Dict[str, float] = {}   # key=col, val on [0,1] scale
        self._fitted: bool = False
        self._lookup: Dict[str, np.ndarray] = {}   # subID → [39] vector

    # ── Public API ───────────────────────────────────────────────────────────

    def fit_transform(self, excel_path: str, train_subids: List[str]) -> None:
        """
        Load Excel, fit on *train_subids*, then encode every subject in both
        MDD and Controls sheets.  Must be called before :meth:`get_vector`.
        """
        raw_df = self._load_raw(excel_path)
        self._fit(raw_df, train_subids)
        self._build_lookup(raw_df)
        self._fitted = True

    def get_vector(self, subid: str) -> np.ndarray:
        subid = str(subid).strip()
        if subid in self._lookup:
            return self._lookup[subid]
        # Fallback: zeros + fully masked
        vec = np.zeros(CLIN_DIM, dtype=np.float32)
        vec[VALUE_DIM:] = 1.0
        return vec

    def save(self, path: str) -> None:
        obj = {
            "edu_mean":     self.edu_mean,
            "edu_std":      self.edu_std,
            "hamd_medians": self.hamd_medians,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def load_stats(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.edu_mean     = float(obj["edu_mean"])
        self.edu_std      = float(obj["edu_std"])
        self.hamd_medians = {k: float(v) for k, v in obj["hamd_medians"].items()}

    def rebuild_lookup(self, excel_path: str) -> None:
        raw_df = self._load_raw(excel_path)
        self._build_lookup(raw_df)
        self._fitted = True

    def __contains__(self, subid: str) -> bool:
        return str(subid).strip() in self._lookup

    def known_count(self) -> int:
        return len(self._lookup)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _load_raw(excel_path: str) -> pd.DataFrame:
        mdd  = pd.read_excel(excel_path, sheet_name="MDD")
        ctrl = pd.read_excel(excel_path, sheet_name="Controls")
        mdd  = mdd.rename(columns={"ID": "subID"})
        ctrl = ctrl.rename(columns={"ID": "subID"})
        for col in HAMD_COLS:
            if col not in ctrl.columns:
                ctrl[col] = np.nan
        keep_cols = ["subID", "Sex", "Education (years)"] + HAMD_COLS
        mdd_sub  = mdd[[c for c in keep_cols if c in mdd.columns]].copy()
        ctrl_sub = ctrl[[c for c in keep_cols if c in ctrl.columns]].copy()

        df = pd.concat([mdd_sub, ctrl_sub], ignore_index=True)
        df = ClinicalPreprocessor._clean_missing(df)
        return df

    @staticmethod
    def _clean_missing(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.replace(-9999, np.nan)
        df = df.replace("",    np.nan)
        df = df.replace("[]",  np.nan)
        df["Education (years)"] = pd.to_numeric(df["Education (years)"], errors="coerce")
        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")
        for col, max_score in HAMD_MAX_SCORES.items():
            if col in df.columns:
                col_vals = pd.to_numeric(df[col], errors="coerce")
                col_vals = col_vals.where(
                    col_vals.between(0.0, float(max_score), inclusive="both"),
                    other=np.nan,
                )
                df[col] = col_vals
        return df

    def _fit(self, df: pd.DataFrame, train_subids: List[str]) -> None:
        train_df = df[df["subID"].isin(set(train_subids))]
        # Education z-score parameters
        edu_vals = train_df["Education (years)"].dropna()
        self.edu_mean = float(edu_vals.mean()) if len(edu_vals) > 0 else 0.0
        self.edu_std  = float(edu_vals.std())  if len(edu_vals) > 1 else 1.0
        if self.edu_std < 1e-8:
            self.edu_std = 1.0
        self.hamd_medians = {}
        for col, max_score in HAMD_MAX_SCORES.items():
            if col in train_df.columns:
                scaled = train_df[col].dropna() / float(max_score)
                self.hamd_medians[col] = (
                    float(scaled.median()) if len(scaled) > 0 else 0.0
                )
            else:
                self.hamd_medians[col] = 0.0

    def _build_lookup(self, df: pd.DataFrame) -> None:
        self._lookup = {}
        for _, row in df.iterrows():
            subid = str(row["subID"]).strip()
            self._lookup[subid] = self._encode_row(row)

    def _encode_row(self, row: pd.Series) -> np.ndarray:
        values = np.zeros(VALUE_DIM, dtype=np.float32)
        masks  = np.zeros(MASK_DIM,  dtype=np.float32)   # 0=observed, 1=missing
        # ── Sex (one-hot, indices 0-1) ────────────────────────────────────────
        sex_val = row.get("Sex", np.nan)
        if pd.isna(sex_val):
            # missing: leave one-hot as zeros, set mask
            masks[0] = 1.0
        else:
            sex_int = int(sex_val)
            if sex_int == 1:
                values[0], values[1] = 1.0, 0.0
            else:                               # 2 → second class
                values[0], values[1] = 0.0, 1.0
            masks[0] = 0.0
        # ── Education (z-score, index 2) ──────────────────────────────────────
        edu_val = row.get("Education (years)", np.nan)
        if pd.isna(edu_val):
            values[2] = 0.0          # imputed with mean (→ 0 after z-score)
            masks[1]  = 1.0
        else:
            values[2] = (float(edu_val) - self.edu_mean) / (self.edu_std + 1e-8)
            masks[1]  = 0.0
        # ── HAMD1-17 (indices 3-19, mask indices 2-18) ───────────────────────
        for i, col in enumerate(HAMD_COLS):
            max_score = float(HAMD_MAX_SCORES[col])
            raw_val   = row.get(col, np.nan)
            if pd.isna(raw_val):
                values[3 + i] = float(self.hamd_medians.get(col, 0.0))
                masks[2 + i]  = 1.0
            else:
                values[3 + i] = float(raw_val) / max_score
                masks[2 + i]  = 0.0
        return np.concatenate([values, masks]).astype(np.float32)