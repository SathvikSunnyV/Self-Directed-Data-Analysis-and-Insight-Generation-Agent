from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class CleaningTools:
    """
    Stateful cleaner.  All other tool classes receive this object
    and read its `.df` property so they always work on the latest
    cleaned dataframe without needing a separate sync step.
    """

    def __init__(self, df: pd.DataFrame, memory=None):
        self._df = df.copy()
        self.memory = memory

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        self._df = value

    # ============================================================
    # CLEAN DATASET
    # ============================================================

    def clean_dataset(self) -> str:
        df = self._df.copy()
        original_rows = len(df)
        original_missing = int(df.isna().sum().sum())

        # ── Remove duplicates ────────────────────────────────────
        duplicate_rows = int(df.duplicated().sum())
        df = df.drop_duplicates()

        # ── Standardize column names ─────────────────────────────
        clean_columns = []
        for col in df.columns:
            col = str(col).strip().lower()
            col = col.replace(" ", "_").replace("-", "_")
            col = re.sub(r"[^a-zA-Z0-9_]", "", col)
            col = col or "col"
            clean_columns.append(col)
        df.columns = clean_columns

        # ── Try datetime conversion for obvious columns ───────────
        for col in df.columns:
            lower = col.lower()
            if any(kw in lower for kw in ["date", "time", "timestamp", "created", "updated"]):
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.6:
                        df[col] = parsed
                except Exception:
                    pass

        # ── Remove infinite values ────────────────────────────────
        df = df.replace([np.inf, -np.inf], np.nan)

        # ── Impute missing values ────────────────────────────────
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                imp = SimpleImputer(strategy="median")
                df[[col]] = imp.fit_transform(df[[col]])
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                pass  # leave datetime untouched
            else:
                if df[col].isna().any():
                    mode = df[col].mode(dropna=True)
                    fill_value = mode.iloc[0] if len(mode) else "Unknown"
                    df[col] = df[col].fillna(fill_value)

        remaining_missing = int(df.isna().sum().sum())

        # Persist cleaned df for downstream tools
        self._df = df

        # ── Memory findings ─────────────────────────────────────
        if self.memory:
            if duplicate_rows > 0:
                self.memory.add_finding(f"Removed {duplicate_rows} duplicate rows")
            reduced = original_missing - remaining_missing
            if reduced > 0:
                self.memory.add_finding(f"Resolved {reduced} missing values during cleaning")

        result = {
            "status": "cleaned",
            "rows_before": int(original_rows),
            "rows_after": int(len(df)),
            "duplicates_removed": duplicate_rows,
            "missing_before": int(original_missing),
            "missing_after": remaining_missing,
            "columns": list(df.columns),
        }
        return json.dumps(result, indent=2, default=str)

    # ============================================================
    # DROP HIGH-MISSING COLUMNS
    # ============================================================

    def drop_high_missing_columns(
        self,
        threshold_pct: float = 60.0,
    ) -> str:
        df = self._df.copy()
        removed = [col for col in df.columns if df[col].isna().mean() * 100 >= threshold_pct]

        if removed:
            df = df.drop(columns=removed)
            self._df = df

        if self.memory and removed:
            self.memory.add_finding(f"Dropped high-missing columns (>{threshold_pct}%): {', '.join(removed)}")

        return json.dumps({
            "threshold_pct": threshold_pct,
            "removed_columns": removed,
            "remaining_columns": list(df.columns),
        }, indent=2)

    # ============================================================
    # REMOVE CONSTANT COLUMNS
    # ============================================================

    def remove_constant_columns(self) -> str:
        df = self._df.copy()
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

        if constant_cols:
            df = df.drop(columns=constant_cols)
            self._df = df

        if self.memory and constant_cols:
            self.memory.add_finding(f"Removed constant columns: {', '.join(constant_cols)}")

        return json.dumps({
            "removed_columns": constant_cols,
            "remaining_columns": list(df.columns),
        }, indent=2)
