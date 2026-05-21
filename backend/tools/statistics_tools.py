from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats


class StatisticsTools:
    """
    Accepts either a CleaningTools instance or a plain DataFrame.
    If a CleaningTools instance is passed, .df is read dynamically
    so this tool always operates on the latest cleaned dataframe.
    """

    def __init__(self, df_source, memory=None):
        self._source = df_source
        self.memory = memory

    @property
    def df(self) -> pd.DataFrame:
        # Support both CleaningTools (with .df property) and plain DataFrames
        if hasattr(self._source, "df"):
            return self._source.df
        return self._source

    # ============================================================
    # DATASET PROFILE
    # ============================================================

    def describe_dataset(self) -> str:
        df = self.df
        profile = {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "missing_total": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": {},
        }

        for col in df.columns:
            s = df[col]
            nn = s.dropna()
            info = {
                "dtype": str(s.dtype),
                "missing": int(s.isna().sum()),
                "missing_pct": round(float(s.isna().mean() * 100), 2),
                "unique": int(s.nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(s):
                info["type"] = "numeric"
                if len(nn):
                    info.update({
                        "min": round(float(nn.min()), 4),
                        "max": round(float(nn.max()), 4),
                        "mean": round(float(nn.mean()), 4),
                        "median": round(float(nn.median()), 4),
                        "std": round(float(nn.std()), 4),
                        "sum": round(float(nn.sum()), 4),
                    })
            elif pd.api.types.is_datetime64_any_dtype(s):
                info["type"] = "datetime"
            else:
                info["type"] = "categorical"
                info["top_values"] = {
                    str(k): int(v)
                    for k, v in s.value_counts(dropna=False).head(5).items()
                }

            profile["columns"][col] = info

        return json.dumps(profile, indent=2, default=str)

    # ============================================================
    # MISSING VALUES
    # ============================================================

    def missing_values(self) -> str:
        df = self.df
        result = [
            {
                "column": col,
                "missing": int(df[col].isna().sum()),
                "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            }
            for col in df.columns
        ]
        result.sort(key=lambda x: x["missing"], reverse=True)
        return json.dumps(result, indent=2)

    # ============================================================
    # COMPUTE STATISTICS
    # ============================================================

    def compute_statistics(
        self,
        cols: list[str] | None = None,
        stat: str = "describe",
    ) -> str:
        df = self.df

        num_cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not num_cols:
            return "No numeric columns found."

        if stat == "describe":
            return df[num_cols].describe().round(4).to_string()

        if stat == "variance":
            result = {}
            for col in num_cols:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                mean = float(s.mean())
                result[col] = {
                    "variance": round(float(s.var()), 4),
                    "std": round(float(s.std()), 4),
                    "range": round(float(s.max() - s.min()), 4),
                    "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
                    "cv_pct": round(float((s.std() / mean) * 100), 2) if mean != 0 else None,
                }
            return json.dumps(result, indent=2)

        if stat == "normality":
            result = {}
            for col in num_cols:
                s = df[col].dropna()
                if len(s) < 8:
                    continue
                sample = s.sample(min(500, len(s)), random_state=42)
                shapiro_stat, p_value = stats.shapiro(sample)
                result[col] = {
                    "shapiro_stat": round(float(shapiro_stat), 4),
                    "p_value": round(float(p_value), 4),
                    "is_normal": bool(p_value > 0.05),
                    "skew": round(float(s.skew()), 4),
                    "kurtosis": round(float(s.kurtosis()), 4),
                }
            return json.dumps(result, indent=2)

        return f"Unknown stat type: {stat}. Use describe, variance, or normality."

    # ============================================================
    # CORRELATION MATRIX
    # ============================================================

    def correlation_matrix(
        self,
        cols: list[str] | None = None,
    ) -> str:
        df = self.df
        num_cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:14]

        if len(num_cols) < 2:
            return "Need at least 2 numeric columns."

        corr = df[num_cols].corr().round(4)
        pairs = []
        for i, col1 in enumerate(num_cols):
            for col2 in num_cols[i + 1:]:
                val = corr.loc[col1, col2]
                if pd.notna(val):
                    pairs.append({"col1": col1, "col2": col2, "corr": round(float(val), 4)})

        pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
        top_pairs = pairs[:15]

        if self.memory:
            for pair in top_pairs[:5]:
                strength = "strong" if abs(pair["corr"]) > 0.7 else "moderate" if abs(pair["corr"]) > 0.4 else "weak"
                direction = "positive" if pair["corr"] >= 0 else "negative"
                self.memory.add_finding(
                    f"{pair['col1']} and {pair['col2']} have {strength} {direction} correlation (r={pair['corr']})"
                )

        return json.dumps({"top_correlations": top_pairs, "matrix": corr.to_dict()}, indent=2)

    # ============================================================
    # SEGMENTATION
    # ============================================================

    def segment_by(
        self,
        group_col: str,
        metric_col: str,
        agg: str = "mean",
    ) -> str:
        df = self.df

        if group_col not in df.columns:
            return f"Group column not found: {group_col}"
        if metric_col not in df.columns:
            return f"Metric column not found: {metric_col}"
        if df[group_col].nunique() > 100:
            return f"'{group_col}' has too many unique values for segmentation (>100)."

        grouped = df.groupby(group_col)[metric_col].agg(agg).reset_index()
        grouped.columns = [group_col, "value"]
        grouped = grouped.dropna().sort_values("value", ascending=False)

        if len(grouped) and self.memory:
            top = grouped.iloc[0]
            self.memory.add_finding(
                f"Highest {metric_col} by {group_col}: {top[group_col]} ({round(float(top['value']), 4)})"
            )

        return json.dumps({
            "group_col": group_col,
            "metric_col": metric_col,
            "aggregation": agg,
            "top_10": grouped.head(10).to_dict("records"),
            "bottom_5": grouped.tail(5).to_dict("records"),
        }, indent=2, default=str)

    # ============================================================
    # OUTLIER DETECTION
    # ============================================================

    def detect_outliers(
        self,
        col: str,
    ) -> str:
        df = self.df

        if col not in df.columns:
            return f"Column not found: {col}"

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 5:
            return "Not enough numeric values for outlier detection."

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        iqr_outliers = s[(s < lower) | (s > upper)]
        z_scores = abs(stats.zscore(s))
        z_outliers = s[z_scores > 3]

        result = {
            "column": col,
            "iqr_outlier_count": int(len(iqr_outliers)),
            "zscore_outlier_count": int(len(z_outliers)),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "sample_iqr_outliers": [round(float(v), 4) for v in iqr_outliers.head(10).tolist()],
        }

        if self.memory and len(iqr_outliers):
            self.memory.add_finding(f"{len(iqr_outliers)} outliers detected in '{col}'")

        return json.dumps(result, indent=2)
