from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MLTools:

    def __init__(self, df_source, memory=None):
        self._source = df_source
        self.memory = memory

    @property
    def df(self) -> pd.DataFrame:
        if hasattr(self._source, "df"):
            return self._source.df
        return self._source

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    def feature_importance(
        self,
        target_col: str,
    ) -> str:
        df = self.df.copy()

        if target_col not in df.columns:
            return f"Target column not found: {target_col}"

        y = df[target_col]
        X = df.drop(columns=[target_col]).copy()

        if X.empty:
            return "No feature columns available."

        # ── Encode features ──────────────────────────────────────
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                try:
                    X[col] = X[col].astype("int64")
                except Exception:
                    X[col] = 0
            elif not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna("Unknown"))

        X = X.replace([np.inf, -np.inf], np.nan)
        imp = SimpleImputer(strategy="median")
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

        if y.nunique(dropna=True) <= 1:
            return f"Target '{target_col}' has only one unique value — cannot train model."

        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 15

        if is_regression:
            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            y_fit = pd.to_numeric(y, errors="coerce").fillna(y.median())
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            le = LabelEncoder()
            y_fit = le.fit_transform(y.astype(str).fillna("Unknown"))

        model.fit(X, y_fit)
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        top_features = {str(k): round(float(v), 6) for k, v in importance.head(20).items()}

        if self.memory and top_features:
            top_feat = list(top_features.keys())[0]
            self.memory.add_finding(f"Most important feature for '{target_col}': {top_feat}")

        return json.dumps({
            "target_column": target_col,
            "task_type": "regression" if is_regression else "classification",
            "top_features": top_features,
        }, indent=2)

    # ============================================================
    # CLUSTER ANALYSIS
    # ============================================================

    def cluster_analysis(
        self,
        cols: list[str] | None = None,
        k: int = 0,
    ) -> str:
        df = self.df.copy()

        num_cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if len(num_cols) < 2:
            return "Need at least 2 numeric columns for clustering."

        X = df[num_cols].replace([np.inf, -np.inf], np.nan)
        imp = SimpleImputer(strategy="median")
        X = pd.DataFrame(imp.fit_transform(X), columns=num_cols)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if k <= 0:
            if len(df) < 50:
                k = 2
            elif len(df) < 500:
                k = 3
            else:
                k = 5

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)
        df["_cluster"] = clusters

        profiles = {}
        for cid in sorted(df["_cluster"].unique()):
            subset = df[df["_cluster"] == cid]
            profiles[int(cid)] = {
                "size": int(len(subset)),
                "means": subset[num_cols].mean().round(4).to_dict(),
            }

        cluster_sizes = {int(cid): int((df["_cluster"] == cid).sum()) for cid in sorted(df["_cluster"].unique())}

        if self.memory:
            self.memory.add_finding(f"K-Means clustering found {k} clusters in the numeric feature space")

        return json.dumps({"k": k, "cluster_sizes": cluster_sizes, "profiles": profiles}, indent=2, default=str)
