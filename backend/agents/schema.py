from __future__ import annotations

import re
from typing import Any

import pandas as pd


class SchemaInspector:
    DOMAIN_KEYWORDS = {
        "finance": {
            "revenue", "profit", "loss", "cost", "price", "amount", "sales", "income",
            "expense", "margin", "budget", "tax", "payment", "invoice", "transaction",
            "balance", "credit", "debit", "roi", "profitability"
        },
        "ecommerce": {
            "product", "order", "customer", "cart", "delivery", "category", "rating",
            "review", "stock", "inventory", "sku", "brand", "purchase", "refund",
            "coupon", "discount", "checkout"
        },
        "hr": {
            "employee", "salary", "department", "hire", "attrition", "tenure",
            "performance", "manager", "role", "promotion", "leave", "headcount"
        },
        "marketing": {
            "campaign", "click", "impression", "conversion", "ctr", "cpc", "cpm",
            "channel", "lead", "funnel", "attribution", "spend", "roas", "engagement"
        },
        "healthcare": {
            "age", "bmi", "glucose", "pressure", "cholesterol", "patient", "diagnosis",
            "symptom", "medication", "hospital", "blood", "disease", "heart", "test"
        },
        "logistics": {
            "shipment", "delivery", "warehouse", "carrier", "tracking", "route",
            "freight", "dispatch", "transit", "supplier", "vendor"
        },
        "general": set(),
    }

    TARGET_HINTS = {
        "target", "label", "class", "outcome", "status", "result", "prediction",
        "fraud", "churn", "attrition", "converted", "approved", "survived", "default"
    }

    def inspect(self, df: pd.DataFrame, filename: str = "dataset.csv") -> dict[str, Any]:
        schema: dict[str, Any] = {
            "filename": filename,
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "domain": "general",
            "domain_confidence": 0,
            "has_time": False,
            "has_text": False,
            "has_target": False,
            "numeric_cols": [],
            "categorical_cols": [],
            "datetime_cols": [],
            "text_cols": [],
            "id_cols": [],
            "potential_targets": [],
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_cols": [],
            "total_missing": int(df.isna().sum().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "columns": {},
        }

        seen_signatures: dict[tuple, str] = {}

        for col in df.columns:
            info = self._inspect_column(df, col)
            schema["columns"][col] = info

            sig = tuple(df[col].dropna().head(50).astype(str).tolist())
            if sig in seen_signatures:
                schema["duplicate_cols"].append(col)
            else:
                seen_signatures[sig] = col

        for col, info in schema["columns"].items():
            role = info["role"]
            if role == "numeric":
                schema["numeric_cols"].append(col)
            elif role == "datetime":
                schema["datetime_cols"].append(col)
                schema["has_time"] = True
            elif role == "text":
                schema["text_cols"].append(col)
                schema["has_text"] = True
            elif role == "id":
                schema["id_cols"].append(col)
            else:
                schema["categorical_cols"].append(col)

        tokens = set()
        for col in df.columns:
            for w in re.split(r"[_\s\-]", str(col).lower()):
                if w:
                    tokens.add(w)

        scores = {
            domain: len(tokens & keywords)
            for domain, keywords in self.DOMAIN_KEYWORDS.items()
        }
        best_domain = max(scores, key=scores.get) if scores else "general"
        best_score = scores.get(best_domain, 0)
        schema["domain"] = best_domain if best_score > 0 else "general"
        schema["domain_confidence"] = int(best_score)

        for col in df.columns:
            name_tokens = set(re.split(r"[_\s\-]", str(col).lower()))
            if name_tokens & self.TARGET_HINTS:
                schema["potential_targets"].append(col)
                schema["has_target"] = True
            elif schema["columns"][col]["role"] == "categorical" and schema["columns"][col]["unique"] <= 8:
                schema["potential_targets"].append(col)

        schema["potential_targets"] = list(dict.fromkeys(schema["potential_targets"]))[:4]
        return schema

    def _inspect_column(self, df: pd.DataFrame, col: str) -> dict[str, Any]:
        s = df[col]
        nn = s.dropna()

        info: dict[str, Any] = {
            "dtype_raw": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
            "role": "unknown",
        }

        name = str(col).lower()

        if re.search(r"(date|time|dt|timestamp|created|updated|year|month)", name):
            try:
                pd.to_datetime(nn.head(20), errors="raise")
                info["role"] = "datetime"
                info["dtype"] = "datetime"
                return info
            except Exception:
                pass

        if s.dtype == object and len(nn) > 0:
            sample = str(nn.iloc[0])
            if re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", sample):
                info["role"] = "datetime"
                info["dtype"] = "datetime"
                return info

        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(nn, errors="coerce").dropna()
            info["dtype"] = "numeric"
            info["role"] = "numeric"
            if len(vals):
                info.update({
                    "min": round(float(vals.min()), 4),
                    "max": round(float(vals.max()), 4),
                    "mean": round(float(vals.mean()), 4),
                    "median": round(float(vals.median()), 4),
                    "std": round(float(vals.std()), 4),
                    "sum": round(float(vals.sum()), 4),
                    "skew": round(float(vals.skew()), 4) if len(vals) > 2 else None,
                    "q25": round(float(vals.quantile(0.25)), 4),
                    "q75": round(float(vals.quantile(0.75)), 4),
                    "zeros": int((vals == 0).sum()),
                    "negatives": int((vals < 0).sum()),
                })
                if info["unique"] == len(df) and vals.is_monotonic_increasing:
                    info["role"] = "id"
            return info

        if s.dtype == bool or info["unique"] == 2:
            info["dtype"] = "boolean"
            info["role"] = "categorical"
            info["top_values"] = s.value_counts(dropna=False).head(10).to_dict()
            return info

        if s.dtype == object:
            avg_len = float(nn.astype(str).str.len().mean()) if len(nn) else 0.0
            unique_ratio = info["unique"] / max(len(s), 1)

            if info["unique"] == len(df) and avg_len < 40:
                info["dtype"] = "string"
                info["role"] = "id"
            elif avg_len > 60 or unique_ratio > 0.7:
                info["dtype"] = "string"
                info["role"] = "text"
                info["avg_len"] = round(avg_len, 1)
            elif info["unique"] <= max(30, int(len(df) * 0.05)):
                info["dtype"] = "string"
                info["role"] = "categorical"
                info["top_values"] = s.value_counts(dropna=False).head(8).to_dict()
            else:
                info["dtype"] = "string"
                info["role"] = "text"
                info["avg_len"] = round(avg_len, 1)
            return info

        info["dtype"] = "other"
        info["role"] = "categorical"
        return info