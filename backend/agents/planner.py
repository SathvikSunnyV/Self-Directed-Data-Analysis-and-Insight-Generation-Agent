from __future__ import annotations


class Planner:
    """
    Dynamic planning engine.

    Decides WHAT analyses should run based on:
    - schema structure
    - detected domain
    - available column types
    - user question
    """

    def create_plan(
        self,
        schema: dict,
        user_question: str | None = None,
    ) -> list[dict]:

        plan: list[dict] = []

        numeric_cols = schema.get("numeric_cols", [])
        categorical_cols = schema.get("categorical_cols", [])
        datetime_cols = schema.get("datetime_cols", [])
        text_cols = schema.get("text_cols", [])
        targets = schema.get("potential_targets", [])

        rows = schema.get("rows", 0)
        domain = schema.get("domain", "general")

        # ============================================================
        # ALWAYS RUN
        # ============================================================

        plan.append({
            "id": "clean_dataset",
            "name": "Data Cleaning",
            "tool": "clean_dataset",
            "args": {},
            "priority": 1,
            "reason": "Dataset should always be cleaned before analysis.",
        })

        plan.append({
            "id": "dataset_profile",
            "name": "Dataset Profiling",
            "tool": "describe_dataset",
            "args": {},
            "priority": 2,
            "reason": "Understand overall dataset structure and quality.",
        })

        plan.append({
            "id": "missing_analysis",
            "name": "Missing Value Analysis",
            "tool": "missing_values",
            "args": {},
            "priority": 3,
            "reason": "Detect columns with high missingness.",
        })

        # ============================================================
        # NUMERIC ANALYSIS
        # ============================================================

        if len(numeric_cols) >= 1:

            plan.append({
                "id": "basic_statistics",
                "name": "Basic Statistical Analysis",
                "tool": "compute_statistics",
                "args": {
                    "cols": numeric_cols[:8],
                    "stat": "describe",
                },
                "priority": 4,
                "reason": "Numeric columns exist.",
            })

            plan.append({
                "id": "variance_analysis",
                "name": "Variance Analysis",
                "tool": "compute_statistics",
                "args": {
                    "cols": numeric_cols[:8],
                    "stat": "variance",
                },
                "priority": 5,
                "reason": "Analyze spread and variability.",
            })

            plan.append({
                "id": "normality_analysis",
                "name": "Distribution Analysis",
                "tool": "compute_statistics",
                "args": {
                    "cols": numeric_cols[:6],
                    "stat": "normality",
                },
                "priority": 6,
                "reason": "Understand distribution characteristics.",
            })

        # ============================================================
        # CORRELATION
        # ============================================================

        if len(numeric_cols) >= 2:

            plan.append({
                "id": "correlation_analysis",
                "name": "Correlation Analysis",
                "tool": "correlation_matrix",
                "args": {
                    "cols": numeric_cols[:12],
                },
                "priority": 7,
                "reason": "Multiple numeric variables detected.",
            })

        # ============================================================
        # OUTLIER DETECTION
        # ============================================================

        if len(numeric_cols) >= 1:

            for col in numeric_cols[:4]:

                plan.append({
                    "id": f"outlier_{col}",
                    "name": f"Outlier Detection - {col}",
                    "tool": "detect_outliers",
                    "args": {
                        "col": col,
                    },
                    "priority": 8,
                    "reason": f"Check anomalies in {col}.",
                })

        # ============================================================
        # SEGMENTATION
        # ============================================================

        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:

            for cat in categorical_cols[:2]:

                for num in numeric_cols[:2]:

                    plan.append({
                        "id": f"segment_{cat}_{num}",
                        "name": f"Segmentation - {cat} vs {num}",
                        "tool": "segment_by",
                        "args": {
                            "group_col": cat,
                            "metric_col": num,
                            "agg": "mean",
                        },
                        "priority": 9,
                        "reason": "Compare performance across categories.",
                    })

        # ============================================================
        # TIME SERIES
        # ============================================================

        if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:

            for dt in datetime_cols[:1]:

                for num in numeric_cols[:2]:

                    plan.append({
                        "id": f"timeseries_{dt}_{num}",
                        "name": f"Time Series Analysis - {num}",
                        "tool": "time_series_analysis",
                        "args": {
                            "date_col": dt,
                            "value_col": num,
                        },
                        "priority": 10,
                        "reason": "Temporal data detected.",
                    })

        # ============================================================
        # FEATURE IMPORTANCE
        # ============================================================

        if len(targets) >= 1 and len(numeric_cols) >= 2:

            for target in targets[:2]:

                plan.append({
                    "id": f"feature_importance_{target}",
                    "name": f"Feature Importance - {target}",
                    "tool": "feature_importance",
                    "args": {
                        "target_col": target,
                    },
                    "priority": 11,
                    "reason": "Potential target variable detected.",
                })

        # ============================================================
        # CLUSTERING
        # ============================================================

        if len(numeric_cols) >= 2 and rows >= 30:

            plan.append({
                "id": "cluster_analysis",
                "name": "Cluster Analysis",
                "tool": "cluster_analysis",
                "args": {
                    "cols": numeric_cols[:6],
                    "k": 0,
                },
                "priority": 12,
                "reason": "Enough numeric data for clustering.",
            })

        # ============================================================
        # TEXT ANALYSIS
        # ============================================================

        if len(text_cols) >= 1:

            for col in text_cols[:2]:

                plan.append({
                    "id": f"text_analysis_{col}",
                    "name": f"Text Analysis - {col}",
                    "tool": "text_frequency",
                    "args": {
                        "col": col,
                        "top_n": 20,
                    },
                    "priority": 13,
                    "reason": "Text column detected.",
                })

        # ============================================================
        # DOMAIN-SPECIFIC LOGIC
        # ============================================================

        if domain == "finance":

            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:

                plan.append({
                    "id": "financial_segmentation",
                    "name": "Financial KPI Segmentation",
                    "tool": "segment_by",
                    "args": {
                        "group_col": categorical_cols[0],
                        "metric_col": numeric_cols[0],
                        "agg": "sum",
                    },
                    "priority": 14,
                    "reason": "Finance domain detected.",
                })

        elif domain == "ecommerce":

            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:

                plan.append({
                    "id": "ecommerce_performance",
                    "name": "E-Commerce Performance Analysis",
                    "tool": "segment_by",
                    "args": {
                        "group_col": categorical_cols[0],
                        "metric_col": numeric_cols[0],
                        "agg": "mean",
                    },
                    "priority": 14,
                    "reason": "E-commerce domain detected.",
                })

        elif domain == "hr":

            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:

                plan.append({
                    "id": "hr_analysis",
                    "name": "HR Workforce Analysis",
                    "tool": "segment_by",
                    "args": {
                        "group_col": categorical_cols[0],
                        "metric_col": numeric_cols[0],
                        "agg": "mean",
                    },
                    "priority": 14,
                    "reason": "HR domain detected.",
                })

        elif domain == "marketing":

            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:

                plan.append({
                    "id": "marketing_analysis",
                    "name": "Marketing Funnel Analysis",
                    "tool": "segment_by",
                    "args": {
                        "group_col": categorical_cols[0],
                        "metric_col": numeric_cols[0],
                        "agg": "sum",
                    },
                    "priority": 14,
                    "reason": "Marketing domain detected.",
                })

        # ============================================================
        # USER QUESTION OVERRIDES
        # ============================================================

        if user_question:

            q = user_question.lower()

            # forecast/trend requests
            if any(
                word in q
                for word in [
                    "forecast",
                    "future",
                    "trend",
                    "predict",
                    "projection",
                ]
            ):

                if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:

                    plan.append({
                        "id": "forced_forecast_analysis",
                        "name": "Forecast Trend Analysis",
                        "tool": "time_series_analysis",
                        "args": {
                            "date_col": datetime_cols[0],
                            "value_col": numeric_cols[0],
                        },
                        "priority": 15,
                        "reason": f"User requested forecasting/trend analysis: {user_question}",
                    })

            # outlier requests
            if any(
                word in q
                for word in [
                    "outlier",
                    "anomaly",
                    "fraud",
                    "unusual",
                    "strange",
                ]
            ):

                if len(numeric_cols) >= 1:

                    plan.append({
                        "id": "forced_outlier_analysis",
                        "name": "Forced Outlier Detection",
                        "tool": "detect_outliers",
                        "args": {
                            "col": numeric_cols[0],
                        },
                        "priority": 15,
                        "reason": f"User requested anomaly detection: {user_question}",
                    })

            # clustering requests
            if any(
                word in q
                for word in [
                    "cluster",
                    "group",
                    "segment",
                    "cohort",
                ]
            ):

                if len(numeric_cols) >= 2:

                    plan.append({
                        "id": "forced_cluster_analysis",
                        "name": "Forced Cluster Analysis",
                        "tool": "cluster_analysis",
                        "args": {
                            "cols": numeric_cols[:6],
                            "k": 0,
                        },
                        "priority": 15,
                        "reason": f"User requested clustering/segmentation: {user_question}",
                    })

        # ============================================================
        # FINAL SYNTHESIS
        # ============================================================

        plan.append({
            "id": "final_synthesis",
            "name": "Insight Synthesis",
            "tool": "finish",
            "args": {
                "summary": "",
            },
            "priority": 100,
            "reason": "Generate final conclusions.",
        })

        # sort by priority
        plan.sort(key=lambda x: x["priority"])

        return plan