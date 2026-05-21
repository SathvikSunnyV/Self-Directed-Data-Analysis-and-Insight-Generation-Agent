from __future__ import annotations

import json
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="deep")


def _save_fig(fig, title: str = "") -> str:
    filename = f"plot_{uuid.uuid4().hex[:12]}.png"
    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


class VisualizationTools:

    def __init__(self, df_source, memory=None):
        self._source = df_source
        self.memory = memory

    @property
    def df(self) -> pd.DataFrame:
        if hasattr(self._source, "df"):
            return self._source.df
        return self._source

    # ============================================================
    # HISTOGRAM
    # ============================================================

    def histogram(
        self,
        col: str,
        bins: int = 30,
    ) -> str:
        if col not in self.df.columns:
            return json.dumps({"error": f"Column not found: {col}"})

        s = pd.to_numeric(self.df[col], errors="coerce").dropna()
        if len(s) < 2:
            return json.dumps({"error": "Not enough numeric values."})

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(s, bins=bins, color="#6d7cff", edgecolor="#0b1020", alpha=0.88)
        ax.set_title(f"Distribution of {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        mean_val = s.mean()
        ax.axvline(mean_val, color="#f7b955", linewidth=2, linestyle="--", label=f"Mean: {mean_val:.2f}")
        ax.legend()

        path = _save_fig(fig, col)
        if self.memory:
            self.memory.add_artifact(path, kind="plot", title=f"Histogram of {col}")

        return json.dumps({"status": "ok", "plot_path": path, "col": col})

    # ============================================================
    # CORRELATION HEATMAP
    # ============================================================

    def correlation_heatmap(
        self,
        cols: list[str] | None = None,
    ) -> str:
        df = self.df
        num_cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:14]

        if len(num_cols) < 2:
            return json.dumps({"error": "Need at least 2 numeric columns."})

        corr = df[num_cols].corr().round(2)
        size = max(7, len(num_cols) * 0.7)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = _save_fig(fig, "correlation_heatmap")
        if self.memory:
            self.memory.add_artifact(path, kind="plot", title="Correlation Heatmap")

        return json.dumps({"status": "ok", "plot_path": path})

    # ============================================================
    # BAR CHART
    # ============================================================

    def bar_chart(
        self,
        group_col: str,
        metric_col: str,
        agg: str = "mean",
        top_n: int = 15,
    ) -> str:
        df = self.df
        if group_col not in df.columns:
            return json.dumps({"error": f"Column not found: {group_col}"})
        if metric_col not in df.columns:
            return json.dumps({"error": f"Column not found: {metric_col}"})

        grouped = df.groupby(group_col)[metric_col].agg(agg).dropna().sort_values(ascending=False).head(top_n)
        if grouped.empty:
            return json.dumps({"error": "No data after grouping."})

        fig, ax = plt.subplots(figsize=(10, max(5, len(grouped) * 0.42)))
        colors = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(grouped)))
        grouped.plot(kind="barh", ax=ax, color=colors, edgecolor="#0b1020")
        ax.set_title(f"{agg.capitalize()} of {metric_col} by {group_col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(f"{agg}({metric_col})")
        ax.invert_yaxis()
        plt.tight_layout()

        path = _save_fig(fig, f"bar_{group_col}_{metric_col}")
        if self.memory:
            self.memory.add_artifact(path, kind="plot", title=f"Bar Chart: {metric_col} by {group_col}")

        return json.dumps({"status": "ok", "plot_path": path})

    # ============================================================
    # LINE CHART
    # ============================================================

    def line_chart(
        self,
        date_col: str,
        value_col: str,
    ) -> str:
        df = self.df.copy()
        if date_col not in df.columns:
            return json.dumps({"error": f"Column not found: {date_col}"})
        if value_col not in df.columns:
            return json.dumps({"error": f"Column not found: {value_col}"})

        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        ts = df[[date_col, value_col]].dropna().sort_values(date_col)
        if len(ts) < 2:
            return json.dumps({"error": "Not enough data."})

        series = ts.set_index(date_col)[value_col]
        monthly = series.resample("ME").mean().dropna()
        plot_series = monthly if len(monthly) >= 3 else series

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(plot_series.index, plot_series.values, color="#6d7cff", linewidth=2.2)
        ax.fill_between(plot_series.index, plot_series.values, alpha=0.14, color="#6d7cff")
        ax.set_title(f"{value_col} over Time", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel(value_col)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        path = _save_fig(fig, f"line_{date_col}_{value_col}")
        if self.memory:
            self.memory.add_artifact(path, kind="plot", title=f"Line Chart: {value_col} over Time")

        return json.dumps({"status": "ok", "plot_path": path})

    # ============================================================
    # SCATTER PLOT
    # ============================================================

    def scatter_plot(
        self,
        x_col: str,
        y_col: str,
        color_col: str | None = None,
    ) -> str:
        df = self.df
        for col in [x_col, y_col]:
            if col not in df.columns:
                return json.dumps({"error": f"Column not found: {col}"})

        plot_df = df[[x_col, y_col] + ([color_col] if color_col and color_col in df.columns else [])].dropna()
        if len(plot_df) < 2:
            return json.dumps({"error": "Not enough data."})

        fig, ax = plt.subplots(figsize=(9, 6))

        if color_col and color_col in df.columns:
            categories = plot_df[color_col].astype(str).unique()
            palette = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for i, cat in enumerate(categories):
                sub = plot_df[plot_df[color_col].astype(str) == cat]
                ax.scatter(sub[x_col], sub[y_col], label=str(cat), alpha=0.72, s=45, color=palette[i])
            ax.legend(title=color_col, fontsize=9, markerscale=1.3)
        else:
            ax.scatter(plot_df[x_col], plot_df[y_col], color="#6d7cff", alpha=0.65, s=45)

        # trend line
        try:
            x_vals = pd.to_numeric(plot_df[x_col], errors="coerce").dropna()
            y_vals = pd.to_numeric(plot_df[y_col], errors="coerce").dropna()
            common_idx = x_vals.index.intersection(y_vals.index)
            if len(common_idx) > 5:
                m, b = np.polyfit(x_vals.loc[common_idx], y_vals.loc[common_idx], 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(x_line, m * x_line + b, color="#f7b955", linewidth=1.5, linestyle="--", label="Trend")
                ax.legend()
        except Exception:
            pass

        ax.set_title(f"{x_col} vs {y_col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.tight_layout()

        path = _save_fig(fig, f"scatter_{x_col}_{y_col}")
        if self.memory:
            self.memory.add_artifact(path, kind="plot", title=f"Scatter: {x_col} vs {y_col}")

        return json.dumps({"status": "ok", "plot_path": path})

    # ============================================================
    # BOX PLOT
    # ============================================================

    def box_plot(
        self,
        cols: list[str] | None = None,
    ) -> str:
        df = self.df
        num_cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        num_cols = [c for c in num_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:10]

        if not num_cols:
            return json.dumps({"error": "No numeric columns found."})

        data = [df[c].dropna().values for c in num_cols]
        fig, ax = plt.subplots(figsize=(max(8, len(num_cols) * 1.2), 6))
        bp = ax.boxplot(data, patch_artist=True, labels=num_cols, notch=False)

        colors = plt.cm.tab10(np.linspace(0, 0.9, len(num_cols)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title("Box Plot — Distribution Overview", fontsize=14, fontweight="bold")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()

        path = _save_fig(fig, "boxplot")
        if self.memory:
            self.memory.add_artifact(path, kind="plot", title="Box Plot")

        return json.dumps({"status": "ok", "plot_path": path})

    # ============================================================
    # AUTO VISUALIZE — picks the best charts for the schema
    # ============================================================

    def auto_visualize(self, schema: dict) -> list[str]:
        """Generate a sensible default set of plots for a dataset."""
        paths: list[str] = []

        numeric_cols = schema.get("numeric_cols", [])
        categorical_cols = schema.get("categorical_cols", [])
        datetime_cols = schema.get("datetime_cols", [])

        # histogram for up to 4 numeric columns
        for col in numeric_cols[:4]:
            try:
                r = json.loads(self.histogram(col))
                if "plot_path" in r:
                    paths.append(r["plot_path"])
            except Exception:
                pass

        # correlation heatmap
        if len(numeric_cols) >= 2:
            try:
                r = json.loads(self.correlation_heatmap(numeric_cols[:12]))
                if "plot_path" in r:
                    paths.append(r["plot_path"])
            except Exception:
                pass

        # bar chart
        if categorical_cols and numeric_cols:
            try:
                r = json.loads(self.bar_chart(categorical_cols[0], numeric_cols[0]))
                if "plot_path" in r:
                    paths.append(r["plot_path"])
            except Exception:
                pass

        # line chart
        if datetime_cols and numeric_cols:
            try:
                r = json.loads(self.line_chart(datetime_cols[0], numeric_cols[0]))
                if "plot_path" in r:
                    paths.append(r["plot_path"])
            except Exception:
                pass

        # scatter for top 2 numeric cols
        if len(numeric_cols) >= 2:
            try:
                r = json.loads(self.scatter_plot(numeric_cols[0], numeric_cols[1]))
                if "plot_path" in r:
                    paths.append(r["plot_path"])
            except Exception:
                pass

        # box plot
        if numeric_cols:
            try:
                r = json.loads(self.box_plot(numeric_cols[:6]))
                if "plot_path" in r:
                    paths.append(r["plot_path"])
            except Exception:
                pass

        return paths
