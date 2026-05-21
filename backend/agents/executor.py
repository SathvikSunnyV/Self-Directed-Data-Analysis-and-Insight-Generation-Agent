from __future__ import annotations

import json
import traceback

from tools.cleaning_tools import CleaningTools
from tools.ml_tools import MLTools
from tools.report_tools import ReportTools
from tools.statistics_tools import StatisticsTools
from tools.text_tools import TextTools
from tools.time_series_tools import TimeSeriesTools
from tools.visualization_tools import VisualizationTools


class Executor:

    def __init__(
        self,
        df,
        memory,
    ):
        self.memory = memory

        # ========================================================
        # TOOL INSTANCES
        # Cleaning gets priority — its cleaned df is shared forward
        # ========================================================

        self.cleaning_tools = CleaningTools(df, memory)

        # All other tools reference the same CleaningTools instance
        # so they always work on the latest cleaned dataframe.
        self.statistics_tools = StatisticsTools(self.cleaning_tools, memory)
        self.ml_tools = MLTools(self.cleaning_tools, memory)
        self.time_series_tools = TimeSeriesTools(self.cleaning_tools, memory)
        self.text_tools = TextTools(self.cleaning_tools, memory)
        self.visualization_tools = VisualizationTools(self.cleaning_tools, memory)
        self.report_tools = ReportTools(memory)

        # ========================================================
        # TOOL MAP
        # ========================================================

        self.tool_map = {

            # ── Cleaning ─────────────────────────────────────────
            "clean_dataset":
                self.cleaning_tools.clean_dataset,

            "drop_high_missing_columns":
                self.cleaning_tools.drop_high_missing_columns,

            "remove_constant_columns":
                self.cleaning_tools.remove_constant_columns,

            # ── Statistics ───────────────────────────────────────
            "describe_dataset":
                self.statistics_tools.describe_dataset,

            "missing_values":
                self.statistics_tools.missing_values,

            "compute_statistics":
                self.statistics_tools.compute_statistics,

            "correlation_matrix":
                self.statistics_tools.correlation_matrix,

            "segment_by":
                self.statistics_tools.segment_by,

            "detect_outliers":
                self.statistics_tools.detect_outliers,

            # ── ML ───────────────────────────────────────────────
            "feature_importance":
                self.ml_tools.feature_importance,

            "cluster_analysis":
                self.ml_tools.cluster_analysis,

            # ── Time Series ──────────────────────────────────────
            "time_series_analysis":
                self.time_series_tools.time_series_analysis,

            "naive_forecast":
                self.time_series_tools.naive_forecast,

            # ── Text ─────────────────────────────────────────────
            "text_frequency":
                self.text_tools.text_frequency,

            "text_length_analysis":
                self.text_tools.text_length_analysis,

            "simple_sentiment":
                self.text_tools.simple_sentiment,

            # ── Visualization ────────────────────────────────────
            "histogram":
                self.visualization_tools.histogram,

            "correlation_heatmap":
                self.visualization_tools.correlation_heatmap,

            "bar_chart":
                self.visualization_tools.bar_chart,

            "line_chart":
                self.visualization_tools.line_chart,

            "scatter_plot":
                self.visualization_tools.scatter_plot,

            "box_plot":
                self.visualization_tools.box_plot,

            # ── Report / finish ──────────────────────────────────
            "finish":
                self._finish,
        }

    # ============================================================
    # FINISH (sentinel — produces a plain summary string)
    # ============================================================

    def _finish(self, summary: str = "") -> str:
        findings = self.memory.findings[-10:]
        if not summary:
            summary = (
                "Analysis complete. "
                + (f"Top findings: {'; '.join(findings[:3])}" if findings else "No strong findings recorded.")
            )
        return json.dumps({"status": "done", "summary": summary})

    # ============================================================
    # EXECUTE
    # ============================================================

    def execute(self, task: dict) -> dict:
        tool_name = task.get("tool", "")
        args = task.get("args", {})

        if tool_name not in self.tool_map:
            error = f"Unknown tool: {tool_name}"
            self.memory.add_error(error)
            return {"status": "error", "tool": tool_name, "error": error}

        tool_fn = self.tool_map[tool_name]

        try:
            result = tool_fn(**args)
            self.memory.act(tool_name, args, result)
            return {
                "status": "success",
                "tool": tool_name,
                "args": args,
                "result": result,
            }

        except Exception as e:
            tb = traceback.format_exc()
            error = f"{tool_name} failed: {str(e)}"
            self.memory.add_error(error)
            return {
                "status": "error",
                "tool": tool_name,
                "args": args,
                "error": error,
                "traceback": tb,
            }

    # ============================================================
    # LIST AVAILABLE TOOLS
    # ============================================================

    def available_tools(self) -> list[str]:
        return sorted(self.tool_map.keys())
