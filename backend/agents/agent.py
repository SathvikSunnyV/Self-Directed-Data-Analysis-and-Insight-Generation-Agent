"""
Autonomous Data Analysis Agent

Architecture:
- Pre-cleans data, inspects schema
- Builds a base plan from schema structure (Planner)
- Executes each step, then asks the LLM brain what to do next
  (dynamic / agile loop — not a fixed sequence)
- LLM interprets every result as business insight, not methodology
- Auto-generates visualizations
- Produces a final executive-level report
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from agents.executor import Executor
from agents.memory import AgentMemory
from agents.planner import Planner
from agents.schema import SchemaInspector
from agents.llm import HuggingFaceLLM
from tools.visualization_tools import VisualizationTools
from tools.report_tools import ReportTools

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


class AutonomousAgent:

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        hf_token: Optional[str] = None,
        max_iterations: int = 40,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.schema_inspector = SchemaInspector()
        self.planner = Planner()
        self.llm = HuggingFaceLLM(model=model, token=hf_token)

    # ============================================================
    # SCHEMA SUMMARY STRING (for LLM context)
    # ============================================================

    def _schema_summary(self, schema: dict) -> str:
        return (
            f"File: {schema.get('filename', '?')} | "
            f"{schema.get('rows', '?')} rows × {schema.get('cols', '?')} cols | "
            f"Domain: {schema.get('domain', 'general')} | "
            f"Numeric: {', '.join(schema.get('numeric_cols', [])[:8]) or 'none'} | "
            f"Categorical: {', '.join(schema.get('categorical_cols', [])[:8]) or 'none'} | "
            f"Datetime: {', '.join(schema.get('datetime_cols', [])[:4]) or 'none'} | "
            f"Targets: {', '.join(schema.get('potential_targets', [])[:4]) or 'none'}"
        )

    # ============================================================
    # INTERPRET A STEP RESULT → BUSINESS INSIGHT
    # ============================================================

    def _interpret(
        self,
        step_name: str,
        raw_result: str,
        schema: dict,
    ) -> str:
        """
        Produce a business-insight interpretation of the tool output.
        Uses LLM when available; falls back to a smart deterministic extractor.
        """
        deterministic = self._extract_insight(step_name, raw_result, schema)

        if not self.llm.is_available():
            return deterministic

        context = (
            f"Dataset: {self._schema_summary(schema)}\n\n"
            f"Analysis performed: {step_name}\n\n"
            f"Raw statistical output:\n{raw_result[:2500]}"
        )
        task = (
            "In 2-4 sentences, describe what this result REVEALS about the data. "
            "Focus on: what trend or pattern exists, which groups or variables stand out, "
            "and what this means for decision-making. "
            "Do NOT say 'the analysis shows' or 'the tool computed'. "
            "Speak directly about the data and what it tells us."
        )

        try:
            reply = self.llm.think(context, task)
            if reply and "LLM_ERROR" not in reply and len(reply) > 20:
                return reply
        except Exception:
            pass

        return deterministic

    def _extract_insight(self, step_name: str, raw_result: str, schema: dict) -> str:
        """
        Deterministic insight extractor — reads the JSON output and produces
        meaningful business-language descriptions of what the numbers mean.
        """
        import json as _json

        if not raw_result or "not found" in raw_result.lower():
            return f"Insufficient data for {step_name.lower()}."
        if raw_result.startswith("ERROR"):
            return f"{step_name} could not complete: {raw_result[:120]}"

        try:
            data = _json.loads(raw_result)
        except Exception:
            # Plain text (e.g. describe() output)
            lines = [l.strip() for l in raw_result.splitlines() if l.strip()]
            if len(lines) > 2:
                return f"Statistical summary shows: {' | '.join(lines[1:4])}."
            return raw_result[:200]

        domain = schema.get("domain", "general")
        parts: list[str] = []

        # ── Data Cleaning ─────────────────────────────────────
        if isinstance(data, dict) and "duplicates_removed" in data:
            dups = data["duplicates_removed"]
            missing_fixed = data.get("missing_before", 0) - data.get("missing_after", 0)
            if dups == 0 and missing_fixed == 0:
                parts.append("The dataset is already clean with no duplicates or missing values.")
            else:
                if dups > 0:
                    parts.append(f"{dups} duplicate records were removed, suggesting data quality issues in the collection pipeline.")
                if missing_fixed > 0:
                    parts.append(f"{missing_fixed} missing values were imputed, which may introduce slight bias in downstream analysis.")

        # ── Missing Values ────────────────────────────────────
        elif isinstance(data, list) and data and "missing_pct" in (data[0] if data else {}):
            problematic = [(r["column"], r["missing_pct"]) for r in data if r.get("missing_pct", 0) > 0]
            if not problematic:
                parts.append("The dataset is complete — no missing values in any column, indicating high data quality.")
            else:
                worst = problematic[0]
                parts.append(
                    f"'{worst[0]}' has the most missing data ({worst[1]}%), which could skew analysis of that dimension. "
                    + (f"Columns {', '.join(c for c,_ in problematic[:3])} require attention before relying on them for decisions." if len(problematic) > 1 else "")
                )

        # ── Correlation ───────────────────────────────────────
        elif isinstance(data, dict) and "top_correlations" in data and data["top_correlations"]:
            pairs = data["top_correlations"]
            strong = [p for p in pairs if abs(p["corr"]) > 0.7]
            moderate = [p for p in pairs if 0.4 < abs(p["corr"]) <= 0.7]
            if strong:
                p = strong[0]
                direction = "increases" if p["corr"] > 0 else "decreases"
                parts.append(
                    f"Strong relationship: as {p['col1']} increases, {p['col2']} strongly {direction} (r={p['corr']}). "
                    f"This suggests {'they may be driven by the same underlying factor' if p['corr'] > 0 else 'a trade-off between these dimensions'}."
                )
                if len(strong) > 1:
                    p2 = strong[1]
                    parts.append(f"Also notable: {p2['col1']} and {p2['col2']} are tightly linked (r={p2['corr']}).")
            elif moderate:
                p = moderate[0]
                parts.append(f"Moderate relationship between {p['col1']} and {p['col2']} (r={p['corr']}) — related but with other influencing factors.")
            else:
                parts.append("Variables appear largely independent — no single factor strongly predicts another.")

        # ── Outliers ──────────────────────────────────────────
        elif isinstance(data, dict) and "iqr_outlier_count" in data:
            col = data["column"]
            n = data["iqr_outlier_count"]
            lb, ub = data.get("lower_bound", "?"), data.get("upper_bound", "?")
            if n == 0:
                parts.append(f"'{col}' values are uniformly distributed with no extreme outliers, indicating consistent behavior across records.")
            else:
                pct = round(n / schema.get("rows", 1) * 100, 1)
                parts.append(
                    f"{n} records ({pct}% of data) in '{col}' fall outside the expected range [{lb} – {ub}]. "
                    f"{'These could represent high-value opportunities or data entry errors worth reviewing.' if pct < 5 else 'This high outlier rate suggests significant variability or potential sub-groups in the data.'}"
                )

        # ── Segmentation ─────────────────────────────────────
        elif isinstance(data, dict) and "top_10" in data and data.get("top_10"):
            group_col = data.get("group_col", "")
            metric_col = data.get("metric_col", "")
            agg = data.get("aggregation", "mean")
            top = data["top_10"]
            bottom = data.get("bottom_5", [])
            if top:
                best_vals = list(top[0].values())
                worst_vals = list(bottom[-1].values()) if bottom else None
                parts.append(
                    f"'{best_vals[0]}' leads all segments with {agg}({metric_col}) = {round(float(best_vals[1]), 2) if isinstance(best_vals[1], (int, float)) else best_vals[1]}."
                )
                if worst_vals:
                    ratio = round(float(best_vals[1]) / float(worst_vals[1]), 1) if float(worst_vals[1]) != 0 else "∞"
                    parts.append(f"The top segment outperforms the bottom by {ratio}×, indicating significant disparity across {group_col} groups.")

        # ── Time Series ───────────────────────────────────────
        elif isinstance(data, dict) and "trend" in data and "value_column" in data:
            col = data["value_column"]
            trend = data["trend"]
            chg = data.get("total_change_pct")
            vol = data.get("volatility_pct")
            peak = data.get("peak", {})
            parts.append(
                f"'{col}' has been {trend} over the period"
                + (f", changing by {chg}%" if chg is not None else "")
                + ("." if chg is None else f" — {'healthy growth' if chg > 0 else 'a declining trend that warrants attention'}.")
            )
            if vol and vol > 20:
                parts.append(f"High volatility ({vol}%) indicates unstable or seasonal behavior rather than a steady trend.")
            if peak.get("date"):
                parts.append(f"Peak was reached on {peak['date']} ({peak.get('value', '?')}), after which the trend reversed.")

        # ── Feature Importance ───────────────────────────────
        elif isinstance(data, dict) and "top_features" in data and data["top_features"]:
            target = data.get("target_column", "target")
            feats = list(data["top_features"].items())
            top1, top2 = feats[0], feats[1] if len(feats) > 1 else None
            parts.append(
                f"'{top1[0]}' is the strongest predictor of '{target}' (importance={top1[1]:.3f}), "
                f"meaning it carries the most information about {target} variation."
            )
            if top2:
                parts.append(
                    f"'{top2[0]}' is the second most important driver (importance={top2[1]:.3f}). "
                    f"Together, these two features likely explain most of the variance in '{target}'."
                )

        # ── Clustering ────────────────────────────────────────
        elif isinstance(data, dict) and "cluster_sizes" in data:
            k = data["k"]
            sizes = data["cluster_sizes"]
            profiles = data.get("profiles", {})
            largest = max(sizes, key=sizes.get)
            smallest = min(sizes, key=sizes.get)
            parts.append(
                f"The data naturally groups into {k} clusters. "
                f"Cluster {largest} is the dominant group ({sizes[largest]} records), "
                f"while cluster {smallest} is the smallest ({sizes[smallest]} records) — potentially a niche or outlier segment."
            )
            # Describe one cluster meaningfully
            if profiles:
                cid = str(largest)
                if cid in profiles:
                    means = profiles[cid].get("means", {})
                    top_cols = list(means.items())[:2]
                    if top_cols:
                        desc = ", ".join(f"{c}≈{round(v,1)}" for c, v in top_cols)
                        parts.append(f"The dominant cluster is characterised by {desc}.")

        # ── Statistics (describe / variance / normality) ──────
        elif isinstance(data, dict) and not parts:
            # Generic: try to extract something meaningful
            keys = list(data.keys())
            if keys:
                first_key = keys[0]
                first_val = data[first_key]
                if isinstance(first_val, dict):
                    stats_str = ", ".join(f"{k}={round(v,2)}" for k, v in list(first_val.items())[:4] if isinstance(v, (int, float)))
                    parts.append(f"'{first_key}' statistics: {stats_str}.")

        if not parts:
            parts.append(f"{step_name} completed — results recorded.")

        return " ".join(parts)

    # ============================================================
    # DYNAMIC REPLANNING — ask LLM what to do next
    # ============================================================

    def _dynamic_steps(
        self,
        schema: dict,
        completed_tools: list[str],
        findings: list[str],
        executor: Executor,
        user_question: str,
    ) -> list[dict]:
        """
        After the base plan, ask the LLM brain whether additional analyses
        would reveal more value. Returns up to 3 new steps with their args.
        """
        if not self.llm.is_available():
            return []

        available_tools = [t for t in executor.available_tools() if t != "finish"]
        schema_summary = self._schema_summary(schema)

        suggested = self.llm.decide_next_steps(
            schema_summary=schema_summary,
            completed_steps=completed_tools,
            key_findings=findings,
            available_tools=available_tools,
            user_question=user_question or "",
        )

        # Validate suggested steps — make sure args reference real columns
        numeric_cols = schema.get("numeric_cols", [])
        categorical_cols = schema.get("categorical_cols", [])
        datetime_cols = schema.get("datetime_cols", [])
        all_cols = set(numeric_cols + categorical_cols + datetime_cols + schema.get("text_cols", []))

        valid = []
        for s in suggested:
            tool = s.get("tool", "")
            args = s.get("args", {})
            if tool not in executor.available_tools():
                continue
            # Fix any column references that don't exist
            fixed_args = {}
            for k, v in args.items():
                if isinstance(v, str) and k.endswith("_col") and v not in all_cols:
                    # Try to find a matching column
                    match = next((c for c in all_cols if v.lower() in c.lower()), None)
                    if match:
                        fixed_args[k] = match
                    else:
                        fixed_args[k] = (numeric_cols[0] if numeric_cols else None)
                elif isinstance(v, list):
                    # Filter to only real columns
                    fixed_args[k] = [c for c in v if c in all_cols] or (numeric_cols[:4] if numeric_cols else [])
                else:
                    fixed_args[k] = v
            s["args"] = fixed_args
            valid.append(s)

        return valid

    # ============================================================
    # FINAL REPORT
    # ============================================================

    def _build_report(
        self,
        filename: str,
        memory: AgentMemory,
        schema: dict,
        step_results: dict,
        user_question: str,
    ) -> str:
        lines: list[str] = []
        lines.append("# Data Analysis Report")
        lines.append("")
        lines.append(f"**File:** {filename}")
        lines.append(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
        if user_question:
            lines.append(f"**Question:** {user_question}")
        lines.append("")

        # Dataset overview
        s = schema
        lines.append("## Dataset Overview")
        lines.append(f"- **Rows:** {s.get('rows', '?')} | **Columns:** {s.get('cols', '?')}")
        lines.append(f"- **Domain:** {s.get('domain', 'general')}")
        lines.append(f"- **Numeric:** {', '.join(s.get('numeric_cols', [])[:10]) or 'None'}")
        lines.append(f"- **Categorical:** {', '.join(s.get('categorical_cols', [])[:10]) or 'None'}")
        lines.append(f"- **Datetime:** {', '.join(s.get('datetime_cols', [])[:6]) or 'None'}")
        lines.append(f"- **Potential targets:** {', '.join(s.get('potential_targets', [])[:6]) or 'None'}")
        lines.append(f"- **Missing values:** {s.get('total_missing', 0)} | **Duplicate rows:** {s.get('duplicate_rows', 0)}")
        lines.append("")

        # Executive summary (LLM or deterministic)
        lines.append("## Executive Summary")
        if self.llm.is_available() and memory.findings:
            try:
                summary = self.llm.summarize_results(
                    schema_summary=self._schema_summary(schema),
                    findings=memory.findings,
                    user_question=user_question or "",
                )
                if summary and "LLM_ERROR" not in summary and len(summary) > 30:
                    lines.append(summary)
                else:
                    lines.append(self._deterministic_summary(memory))
            except Exception:
                lines.append(self._deterministic_summary(memory))
        else:
            lines.append(self._deterministic_summary(memory))
        lines.append("")

        # Key findings
        lines.append("## Key Findings")
        if memory.findings:
            for f in memory.findings:
                # Strip step-name prefix if present (e.g. "Correlation Analysis: ...")
                text = f.split(": ", 1)[-1] if ": " in f else f
                lines.append(f"- {text}")
        else:
            lines.append("- No strong patterns identified.")
        lines.append("")

        # Analysis steps
        lines.append("## Analysis Details")
        for step_id, sr in step_results.items():
            if sr.get("tool") == "finish":
                continue
            lines.append(f"### {sr['name']}")
            interp = sr.get("interpretation", "")
            if interp and "could not complete" not in interp and "completed — results" not in interp:
                lines.append(interp)
            raw = sr.get("raw_result", "")
            if raw and not raw.startswith("ERROR") and len(raw) < 2000:
                lines.append("")
                lines.append("```")
                lines.append(raw[:1500])
                lines.append("```")
            lines.append("")

        # Artifacts
        if memory.artifacts:
            lines.append("## Generated Visualizations")
            for a in memory.artifacts:
                if a.get("kind") == "plot":
                    lines.append(f"- {a.get('title', 'Plot')}: `{a.get('path', '')}`")
            lines.append("")

        # Errors
        errors = [e for e in memory.errors if "Pre-clean" not in e]
        if errors:
            lines.append("## Notes")
            for e in errors[:5]:
                lines.append(f"- {e}")
            lines.append("")

        return "\n".join(lines)

    def _deterministic_summary(self, memory: AgentMemory) -> str:
        findings = [f.split(": ", 1)[-1] if ": " in f else f for f in memory.findings if "ERROR" not in f]
        if not findings:
            return "The dataset was analysed. No strong patterns were detected in the available data."
        top = findings[:5]
        return "Key patterns identified: " + " | ".join(top)

    # ============================================================
    # MAIN ANALYZE METHOD
    # ============================================================

    def analyze(
        self,
        df: pd.DataFrame,
        filename: str = "dataset.csv",
        user_question: Optional[str] = None,
        progress_cb: Optional[Callable[[str, str, int, int], None]] = None,
    ) -> dict:

        memory = AgentMemory()
        memory.observe(f"Starting analysis for {filename}")

        # ── Step 1: Executor + pre-clean ──────────────────────
        executor = Executor(df, memory)

        memory.observe("Pre-cleaning before schema inspection…")
        try:
            executor.execute({"id": "_pre_clean", "tool": "clean_dataset", "args": {}})
        except Exception as e:
            memory.add_error(f"Pre-clean failed: {e}")

        # ── Step 2: Schema on cleaned df ──────────────────────
        clean_df = executor.cleaning_tools.df
        schema = self.schema_inspector.inspect(clean_df, filename=filename)
        memory.schema = schema
        memory.observe(
            f"Schema: {schema['rows']} rows × {schema['cols']} cols | domain={schema['domain']}"
        )

        # ── Step 3: Base plan (no clean step — already done) ──
        base_plan = self.planner.create_plan(schema, user_question=user_question)
        base_plan = [s for s in base_plan if s.get("tool") not in ("clean_dataset", "finish")]
        memory.plan = base_plan

        # Estimate total (base + up to 6 dynamic)
        estimated_total = len(base_plan) + 6
        step_results: dict[str, dict] = {}
        completed_tools: list[str] = []
        dynamic_done = False
        idx = 0

        def _run_step(step: dict, step_idx: int, total_est: int) -> str:
            """Execute one step, interpret it, record findings. Returns raw_result."""
            nonlocal idx
            tool = step.get("tool", "")
            name = step.get("name", tool)

            if progress_cb:
                try:
                    progress_cb("agent", f"▶ {name}", step_idx, total_est)
                except Exception:
                    pass

            memory.observe(f"Step {step_idx}: {name}")
            exec_result = executor.execute(step)

            raw = (
                exec_result.get("result", "")
                if exec_result["status"] == "success"
                else f"ERROR: {exec_result.get('error', '')}"
            )

            interpretation = self._interpret(name, str(raw), schema)
            memory.think(interpretation)

            if exec_result["status"] == "success":
                # Only add meaningful findings (not process descriptions)
                if interpretation and "completed — results" not in interpretation and "could not complete" not in interpretation:
                    memory.add_finding(f"{name}: {interpretation}")
            else:
                memory.add_error(str(raw))

            step_results[step.get("id", f"step_{step_idx}")] = {
                "name": name,
                "tool": tool,
                "args": step.get("args", {}),
                "raw_result": str(raw)[:3000],
                "interpretation": interpretation,
            }
            completed_tools.append(tool)
            return str(raw)

        # ── Step 4: Execute base plan ──────────────────────────
        for step in base_plan:
            idx += 1
            if idx > self.max_iterations:
                break
            _run_step(step, idx, estimated_total)

        # ── Step 5: Dynamic extension — LLM decides what's next ─
        if not dynamic_done and len(completed_tools) > 0:
            if progress_cb:
                try:
                    progress_cb("agent", "🧠 Thinking: what else would be valuable?", idx, estimated_total)
                except Exception:
                    pass

            memory.observe("Asking LLM brain for additional analyses…")
            extra_steps = self._dynamic_steps(
                schema=schema,
                completed_tools=completed_tools,
                findings=[f for f in memory.findings if "ERROR" not in f],
                executor=executor,
                user_question=user_question or "",
            )

            dynamic_done = True
            if extra_steps:
                memory.observe(f"LLM suggested {len(extra_steps)} additional steps")
                estimated_total = idx + len(extra_steps) + 2

                for i, step in enumerate(extra_steps):
                    if idx >= self.max_iterations:
                        break
                    idx += 1
                    # Build proper step dict
                    full_step = {
                        "id": f"dynamic_{i}_{step['tool']}",
                        "name": f"[Dynamic] {step['tool'].replace('_', ' ').title()}",
                        "tool": step["tool"],
                        "args": step.get("args", {}),
                        "reason": step.get("reason", "LLM-suggested"),
                    }
                    if progress_cb:
                        try:
                            progress_cb(
                                "agent",
                                f"🔍 {full_step['name']} ({step.get('reason', '')})",
                                idx,
                                estimated_total,
                            )
                        except Exception:
                            pass
                    _run_step(full_step, idx, estimated_total)
            else:
                memory.observe("LLM: no additional analyses needed")

        # ── Step 6: Visualizations ────────────────────────────
        if progress_cb:
            try:
                progress_cb("agent", "📊 Generating visualizations", idx, estimated_total)
            except Exception:
                pass

        try:
            viz_df = executor.cleaning_tools.df
            viz = VisualizationTools(viz_df, memory)
            plot_paths = viz.auto_visualize(schema)
            memory.observe(f"Generated {len(plot_paths)} plots")
        except Exception as e:
            memory.add_error(f"Visualization failed: {e}")
            plot_paths = []

        # ── Step 7: Report ────────────────────────────────────
        if progress_cb:
            try:
                progress_cb("agent", "📝 Building report", idx, estimated_total)
            except Exception:
                pass

        report_md = self._build_report(filename, memory, schema, step_results, user_question or "")
        report_tools = ReportTools(memory)
        report_path = report_tools.save_report(report_md)

        final_summary = memory.summary()
        final_summary["report_path"] = report_path

        return {
            "status": "complete",
            "filename": filename,
            "schema": schema,
            "plan": base_plan,
            "results": step_results,
            "plot_paths": plot_paths,
            "memory": {
                "summary": final_summary,
                "observations": memory.observations[-30:],
                "thoughts": memory.thoughts[-30:],
                "findings": memory.findings,
                "errors": memory.errors,
                "artifacts": memory.artifacts,
            },
            "report_markdown": report_md,
            "report_path": report_path,
        }
