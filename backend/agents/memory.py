from __future__ import annotations

from datetime import datetime
from typing import Any


class AgentMemory:
    def __init__(self):
        self.observations: list[str] = []
        self.thoughts: list[str] = []
        self.actions: list[dict] = []
        self.findings: list[str] = []
        self.errors: list[str] = []
        self.artifacts: list[dict] = []
        self.plan: list[dict] = []
        self.schema: dict[str, Any] = {}
        self.started_at = datetime.now()
        self.iteration = 0

    def observe(self, msg: str):
        self.observations.append(str(msg))

    def think(self, msg: str):
        self.thoughts.append(str(msg))

    def act(self, tool: str, args: dict, result: Any):
        self.actions.append(
            {
                "tool": tool,
                "args": args,
                "result": str(result)[:1200],
            }
        )

    def add_finding(self, msg: str):
        msg = str(msg)
        if msg not in self.findings:
            self.findings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(str(msg))

    def add_artifact(self, path: str, kind: str, title: str = ""):
        self.artifacts.append(
            {
                "path": path,
                "kind": kind,
                "title": title,
            }
        )

    def build_context(self) -> dict:
        return {
            "iteration": self.iteration,
            "observations": self.observations[-10:],
            "thoughts": self.thoughts[-10:],
            "actions": self.actions[-10:],
            "findings": self.findings[-10:],
            "errors": self.errors[-10:],
            "artifacts": self.artifacts[-10:],
            "plan": self.plan[-10:],
            "schema": self.schema,
        }

    def context_window(self) -> str:
        lines = []
        lines.append(f"ITERATION: {self.iteration}")

        if self.schema:
            s = self.schema
            lines.append(
                f"DATASET: {s.get('filename', '?')} | {s.get('rows', '?')} rows x {s.get('cols', '?')} cols"
            )
            lines.append(
                f"DOMAIN: {s.get('domain', 'general')} | "
                f"numeric={len(s.get('numeric_cols', []))} "
                f"categorical={len(s.get('categorical_cols', []))} "
                f"datetime={len(s.get('datetime_cols', []))} "
                f"text={len(s.get('text_cols', []))}"
            )

        if self.plan:
            lines.append("PLAN:")
            for i, step in enumerate(self.plan[-12:], 1):
                if isinstance(step, dict):
                    lines.append(f"  {i}. {step.get('name', '?')} ({step.get('id', '?')})")
                else:
                    lines.append(f"  {i}. {step}")

        if self.findings:
            lines.append("FINDINGS:")
            for f in self.findings[-10:]:
                lines.append(f"  - {f}")

        if self.actions:
            lines.append("RECENT ACTIONS:")
            for a in self.actions[-5:]:
                lines.append(
                    f"  - {a['tool']}({str(a['args'])[:120]}) -> {str(a['result'])[:200]}"
                )

        if self.errors:
            lines.append("RECENT ERRORS:")
            for e in self.errors[-3:]:
                lines.append(f"  - {e}")

        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "iterations": self.iteration,
            "observations": len(self.observations),
            "thoughts": len(self.thoughts),
            "actions": len(self.actions),
            "findings": len(self.findings),
            "errors": len(self.errors),
            "artifacts": len(self.artifacts),
            "elapsed_seconds": round((datetime.now() - self.started_at).total_seconds(), 2),
        }