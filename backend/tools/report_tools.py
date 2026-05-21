from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)


class ReportTools:

    def __init__(self, memory=None):

        self.memory = memory

    # ============================================================
    # MARKDOWN REPORT
    # ============================================================

    def generate_markdown_report(
        self,
        title: str,
        findings: list[str],
        thoughts: list[str],
        artifacts: list[dict],
    ) -> str:

        report = []

        report.append(f"# {title}")
        report.append("")

        report.append(
            f"Generated: "
            f"{datetime.now().isoformat()}"
        )

        report.append("")

        # ========================================================
        # FINDINGS
        # ========================================================

        report.append("## Key Findings")
        report.append("")

        if findings:

            for finding in findings:

                report.append(
                    f"- {finding}"
                )

        else:

            report.append(
                "- No findings generated"
            )

        report.append("")

        # ========================================================
        # REASONING
        # ========================================================

        report.append("## Agent Reasoning")
        report.append("")

        if thoughts:

            for thought in thoughts:

                report.append(
                    f"- {thought}"
                )

        else:

            report.append(
                "- No reasoning generated"
            )

        report.append("")

        # ========================================================
        # ARTIFACTS
        # ========================================================

        report.append("## Generated Artifacts")
        report.append("")

        if artifacts:

            for artifact in artifacts:

                title = artifact.get(
                    "title",
                    "Artifact",
                )

                path = artifact.get(
                    "path",
                    "",
                )

                report.append(
                    f"- {title}: `{path}`"
                )

        else:

            report.append(
                "- No artifacts generated"
            )

        report.append("")

        return "\n".join(report)

    # ============================================================
    # SAVE REPORT
    # ============================================================

    def save_report(
        self,
        markdown: str,
    ) -> str:

        filename = (
            f"report_{uuid.uuid4().hex}.md"
        )

        path = OUTPUTS_DIR / filename

        path.write_text(
            markdown,
            encoding="utf-8",
        )

        if self.memory:

            self.memory.add_artifact(
                str(path),
                kind="report",
                title="Analysis Report",
            )

        return str(path)

    # ============================================================
    # JSON REPORT
    # ============================================================

    def save_json_report(
        self,
        data: dict,
    ) -> str:

        filename = (
            f"report_{uuid.uuid4().hex}.json"
        )

        path = OUTPUTS_DIR / filename

        path.write_text(
            json.dumps(
                data,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        if self.memory:

            self.memory.add_artifact(
                str(path),
                kind="json_report",
                title="JSON Analysis Report",
            )

        return str(path)