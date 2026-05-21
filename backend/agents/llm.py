"""
HuggingFace LLM backend.

Uses the HuggingFace Serverless Inference API via huggingface_hub.
No local model needed — works with any HF token (free tier included).

Supported models (chat-capable, free serverless):
  - meta-llama/Llama-3.2-3B-Instruct   (default — fast, capable)
  - meta-llama/Llama-3.1-8B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - mistralai/Mistral-7B-Instruct-v0.3
  - microsoft/Phi-3.5-mini-instruct
  - google/gemma-3-4b-it
  - HuggingFaceH4/zephyr-7b-beta        (no token needed)
  - NousResearch/Hermes-3-Llama-3.1-8B

Set HF_TOKEN env var for gated models (Llama etc.).
Leave unset to use unauthenticated access for open models.
"""
from __future__ import annotations

import json
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# Default model — fast and capable on free tier
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Fallback model that works without authentication
FALLBACK_MODEL = "HuggingFaceH4/zephyr-7b-beta"


class HuggingFaceLLM:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        token: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ):
        self.model = model
        self.token = token or os.environ.get("HF_TOKEN") or None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client = None
        self._available: Optional[bool] = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(token=self.token)
        return self._client

    # ============================================================
    # AVAILABILITY CHECK
    # ============================================================

    def is_available(self) -> bool:
        """Check connectivity + that the model responds."""
        if self._available is not None:
            return self._available
        try:
            client = self._get_client()
            # Lightweight ping
            resp = client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model=self.model,
                max_tokens=5,
            )
            self._available = bool(resp.choices[0].message.content)
        except Exception:
            # Try fallback model silently
            try:
                client = self._get_client()
                resp = client.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=FALLBACK_MODEL,
                    max_tokens=5,
                )
                if resp.choices[0].message.content:
                    self.model = FALLBACK_MODEL
                    self._available = True
                else:
                    self._available = False
            except Exception:
                self._available = False
        return self._available

    # ============================================================
    # LIST AVAILABLE MODELS
    # ============================================================

    def list_models(self) -> list[str]:
        return [
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3.5-mini-instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "google/gemma-3-4b-it",
        ]

    # ============================================================
    # CORE GENERATE
    # ============================================================

    def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        try:
            client = self._get_client()
            resp = client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=max_tokens or self.max_new_tokens,
                temperature=temperature or self.temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"LLM_ERROR: {str(e)}"

    # ============================================================
    # THINK — interpret a single step result
    # ============================================================

    def think(self, context: str, task: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert data analyst. "
                    "Your role is to interpret statistical results and explain "
                    "what they reveal about the data — the trends, patterns, anomalies, "
                    "and business implications. "
                    "Never describe the process or methodology. "
                    "Always describe what the numbers MEAN and what they REVEAL about the data."
                ),
            },
            {
                "role": "user",
                "content": f"{context}\n\nTask: {task}",
            },
        ]
        return self.generate(messages, max_tokens=350)

    # ============================================================
    # DECIDE NEXT STEP — dynamic planning
    # ============================================================

    def decide_next_steps(
        self,
        schema_summary: str,
        completed_steps: list[str],
        key_findings: list[str],
        available_tools: list[str],
        user_question: str,
    ) -> list[dict]:
        """
        Given what we know so far, decide what additional analysis steps
        would reveal the most value. Returns a list of step dicts.
        """
        findings_text = "\n".join(f"- {f}" for f in key_findings[-12:]) or "None yet."
        done_text = ", ".join(completed_steps[-15:]) or "None."
        tools_text = "\n".join(f"- {t}" for t in available_tools)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous data analysis agent. "
                    "Your job is to decide what additional analyses would reveal the most value "
                    "given what has already been found. "
                    "You must respond ONLY with a valid JSON array. No explanation, no markdown, no extra text.\n"
                    "Each item: {\"tool\": \"<tool_name>\", \"args\": {<args>}, \"reason\": \"<one sentence why>\"}\n"
                    "Return an EMPTY array [] if no further analysis is needed."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DATASET SCHEMA:\n{schema_summary}\n\n"
                    f"USER QUESTION: {user_question or 'General analysis'}\n\n"
                    f"ALREADY COMPLETED: {done_text}\n\n"
                    f"KEY FINDINGS SO FAR:\n{findings_text}\n\n"
                    f"AVAILABLE TOOLS:\n{tools_text}\n\n"
                    "Based on the findings above, what 1-3 additional analyses would reveal the most insight? "
                    "Only suggest tools that would genuinely add value given what's already been found. "
                    "Respond with ONLY a JSON array."
                ),
            },
        ]

        raw = self.generate(messages, max_tokens=600, temperature=0.1)

        # Parse JSON array from response
        try:
            # Strip markdown code fences if present
            clean = raw.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            # Find the array
            start = clean.find("[")
            end = clean.rfind("]")
            if start != -1 and end != -1:
                steps = json.loads(clean[start:end + 1])
                # Validate structure
                valid = []
                for s in steps:
                    if isinstance(s, dict) and "tool" in s:
                        valid.append({
                            "tool": str(s["tool"]),
                            "args": s.get("args", {}),
                            "reason": s.get("reason", "LLM-suggested step"),
                        })
                return valid[:3]  # cap at 3 extra steps
        except Exception:
            pass
        return []

    # ============================================================
    # SUMMARIZE RESULTS — final executive summary
    # ============================================================

    def summarize_results(
        self,
        schema_summary: str,
        findings: list[str],
        user_question: str,
    ) -> str:
        findings_text = "\n".join(f"- {f}" for f in findings)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior data analyst writing an executive summary. "
                    "Focus entirely on what the data reveals — the patterns, trends, anomalies, and implications. "
                    "Do NOT describe methods, tools, or processes. "
                    "Write in clear business language. Be specific about numbers where available. "
                    "Structure: 1) Key pattern/trend, 2) Notable anomalies or risks, 3) Actionable recommendations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"DATASET: {schema_summary}\n\n"
                    f"USER QUESTION: {user_question or 'General analysis'}\n\n"
                    f"FINDINGS:\n{findings_text}\n\n"
                    "Write a 3-5 sentence executive summary focusing on what matters most."
                ),
            },
        ]
        return self.generate(messages, max_tokens=500, temperature=0.2)

    # ============================================================
    # GENERATE JSON (generic)
    # ============================================================

    def generate_json(self, prompt: str) -> Optional[dict]:
        messages = [
            {"role": "system", "content": "Respond ONLY with valid JSON. No explanation."},
            {"role": "user", "content": prompt},
        ]
        raw = self.generate(messages, max_tokens=400, temperature=0)
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
        except Exception:
            pass
        return None