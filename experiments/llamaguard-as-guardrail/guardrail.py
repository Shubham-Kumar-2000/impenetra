"""
LlamaGuard-as-Guardrail: Injectable Guardrail Implementation.

Implements BaseGuardrail by evaluating incoming prompts against
Meta's LlamaGuard-3 model hosted locally on Ollama.
"""

import time
import requests
from impenetra.core.guardrail import BaseGuardrail, GuardrailResult

OLLAMA_BASE = "http://localhost:11434"


def _detect_guard_model() -> str:
    """Auto-detect available llama-guard3 model tag from Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        for candidate in ["llama-guard3:latest", "llama-guard3:8b", "llama-guard3:1b"]:
            if candidate in models:
                return candidate
        for m in models:
            if "llama-guard3" in m:
                return m
    except Exception:
        pass
    return "llama-guard3:latest"


class LlamaGuardGuardrail(BaseGuardrail):
    """
    Input guardrail using Meta's LlamaGuard-3 via Ollama.
    Injectable into Impenetra's create_app(guardrail=...).
    """

    def __init__(self, model_name: str | None = None, base_url: str = OLLAMA_BASE):
        self._base_url = base_url.rstrip("/")
        self._model = model_name or _detect_guard_model()
        self._name = f"llamaguard_{self._model}"

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str) -> GuardrailResult:
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": text}],
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            classification = resp.json()["message"]["content"].strip().lower()
        except requests.RequestException as e:
            elapsed = (time.perf_counter() - start) * 1000
            # Fail-open with warning detail
            return GuardrailResult(
                allowed=True,
                confidence=0.0,
                category="guardrail_error",
                details=f"Guardrail error: {e}",
                latency_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        is_safe = classification.startswith("safe")
        category = "" if is_safe else classification

        return GuardrailResult(
            allowed=is_safe,
            confidence=0.95 if is_safe else 0.99,
            category=category,
            details=f"LlamaGuard verdict: {classification}",
            latency_ms=elapsed,
        )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]
