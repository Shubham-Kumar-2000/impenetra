"""
Experiment 1: LlamaGuard-as-Guardrail Server
=============================================
Uses llama-guard3:8b (via Ollama) as an input guardrail classifier.
Unsafe prompts are blocked before reaching the main LLM.
Safe prompts are forwarded to llama3.1:8b-instruct-q8_0 for a real response.

Architecture:
    Garak → FastAPI (8000) → llama-guard3:8b (classify)
                           → llama3.1:8b-instruct-q8_0 (respond, if safe)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import uuid
import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from impenetra.core.guardrail import BaseGuardrail, GuardrailResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE = "http://localhost:11434"
MAIN_MODEL = "llama3.1:8b-instruct-q8_0"
DEFAULT_REFUSAL = "I cannot fulfill this request because it violates safety guidelines."


def _detect_guard_model() -> str:
    """Auto-detect available llama-guard3 tag from Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        for candidate in ["llama-guard3:8b", "llama-guard3:1b", "llama-guard3:latest"]:
            if candidate in models:
                return candidate
        # Fallback: any llama-guard3 variant
        for m in models:
            if "llama-guard3" in m:
                return m
    except Exception:
        pass
    return "llama-guard3:latest"  # default


GUARD_MODEL = _detect_guard_model()


# ---------------------------------------------------------------------------
# LlamaGuard Guardrail Implementation
# ---------------------------------------------------------------------------
class LlamaGuardGuardrail(BaseGuardrail):
    """
    Input guardrail using Meta's LlamaGuard3 via Ollama.

    Sends the user prompt to llama-guard3:8b for safety classification.
    Returns 'safe' or 'unsafe S<n>' where S<n> is the hazard category code.
    See: https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/
    """

    @property
    def name(self) -> str:
        return "llama_guard_ollama"

    def check(self, text: str) -> GuardrailResult:
        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": GUARD_MODEL,
                    "messages": [{"role": "user", "content": text}],
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            classification = resp.json()["message"]["content"].strip().lower()
        except requests.RequestException as e:
            elapsed = (time.perf_counter() - start) * 1000
            # Fail-open: if guardrail is unavailable, log error but allow
            return GuardrailResult(
                allowed=True,
                confidence=0.0,
                details=f"Guardrail error (fail-open): {e}",
                latency_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        is_safe = classification.startswith("safe")
        return GuardrailResult(
            allowed=is_safe,
            confidence=0.95,
            category="" if is_safe else classification,  # e.g. "unsafe s2"
            details=f"LlamaGuard3 verdict: {classification}",
            latency_ms=elapsed,
        )

    def check_batch(self, texts: list[str]) -> list[GuardrailResult]:
        return [self.check(t) for t in texts]


# ---------------------------------------------------------------------------
# FastAPI App (OpenAI-compatible + guardrail-aware)
# ---------------------------------------------------------------------------
guardrail = LlamaGuardGuardrail()
app = FastAPI(
    title="LlamaGuard-as-Guardrail Experiment",
    description="Experiment 1: LlamaGuard3 input filter + llama3.1:8b-instruct-q8_0 LLM",
    version="1.0.0",
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user, system, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    model: str = Field(default="llamaguard-experiment", description="Model name (ignored)")
    messages: list[ChatMessage]
    temperature: float | None = Field(default=0.7)
    max_tokens: int | None = Field(default=512)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "experiment": "llamaguard-as-guardrail",
        "guard_model": GUARD_MODEL,
        "main_model": MAIN_MODEL,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible endpoint. Garak targets this directly."""
    # Extract last user message for guardrail check
    user_messages = [m for m in req.messages if m.role == "user"]
    user_text = user_messages[-1].content if user_messages else ""

    # --- Guardrail check ---
    result: GuardrailResult = guardrail.check(user_text)

    if not result.allowed:
        content = DEFAULT_REFUSAL
        blocked = True
    else:
        # --- Forward to main LLM via Ollama ---
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        try:
            llm_resp = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": MAIN_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": req.temperature or 0.7},
                },
                timeout=120,
            )
            llm_resp.raise_for_status()
            content = llm_resp.json()["message"]["content"]
        except requests.RequestException as e:
            content = f"[LLM Error] {e}"
        blocked = False

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "guardrail": {
            "active": True,
            "blocked": blocked,
            "category": result.category,
            "confidence": result.confidence,
            "latency_ms": round(result.latency_ms, 2),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
