"""
Experiment 2: NeMo Guardrails Server
=====================================
Uses the Impenetra API framework (create_app) with the injectable
NeMoGuardrail component. Prompts are evaluated against NeMo Guardrails,
and allowed prompts are processed without an llm_caller (using DEFAULT_ACCEPTANCE).
Blocked prompts return DEFAULT_REFUSAL.
"""

import os
import requests
import uvicorn

from impenetra.api.app import ChatMessage, create_app
from impenetra.experiments.nemo_guardrails.guardrail import (
    NeMoGuardrail,
    resolve_main_model_api_base,
)

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
api_base = resolve_main_model_api_base()
guardrail = NeMoGuardrail(main_model_api_base=api_base)


def openai_llm_caller(messages: list[ChatMessage]) -> str:
    """Send conversation messages to the remote OpenAI /v1/chat/completions endpoint."""
    endpoint = f"{api_base}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": 0.0,
    }
    resp = requests.post(endpoint, json=payload, timeout=120)
    if not resp.ok:
        raise RuntimeError(
            f"Remote OpenAI endpoint {endpoint} returned HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# Inject into Impenetra FastAPI app without llm_caller;
# create_app automatically serves DEFAULT_ACCEPTANCE on allowed prompts and DEFAULT_REFUSAL on blocked prompts
app = create_app(
    guardrail=guardrail,
    model_name=MODEL_NAME,
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
