"""
Experiment: PromptGuard-as-Guardrail Server
===========================================
Uses the Impenetra API framework (create_app) with the injectable
PromptGuardGuardrail component connecting to a Meta Llama Prompt Guard 2 86M API
server exposed by prompt_guard_colab.ipynb (POST /predict).
"""

import os

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.promptguard_as_guardrail.guardrail import (
    DEFAULT_PROMPT_GUARD_URL,
    PromptGuardGuardrail,
)

# Accept Prompt Guard endpoint URL via environment variable
prompt_guard_url = os.getenv("PROMPT_GUARD_URL", DEFAULT_PROMPT_GUARD_URL)
guardrail = PromptGuardGuardrail(endpoint_url=prompt_guard_url)

# Inject into Impenetra FastAPI app
app = create_app(
    guardrail=guardrail,
    backend_url=os.getenv("OLLAMA_BACKEND_URL"),
    model_name=os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b-instruct-q8_0"),
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
