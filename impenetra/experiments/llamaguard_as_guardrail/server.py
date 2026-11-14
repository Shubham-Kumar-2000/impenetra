"""
Experiment 1: LlamaGuard-as-Guardrail Server (Gradio Space Integration)
=======================================================================
Uses the Impenetra API framework (create_app) with the injectable
LlamaGuardGuardrail component connecting to a LlamaGuard Gradio Space
("llama-guard-space" or custom Gradio URL).
"""

import os

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.llamaguard_as_guardrail.guardrail import (
    DEFAULT_GRADIO_URL,
    LlamaGuardGuardrail,
)

# Accept Gradio URL / Space name via environment variable
gradio_url = os.getenv("LLAMA_GUARD_GRADIO_URL", DEFAULT_GRADIO_URL)
guardrail = LlamaGuardGuardrail(gradio_url=gradio_url)

# Inject into Impenetra FastAPI app
app = create_app(
    guardrail=guardrail,
    backend_url=os.getenv("OLLAMA_BACKEND_URL"),
    model_name=os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b-instruct-q8_0"),
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
