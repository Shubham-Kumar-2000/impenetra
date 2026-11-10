"""
Experiment 1: LlamaGuard-as-Guardrail Server
=============================================
Uses the Impenetra API framework (create_app) with the injectable
LlamaGuardGuardrail component. Safe prompts are routed to Ollama's
llama3.1:8b-instruct-q8_0.
"""

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.llamaguard_as_guardrail.guardrail import LlamaGuardGuardrail

guardrail = LlamaGuardGuardrail()

# Inject into Impenetra FastAPI app
app = create_app(
    guardrail=guardrail,
    backend_url="http://localhost:11434",
    model_name="llama3.1:8b-instruct-q8_0",
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
