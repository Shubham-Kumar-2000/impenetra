"""
Experiment 3: System-Prompt-Without-Guardrail Server
=====================================================
Uses the Impenetra API framework (create_app) with PassThroughGuardrail
and injects the charter system prompt directly into the LLM call.
Target model: Ollama llama3.1:8b-instruct-q8_0.
"""

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.system_prompt_without_guardrail.guardrail import (
    CHARTER_SYSTEM_PROMPT,
    PassThroughGuardrail,
)

guardrail = PassThroughGuardrail()

# Inject into Impenetra FastAPI app with the charter system prompt
app = create_app(
    guardrail=guardrail,
    backend_url="http://localhost:11434",
    model_name="llama3.1:8b-instruct-q8_0",
    system_prompt=CHARTER_SYSTEM_PROMPT,
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
