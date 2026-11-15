"""
Experiment: System-Prompt-as-Guardrail Server (LLM-as-a-Judge)
=============================================================
Uses the Impenetra API framework (create_app) with SystemPromptGuardrail.
Evaluates incoming prompts using the remote LLM as a judge.
Allowed prompts return DEFAULT_ACCEPTANCE, and blocked prompts return DEFAULT_REFUSAL.
"""

import os

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.system_prompt_as_guardrail.guardrail import (
    DEFAULT_MODEL_NAME,
    SystemPromptGuardrail,
    resolve_main_model_api_base,
)

model_name = os.getenv("MAIN_MODEL_NAME", DEFAULT_MODEL_NAME)
api_base = resolve_main_model_api_base()
guardrail = SystemPromptGuardrail(main_model_api_base=api_base, model_name=model_name)

# Inject into Impenetra FastAPI app with llm_caller kept as null (None)
app = create_app(
    guardrail=guardrail,
    llm_caller=None,
    agent=False,
    model_name=model_name,
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
