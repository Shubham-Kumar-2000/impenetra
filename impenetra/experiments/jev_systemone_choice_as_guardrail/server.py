"""
Experiment: Jev SystemOne Choice Guardrail Server (TypeSafe.ai)
===============================================================
Uses the Impenetra API framework (create_app) with JevSystemOneChoiceGuardrail.
Evaluates incoming prompts using TypeSafe.ai's Jev System One model with a multi-class
'choice' primitive.
Server runs without an LLM caller (llm_caller=None, agent=False).
Allowed prompts return DEFAULT_ACCEPTANCE, and blocked prompts return DEFAULT_REFUSAL.
LLM judge calls remain 0 across all requests.
"""

import os

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.jev_systemone_choice_as_guardrail.guardrail import (
    DEFAULT_MODEL_NAME,
    JevSystemOneChoiceGuardrail,
    resolve_typesafe_api_base,
    resolve_typesafe_api_key,
)

api_key = resolve_typesafe_api_key()
api_base = resolve_typesafe_api_base()
model_name = os.getenv("TYPESAFE_MODEL_NAME", DEFAULT_MODEL_NAME)
guardrail = JevSystemOneChoiceGuardrail(api_key=api_key, api_base=api_base, model_name=model_name)

# Inject into Impenetra FastAPI app with llm_caller kept as null (None) and agent=False
app = create_app(
    guardrail=guardrail,
    llm_caller=None,
    agent=False,
    model_name=model_name,
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
