"""
Experiment: DSPy-GEPA Optimized System-Prompt-as-Guardrail Server (LLM-as-a-Judge)
==================================================================================
Uses the Impenetra API framework (create_app) with DSPyGEPASystemPromptGuardrail.
Evaluates incoming prompts using the remote LLM as a judge with the GEPA-optimized
safety system prompt loaded from trained_system_prompt.txt.
Allowed prompts return DEFAULT_ACCEPTANCE, and blocked prompts return DEFAULT_REFUSAL.
"""

import logging
import os
import sys

import uvicorn

from impenetra.api.app import create_app
from impenetra.experiments.dspy_gepa_system_prompt_as_guardrail.guardrail import (
    DEFAULT_MODEL_NAME,
    DSPyGEPASystemPromptGuardrail,
    resolve_main_model_api_base,
)

logger = logging.getLogger("dspy_gepa_guardrail_server")


def create_server_app(
    api_base: str | None = None,
    model_name: str | None = None,
):
    """Factory creating the FastAPI app with DSPyGEPASystemPromptGuardrail."""
    resolved_model = model_name or os.getenv("MAIN_MODEL_NAME", DEFAULT_MODEL_NAME)
    raw_url = api_base or os.getenv("SYSTEM_PROMPT_MAIN_MODEL_API_BASE", "https://localhost:8000")
    resolved_base = resolve_main_model_api_base(raw_url)

    guard = DSPyGEPASystemPromptGuardrail(
        main_model_api_base=resolved_base,
        model_name=resolved_model,
    )
    logger.info(f"Loaded DSPyGEPASystemPromptGuardrail with prompt source: {guard.prompt_source}")

    return create_app(
        guardrail=guard,
        llm_caller=None,
        agent=False,
        model_name=resolved_model,
    )


app = create_server_app()

if __name__ == "__main__":
    if not os.getenv("SYSTEM_PROMPT_MAIN_MODEL_API_BASE"):
        logger.error("SYSTEM_PROMPT_MAIN_MODEL_API_BASE environment variable is required.")
        sys.exit(1)
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
