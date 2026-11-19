"""
DSPy-GEPA Optimized System Prompt as Guardrail Experiment.
"""

from .guardrail import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAINED_PROMPT_FILE,
    JUDGE_SYSTEM_PROMPT,
    DSPyGEPASystemPromptGuardrail,
    load_system_prompt,
    parse_judge_decision,
    resolve_main_model_api_base,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TRAINED_PROMPT_FILE",
    "JUDGE_SYSTEM_PROMPT",
    "DSPyGEPASystemPromptGuardrail",
    "load_system_prompt",
    "parse_judge_decision",
    "resolve_main_model_api_base",
]
