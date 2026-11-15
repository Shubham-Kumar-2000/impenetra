"""PromptGuard-as-guardrail experiment package."""

from impenetra.experiments.promptguard_as_guardrail.guardrail import (
    DEFAULT_PROMPT_GUARD_URL,
    MALICIOUS_THRESHOLD,
    PromptGuardGuardrail,
)

__all__ = [
    "DEFAULT_PROMPT_GUARD_URL",
    "MALICIOUS_THRESHOLD",
    "PromptGuardGuardrail",
]
