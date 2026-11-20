"""Vector Similarity as Guardrail Experiment."""

from impenetra.experiments.vector_similarity_as_guardrail.guardrail import (
    MAX_TOKEN_DIFF_RATIO,
    MIN_TOKEN_THRESHOLD,
    SECURED_MAX_EXCESS_TOKENS,
    SIMILARITY_THRESHOLD,
    TOKEN_DIFF_PERCENT,
    VectorSimilarityGuardrail,
    is_token_diff_valid,
    load_guardrail_config,
)

__all__ = [
    "VectorSimilarityGuardrail",
    "SIMILARITY_THRESHOLD",
    "TOKEN_DIFF_PERCENT",
    "MAX_TOKEN_DIFF_RATIO",
    "SECURED_MAX_EXCESS_TOKENS",
    "MIN_TOKEN_THRESHOLD",
    "is_token_diff_valid",
    "load_guardrail_config",
]
