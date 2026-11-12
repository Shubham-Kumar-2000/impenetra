"""Impenetra — Low-latency, high-accuracy guardrails research for AI agents."""

from impenetra.food_agent import (
    DEFAULT_FOOD_TOOLS,
    FOOD_AGENT_SYSTEM_PROMPT,
    create_food_agent,
    get_default_agent,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "create_food_agent",
    "get_default_agent",
    "FOOD_AGENT_SYSTEM_PROMPT",
    "DEFAULT_FOOD_TOOLS",
]
