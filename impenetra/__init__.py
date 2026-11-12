"""Impenetra — Low-latency, high-accuracy guardrails research for AI agents."""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "create_food_agent",
    "get_default_agent",
    "FOOD_AGENT_SYSTEM_PROMPT",
    "DEFAULT_FOOD_TOOLS",
]


def __getattr__(name: str):
    if name in {
        "create_food_agent",
        "get_default_agent",
        "FOOD_AGENT_SYSTEM_PROMPT",
        "DEFAULT_FOOD_TOOLS",
    }:
        import impenetra.food_agent as _fa

        return getattr(_fa, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
