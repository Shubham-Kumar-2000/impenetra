"""Food agent package following Crave Craft specifications."""

from impenetra.food_agent.food_agent import (
    FOOD_AGENT_SYSTEM_PROMPT,
    DefaultFoodChatModel,
    MockFoodChatModel,
    create_food_agent,
    get_default_agent,
)
from impenetra.food_agent.food_tools import (
    DEFAULT_FOOD_TOOLS,
    get_food_details,
    get_recipe_details,
    recipes_search,
    search_food,
)

__all__ = [
    "create_food_agent",
    "get_default_agent",
    "FOOD_AGENT_SYSTEM_PROMPT",
    "DefaultFoodChatModel",
    "MockFoodChatModel",
    "DEFAULT_FOOD_TOOLS",
    "search_food",
    "get_food_details",
    "recipes_search",
    "get_recipe_details",
]
