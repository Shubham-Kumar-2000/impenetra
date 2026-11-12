"""Food agent tools following Crave Craft specifications."""

import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

try:
    from fatsecret import Fatsecret  # type: ignore[import-not-found]
except ImportError:
    Fatsecret = None  # type: ignore

consumer_key = os.getenv("FAT_SECRET_CONSUMER_KEY")
consumer_secret = os.getenv("FAT_SECRET_CONSUMER_SECRET")

if not consumer_key or not consumer_secret:
    raise ValueError(
        "Please set the FAT_SECRET_CONSUMER_KEY and "
        "FAT_SECRET_CONSUMER_SECRET environment variables."
    )

fs: Optional[Any] = None
if Fatsecret is not None:
    try:
        fs = Fatsecret(consumer_key, consumer_secret)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Fatsecret client: {e}") from e


@tool
def search_food(query: str):
    """
    Search for different food items.

    Args:
        query (str): The search query for food items.

    Returns:
        list: A list of food items matching the search query.
    """
    print(f"Searching for food items with query: {query}")
    return fs.foods_search(query) if fs else []


@tool
def get_food_details(food_id: int):
    """
    Get detailed information about a specific food item using its ID.

    Args:
        food_id (int): The ID of the food item.

    Returns:
        dict: Detailed information about the food item.
    """
    print(f"Getting details for food item with ID: {food_id}")
    return fs.food_get(food_id) if fs else {}


@tool
def recipes_search(query: str):
    """
    Search for different food recipes.

    Args:
        query (str): The search query for recipes.

    Returns:
        list: A list of recipes matching the search query.
    """
    print(f"Searching for recipes with query: {query}")
    return fs.recipes_search(query) if fs else []


@tool
def get_recipe_details(recipe_id: int):
    """
    Get detailed information about a specific recipe using its ID.

    Args:
        recipe_id (int): The ID of the recipe.

    Returns:
        dict: Detailed information about the recipe.
    """
    print(f"Getting details for recipe with ID: {recipe_id}")
    return fs.recipe_get(recipe_id) if fs else {}


DEFAULT_FOOD_TOOLS = [search_food, get_food_details, recipes_search, get_recipe_details]
