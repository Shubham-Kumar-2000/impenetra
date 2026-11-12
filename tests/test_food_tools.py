"""Unit tests specifically for impenetra/food_agent/food_tools.py."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest

from impenetra.food_agent.food_tools import (
    DEFAULT_FOOD_TOOLS,
    get_food_details,
    get_recipe_details,
    recipes_search,
    search_food,
)


def test_food_tools_definitions():
    """Verify tool metadata, names, counts, and descriptions."""
    tool_names = [t.name for t in DEFAULT_FOOD_TOOLS]
    assert "search_food" in tool_names
    assert "get_food_details" in tool_names
    assert "recipes_search" in tool_names
    assert "get_recipe_details" in tool_names
    assert len(DEFAULT_FOOD_TOOLS) == 4

    assert "Search for different food items" in search_food.description
    assert "Get detailed information about a specific food item" in get_food_details.description
    assert "Search for different food recipes" in recipes_search.description
    assert "Get detailed information about a specific recipe" in get_recipe_details.description


def test_search_food_tool_invoke(monkeypatch):
    """Verify search_food.invoke() calls fs.foods_search with query."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.foods_search.return_value = [
        {"food_id": 101, "food_name": "Apple", "food_type": "Generic"},
        {"food_id": 102, "food_name": "Apple Cider", "food_type": "Beverage"},
    ]
    monkeypatch.setattr(ft, "fs", mock_fs)

    results = search_food.invoke({"query": "apple"})
    mock_fs.foods_search.assert_called_once_with("apple")
    assert len(results) == 2
    assert results[0]["food_name"] == "Apple"


def test_search_food_underlying_function(monkeypatch):
    """Verify search_food underlying function can be invoked directly."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.foods_search.return_value = [{"food_id": 103, "food_name": "Banana"}]
    monkeypatch.setattr(ft, "fs", mock_fs)

    results = search_food.func("banana")
    mock_fs.foods_search.assert_called_once_with("banana")
    assert results[0]["food_id"] == 103


def test_get_food_details_tool_invoke(monkeypatch):
    """Verify get_food_details.invoke() calls fs.food_get with food_id."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.food_get.return_value = {
        "food_id": 101,
        "food_name": "Apple",
        "servings": {"serving": [{"calories": "52", "protein": "0.26"}]},
    }
    monkeypatch.setattr(ft, "fs", mock_fs)

    details = get_food_details.invoke({"food_id": 101})
    mock_fs.food_get.assert_called_once_with(101)
    assert details["food_name"] == "Apple"
    assert "servings" in details


def test_get_food_details_underlying_function(monkeypatch):
    """Verify get_food_details underlying function can be invoked directly."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.food_get.return_value = {"food_id": 104, "food_name": "Avocado"}
    monkeypatch.setattr(ft, "fs", mock_fs)

    details = get_food_details.func(104)
    mock_fs.food_get.assert_called_once_with(104)
    assert details["food_name"] == "Avocado"


def test_recipes_search_tool_invoke(monkeypatch):
    """Verify recipes_search.invoke() calls fs.recipes_search with query."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.recipes_search.return_value = [
        {"recipe_id": 201, "recipe_name": "Mediterranean Quinoa Salad"},
        {"recipe_id": 202, "recipe_name": "Quinoa Veggie Bowl"},
    ]
    monkeypatch.setattr(ft, "fs", mock_fs)

    results = recipes_search.invoke({"query": "quinoa"})
    mock_fs.recipes_search.assert_called_once_with("quinoa")
    assert len(results) == 2
    assert results[0]["recipe_name"] == "Mediterranean Quinoa Salad"


def test_recipes_search_underlying_function(monkeypatch):
    """Verify recipes_search underlying function can be invoked directly."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.recipes_search.return_value = [{"recipe_id": 203, "recipe_name": "Grilled Chicken"}]
    monkeypatch.setattr(ft, "fs", mock_fs)

    results = recipes_search.func("chicken")
    mock_fs.recipes_search.assert_called_once_with("chicken")
    assert results[0]["recipe_id"] == 203


def test_get_recipe_details_tool_invoke(monkeypatch):
    """Verify get_recipe_details.invoke() calls fs.recipe_get with recipe_id."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.recipe_get.return_value = {
        "recipe_id": 201,
        "recipe_name": "Mediterranean Quinoa Salad",
        "cooking_time_min": 15,
        "ingredients": ["quinoa", "cucumber", "tomato", "feta"],
    }
    monkeypatch.setattr(ft, "fs", mock_fs)

    details = get_recipe_details.invoke({"recipe_id": 201})
    mock_fs.recipe_get.assert_called_once_with(201)
    assert details["recipe_id"] == 201
    assert "ingredients" in details


def test_get_recipe_details_underlying_function(monkeypatch):
    """Verify get_recipe_details underlying function can be invoked directly."""
    import impenetra.food_agent.food_tools as ft

    mock_fs = MagicMock()
    mock_fs.recipe_get.return_value = {"recipe_id": 205, "recipe_name": "Fruit Smoothie"}
    monkeypatch.setattr(ft, "fs", mock_fs)

    details = get_recipe_details.func(205)
    mock_fs.recipe_get.assert_called_once_with(205)
    assert details["recipe_name"] == "Fruit Smoothie"


def test_tools_when_fs_client_is_none(monkeypatch):
    """Verify all tools safely return empty results when the fs client is None."""
    import impenetra.food_agent.food_tools as ft

    monkeypatch.setattr(ft, "fs", None)

    assert search_food.invoke({"query": "pizza"}) == []
    assert search_food.func("pizza") == []

    assert get_food_details.invoke({"food_id": 999}) == {}
    assert get_food_details.func(999) == {}

    assert recipes_search.invoke({"query": "pasta"}) == []
    assert recipes_search.func("pasta") == []

    assert get_recipe_details.invoke({"recipe_id": 888}) == {}
    assert get_recipe_details.func(888) == {}


def test_missing_consumer_key_raises_value_error(monkeypatch):
    """Verify missing FAT_SECRET_CONSUMER_KEY raises ValueError on import."""
    import impenetra.food_agent.food_tools as ft

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    monkeypatch.delenv("FAT_SECRET_CONSUMER_KEY", raising=False)
    monkeypatch.setenv("FAT_SECRET_CONSUMER_SECRET", "dummy_secret")

    with pytest.raises(ValueError, match="Please set the FAT_SECRET_CONSUMER_KEY"):
        importlib.reload(ft)

    monkeypatch.undo()
    importlib.reload(ft)


def test_missing_consumer_secret_raises_value_error(monkeypatch):
    """Verify missing FAT_SECRET_CONSUMER_SECRET raises ValueError on import."""
    import impenetra.food_agent.food_tools as ft

    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    monkeypatch.setenv("FAT_SECRET_CONSUMER_KEY", "dummy_key")
    monkeypatch.delenv("FAT_SECRET_CONSUMER_SECRET", raising=False)

    with pytest.raises(ValueError, match="Please set the FAT_SECRET_CONSUMER_KEY"):
        importlib.reload(ft)

    monkeypatch.undo()
    importlib.reload(ft)


def test_fatsecret_client_init_error_raises_runtime_error(monkeypatch):
    """Verify RuntimeError is raised when Fatsecret class raises an initialization exception."""
    import impenetra.food_agent.food_tools as ft

    mock_mod = MagicMock()
    mock_mod.Fatsecret = MagicMock(side_effect=Exception("OAuth handshake failed"))
    monkeypatch.setitem(sys.modules, "fatsecret", mock_mod)
    monkeypatch.setenv("FAT_SECRET_CONSUMER_KEY", "test_key")
    monkeypatch.setenv("FAT_SECRET_CONSUMER_SECRET", "test_secret")

    with pytest.raises(RuntimeError, match="Failed to initialize Fatsecret client"):
        importlib.reload(ft)

    monkeypatch.undo()
    importlib.reload(ft)
