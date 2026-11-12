"""Pytest configuration and global test fixtures for Impenetra.

Ensures FatSecret API credentials and mock clients are configured
so test collection and test execution run deterministically in CI and local environments.
"""

import os
import sys
from unittest.mock import MagicMock

# 1. Set mock credentials in environment if not already set
os.environ.setdefault("FAT_SECRET_CONSUMER_KEY", "mock_fat_secret_consumer_key")
os.environ.setdefault("FAT_SECRET_CONSUMER_SECRET", "mock_fat_secret_consumer_secret")

# 2. Mock fatsecret library if not installed in current environment (e.g., in CI)
if "fatsecret" not in sys.modules:
    mock_fs_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.foods_search.return_value = [
        {"food_id": 101, "food_name": "Apple", "food_type": "Generic"},
        {"food_id": 102, "food_name": "Banana", "food_type": "Generic"},
    ]
    mock_client_instance.food_get.return_value = {
        "food_id": 101,
        "food_name": "Apple",
        "servings": {"serving": [{"calories": "52", "protein": "0.26"}]},
    }
    mock_client_instance.recipes_search.return_value = [
        {"recipe_id": 201, "recipe_name": "Mediterranean Quinoa Salad"},
    ]
    mock_client_instance.recipe_get.return_value = {
        "recipe_id": 201,
        "recipe_name": "Mediterranean Quinoa Salad",
        "cooking_time_min": 15,
        "ingredients": ["quinoa", "cucumber", "tomato", "feta"],
    }
    mock_fs_module.Fatsecret.return_value = mock_client_instance
    sys.modules["fatsecret"] = mock_fs_module
