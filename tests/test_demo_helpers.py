import numpy as np

from src.demo import normalize_ingredient, normalize_scores, normalized_recipe_ingredients


def test_normalize_ingredient_matches_graph_grain() -> None:
    assert normalize_ingredient(" Fresh Lemon-Juice! ") == "fresh_lemon_juice"
    assert normalize_ingredient("None") == ""


def test_normalized_recipe_ingredients_parses_foodcom_list() -> None:
    assert normalized_recipe_ingredients("['Chicken Breast', 'olive oil']") == {
        "chicken_breast",
        "olive_oil",
    }


def test_normalize_scores_handles_constant_values() -> None:
    assert np.array_equal(normalize_scores(np.array([2.0, 2.0])), np.array([1.0, 1.0]))
