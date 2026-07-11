import numpy as np
import pandas as pd

from src.demo_core import (
    DemoContext,
    HybridState,
    cluster_recommendations,
    graph_neighbors,
    graph_pair_recommendations,
    normalize_ingredient,
    normalize_scores,
    normalized_recipe_ingredients,
    recommend_hybrid_recipes,
)


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


def test_cluster_recommendations_exclude_selected_recipe() -> None:
    recipes = pd.DataFrame(
        {
            "RecipeId": [1, 2, 3],
            "Name": ["Selected", "Cluster match", "Other cluster"],
            "AggregatedRating": [5.0, 4.8, 5.0],
            "ReviewCount": [10, 100, 100],
        }
    )
    context = DemoContext(
        recipes=recipes,
        cluster_assignments=pd.DataFrame({"RecipeId": [1, 2, 3], "cluster": [0, 0, 1]}),
        cluster_profiles=pd.DataFrame({"cluster": [0, 1]}),
    )
    recommendations = cluster_recommendations(context, 1)
    assert recommendations["RecipeId"].tolist() == [2]


def test_graph_pair_recommendations_require_selected_ingredient_and_neighbor() -> None:
    recipes = pd.DataFrame(
        {
            "RecipeId": [1, 2],
            "Name": ["Chicken with Onion", "Chicken only"],
            "RecipeIngredientParts": ["['chicken', 'onion']", "['chicken', 'pepper']"],
            "AggregatedRating": [4.5, 5.0],
            "ReviewCount": [20, 20],
        }
    )
    context = DemoContext(recipes=recipes, cluster_assignments=None, cluster_profiles=None)
    nodes = pd.DataFrame({"ingredient": ["chicken", "onion"], "pagerank_weighted": [0.2, 0.8]})
    edges = pd.DataFrame({"source": ["chicken"], "target": ["onion"], "cooccurrence_count": [30]})
    neighbors = graph_neighbors(nodes, edges, "chicken")
    recommendations = graph_pair_recommendations(context, "chicken", neighbors)
    assert recommendations["RecipeId"].tolist() == [1]


def test_hybrid_recommendations_exclude_seen_recipes() -> None:
    class Collaborative:
        user_to_idx = {7: 0}
        recipe_to_idx = {1: 0, 2: 1, 3: 2}
        user_means = np.array([4.0])
        user_factors = np.array([[1.0]])
        recipe_factors = np.array([[1.0], [2.0], [3.0]])

    class Content:
        recipe_to_svd_idx = {1: 0, 2: 1, 3: 2}
        user_profiles = {7: np.array([1.0])}
        global_profile = np.array([1.0])
        X_content_normalized = np.array([[1.0], [2.0], [3.0]])

    recipes = pd.DataFrame({"RecipeId": [1, 2, 3], "Name": ["Seen", "Two", "Three"]})
    context = DemoContext(recipes=recipes, cluster_assignments=None, cluster_profiles=None)
    state = HybridState(
        train_reviews=pd.DataFrame({"AuthorId": [7], "RecipeId": [1]}),
        collaborative_model=Collaborative(),
        content_model=Content(),
    )
    recommendations = recommend_hybrid_recipes(context, state, 7, top_k=3)
    assert 1 not in recommendations["RecipeId"].tolist()
