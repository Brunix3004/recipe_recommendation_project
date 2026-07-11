"""Reusable data and recommendation logic for the CLI and Streamlit demos."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPES_PATH = PROJECT_ROOT / "data/processed/recipes_processed.csv"
REVIEWS_PATH = PROJECT_ROOT / "data/processed/reviews_processed.csv"
CONTENT_SVD_PATH = PROJECT_ROOT / "artifacts/week5/pca_svd/X_content_svd.npy"
CONTENT_RECIPE_IDS_PATH = PROJECT_ROOT / "artifacts/week5/pca_svd/reduced_recipe_ids.csv"
CLUSTER_ASSIGNMENTS_PATH = PROJECT_ROOT / "artifacts/week7/clustering_models/cluster_assignments_k5.csv"
CLUSTER_PROFILES_PATH = PROJECT_ROOT / "artifacts/week7/clustering_reports/cluster_profile_summary.csv"
GRAPH_NODES_PATH = PROJECT_ROOT / "artifacts/week12/ingredient_graph/ingredient_nodes.csv"
GRAPH_EDGES_PATH = PROJECT_ROOT / "artifacts/week12/ingredient_graph/ingredient_edges.csv"

RECIPE_COLUMNS = [
    "RecipeId",
    "Name",
    "Description",
    "Images",
    "RecipeCategory",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
    "RecipeServings",
    "AggregatedRating",
    "ReviewCount",
    "Calories",
    "ProteinContent",
    "TotalTime_Minutes",
]
INVALID_INGREDIENTS = {"", "nan", "none", "null", "missing", "character_0"}


@dataclass
class HybridState:
    train_reviews: pd.DataFrame
    collaborative_model: Any
    content_model: Any


@dataclass
class DemoContext:
    recipes: pd.DataFrame
    cluster_assignments: pd.DataFrame | None
    cluster_profiles: pd.DataFrame | None
    graph_nodes: pd.DataFrame | None = None
    graph_edges: pd.DataFrame | None = None


def safe_parse_list(value: Any) -> list[str]:
    """Parse Food.com list-like cells without evaluating arbitrary input."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def normalize_ingredient(token: str) -> str:
    """Match the normalized node grain used by the Week 12 graph."""
    token = str(token).strip().lower()
    if token in INVALID_INGREDIENTS:
        return ""
    token = token.replace("&", " and ").replace("'", "")
    token = re.sub(r"[^\w\s-]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    token = re.sub(r"[\s-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return "" if token in INVALID_INGREDIENTS else token


def normalized_recipe_ingredients(value: Any) -> set[str]:
    return {token for token in (normalize_ingredient(item) for item in safe_parse_list(value)) if token}


def first_recipe_image(value: Any) -> str | None:
    """Return the first usable source image from a Food.com Images cell."""
    for image in safe_parse_list(value):
        if image.startswith(("https://", "http://")):
            return image
    return None


def bayesian_recipe_score(
    rating: pd.Series, reviews: pd.Series, global_mean: float, minimum_reviews: int = 10
) -> pd.Series:
    rating_numeric = pd.to_numeric(rating, errors="coerce").fillna(global_mean)
    review_numeric = pd.to_numeric(reviews, errors="coerce").fillna(0.0)
    return (review_numeric * rating_numeric + minimum_reviews * global_mean) / (review_numeric + minimum_reviews)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    minimum, maximum = float(scores.min()), float(scores.max())
    if maximum <= minimum:
        return np.ones_like(scores, dtype=float)
    return (scores - minimum) / (maximum - minimum)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def load_context() -> DemoContext:
    if not RECIPES_PATH.exists():
        raise FileNotFoundError(f"Processed recipes were not found at {RECIPES_PATH}.")
    available_columns = set(pd.read_csv(RECIPES_PATH, nrows=0).columns)
    missing_columns = sorted(set(RECIPE_COLUMNS) - available_columns)
    if missing_columns:
        raise ValueError(f"recipes_processed.csv is missing columns: {', '.join(missing_columns)}")

    recipes = pd.read_csv(RECIPES_PATH, usecols=RECIPE_COLUMNS)
    recipes["RecipeId"] = pd.to_numeric(recipes["RecipeId"], errors="coerce")
    recipes = recipes.dropna(subset=["RecipeId"]).copy()
    recipes["RecipeId"] = recipes["RecipeId"].astype(int)

    assignments = load_optional_csv(CLUSTER_ASSIGNMENTS_PATH)
    profiles = load_optional_csv(CLUSTER_PROFILES_PATH)
    if assignments is not None and not {"RecipeId", "cluster"}.issubset(assignments.columns):
        assignments = None
    if profiles is not None and "cluster" not in profiles.columns:
        profiles = None
    return DemoContext(recipes=recipes, cluster_assignments=assignments, cluster_profiles=profiles)


def search_recipes(context: DemoContext, query: str, limit: int = 12) -> pd.DataFrame:
    if not query.strip():
        return context.recipes.iloc[0:0].copy()
    return context.recipes[
        context.recipes["Name"].fillna("").str.contains(query, case=False, regex=False)
    ].head(limit).copy()


def get_cluster_profile(context: DemoContext, recipe_id: int) -> tuple[int, pd.Series | None] | None:
    if context.cluster_assignments is None:
        return None
    matches = context.cluster_assignments.loc[
        pd.to_numeric(context.cluster_assignments["RecipeId"], errors="coerce").eq(recipe_id)
    ]
    if matches.empty:
        return None
    cluster_id = int(matches.iloc[0]["cluster"])
    if context.cluster_profiles is None:
        return cluster_id, None
    profile = context.cluster_profiles.loc[context.cluster_profiles["cluster"].eq(cluster_id)]
    return cluster_id, None if profile.empty else profile.iloc[0]


def cluster_recommendations(context: DemoContext, recipe_id: int, limit: int = 5) -> pd.DataFrame:
    cluster_info = get_cluster_profile(context, recipe_id)
    if cluster_info is None or context.cluster_assignments is None:
        return context.recipes.iloc[0:0].copy()
    cluster_id, _ = cluster_info
    cluster_recipe_ids = set(
        pd.to_numeric(
            context.cluster_assignments.loc[context.cluster_assignments["cluster"].eq(cluster_id), "RecipeId"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    candidates = context.recipes[
        context.recipes["RecipeId"].isin(cluster_recipe_ids) & context.recipes["RecipeId"].ne(recipe_id)
    ].copy()
    if candidates.empty:
        return candidates
    global_mean = float(pd.to_numeric(context.recipes["AggregatedRating"], errors="coerce").mean())
    candidates["rank_score"] = bayesian_recipe_score(
        candidates["AggregatedRating"], candidates["ReviewCount"], global_mean
    )
    return candidates.sort_values(
        ["rank_score", "ReviewCount", "Name"], ascending=[False, False, True]
    ).head(limit)


def load_graph_artifacts(context: DemoContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    if context.graph_nodes is None:
        context.graph_nodes = load_optional_csv(GRAPH_NODES_PATH)
    if context.graph_edges is None:
        context.graph_edges = load_optional_csv(GRAPH_EDGES_PATH)
    required_nodes = {"ingredient", "pagerank_weighted"}
    required_edges = {"source", "target", "cooccurrence_count"}
    if context.graph_nodes is None or not required_nodes.issubset(context.graph_nodes.columns):
        raise FileNotFoundError("Ingredient graph node data is missing or invalid.")
    if context.graph_edges is None or not required_edges.issubset(context.graph_edges.columns):
        raise FileNotFoundError("Ingredient graph edge data is missing or invalid.")
    return context.graph_nodes, context.graph_edges


def graph_ingredient_options(nodes: pd.DataFrame, raw_ingredient: str, limit: int = 10) -> pd.DataFrame:
    token = normalize_ingredient(raw_ingredient)
    if not token:
        return nodes.iloc[0:0].copy()
    exact = nodes.loc[nodes["ingredient"].eq(token)]
    if not exact.empty:
        return exact.copy()
    return nodes[nodes["ingredient"].str.contains(token, case=False, regex=False, na=False)].sort_values(
        "pagerank_weighted", ascending=False
    ).head(limit)


def graph_neighbors(nodes: pd.DataFrame, edges: pd.DataFrame, ingredient: str, limit: int = 5) -> pd.DataFrame:
    outgoing = edges.loc[edges["source"].eq(ingredient), ["target", "cooccurrence_count"]].rename(
        columns={"target": "neighbor"}
    )
    incoming = edges.loc[edges["target"].eq(ingredient), ["source", "cooccurrence_count"]].rename(
        columns={"source": "neighbor"}
    )
    neighbors = pd.concat([outgoing, incoming], ignore_index=True)
    if neighbors.empty:
        return neighbors
    scores = nodes[["ingredient", "pagerank_weighted"]].rename(
        columns={"ingredient": "neighbor", "pagerank_weighted": "neighbor_pagerank"}
    )
    return neighbors.merge(scores, on="neighbor", how="left").sort_values(
        ["neighbor_pagerank", "cooccurrence_count", "neighbor"], ascending=[False, False, True]
    ).head(limit)


def graph_pair_recommendations(
    context: DemoContext, ingredient: str, neighbors: pd.DataFrame, limit: int = 5
) -> pd.DataFrame:
    if neighbors.empty:
        return context.recipes.iloc[0:0].copy()
    search_phrase = ingredient.replace("_", " ")
    candidates = context.recipes[
        context.recipes["RecipeIngredientParts"].fillna("").str.contains(search_phrase, case=False, regex=False)
    ].copy()
    partner_scores = dict(zip(neighbors["neighbor"], neighbors["neighbor_pagerank"]))
    partner_tokens = set(partner_scores)
    matched_partners: list[str] = []
    for value in candidates["RecipeIngredientParts"]:
        tokens = normalized_recipe_ingredients(value)
        shared = sorted(tokens & partner_tokens, key=lambda item: partner_scores[item], reverse=True)
        matched_partners.append(shared[0] if ingredient in tokens and shared else "")
    candidates["graph_partner"] = matched_partners
    candidates = candidates.loc[candidates["graph_partner"].ne("")].copy()
    if candidates.empty:
        return candidates
    global_mean = float(pd.to_numeric(context.recipes["AggregatedRating"], errors="coerce").mean())
    candidates["rank_score"] = bayesian_recipe_score(
        candidates["AggregatedRating"], candidates["ReviewCount"], global_mean
    )
    candidates["partner_pagerank"] = candidates["graph_partner"].map(partner_scores)
    return candidates.sort_values(
        ["partner_pagerank", "rank_score", "ReviewCount"], ascending=[False, False, False]
    ).head(limit)


def fit_hybrid_models() -> HybridState:
    """Fit the existing Week 10 models for one application session."""
    required_paths = [REVIEWS_PATH, CONTENT_SVD_PATH, CONTENT_RECIPE_IDS_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing recommendation inputs: " + ", ".join(missing_paths))
    try:
        from features.run_recommendation_experiments import (
            CollaborativeSVDRecommender,
            ContentSVDRecommender,
            chronological_split,
            get_5_core,
            load_reviews,
        )
    except ModuleNotFoundError:
        from src.features.run_recommendation_experiments import (
            CollaborativeSVDRecommender,
            ContentSVDRecommender,
            chronological_split,
            get_5_core,
            load_reviews,
        )

    reviews = load_reviews(REVIEWS_PATH)
    reviews_core = get_5_core(reviews, k=5)
    train_reviews, _ = chronological_split(reviews_core)
    collaborative_model = CollaborativeSVDRecommender(n_factors=50)
    collaborative_model.fit(train_reviews)
    content_model = ContentSVDRecommender(CONTENT_SVD_PATH, CONTENT_RECIPE_IDS_PATH)
    content_model.fit(train_reviews)
    return HybridState(train_reviews, collaborative_model, content_model)


def active_user_ids(state: HybridState) -> list[int]:
    return sorted(int(user_id) for user_id in state.collaborative_model.user_to_idx)


def user_history(context: DemoContext, state: HybridState, author_id: int, limit: int = 5) -> pd.DataFrame:
    history = state.train_reviews.loc[state.train_reviews["AuthorId"].eq(author_id)].sort_values(
        "DateSubmitted", ascending=False
    ).head(limit)
    return history.merge(context.recipes[["RecipeId", "Name"]], on="RecipeId", how="left")


def recommend_hybrid_recipes(
    context: DemoContext, state: HybridState, author_id: int, top_k: int = 3
) -> pd.DataFrame:
    cf_model, content_model = state.collaborative_model, state.content_model
    if author_id not in cf_model.user_to_idx:
        raise ValueError("AuthorId is not an active user in the trained 5-core data.")
    seen_recipe_ids = set(
        state.train_reviews.loc[state.train_reviews["AuthorId"].eq(author_id), "RecipeId"].astype(int)
    )
    metadata_ids = set(context.recipes["RecipeId"].astype(int))
    candidate_ids = sorted(
        recipe_id
        for recipe_id in cf_model.recipe_to_idx
        if recipe_id not in seen_recipe_ids
        and recipe_id in content_model.recipe_to_svd_idx
        and recipe_id in metadata_ids
    )
    if not candidate_ids:
        return pd.DataFrame()
    cf_indices = np.array([cf_model.recipe_to_idx[recipe_id] for recipe_id in candidate_ids])
    content_indices = np.array([content_model.recipe_to_svd_idx[recipe_id] for recipe_id in candidate_ids])
    user_index = cf_model.user_to_idx[author_id]
    cf_scores = cf_model.user_means[user_index] + cf_model.recipe_factors[cf_indices] @ cf_model.user_factors[user_index]
    profile = content_model.user_profiles.get(author_id, content_model.global_profile)
    content_scores = content_model.X_content_normalized[content_indices] @ profile
    hybrid_scores = 0.6 * normalize_scores(cf_scores) + 0.4 * normalize_scores(content_scores)
    top_indices = np.argsort(-hybrid_scores)[:top_k]
    ranked = pd.DataFrame(
        {
            "RecipeId": [candidate_ids[index] for index in top_indices],
            "hybrid_score": hybrid_scores[top_indices],
            "cf_score": cf_scores[top_indices],
            "content_score": content_scores[top_indices],
        }
    )
    return ranked.merge(context.recipes, on="RecipeId", how="left", validate="one_to_one")
