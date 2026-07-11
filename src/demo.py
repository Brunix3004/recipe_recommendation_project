#!/usr/bin/env python3
"""Interactive terminal demo for the Recipe Recommendation Project."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    Console = None  # type: ignore[assignment,misc]
    Panel = None  # type: ignore[assignment,misc]
    Table = None  # type: ignore[assignment,misc]
    RICH_AVAILABLE = False


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
    "RecipeCategory",
    "RecipeIngredientParts",
    "AggregatedRating",
    "ReviewCount",
    "Calories",
    "TotalTime_Minutes",
]
INVALID_INGREDIENTS = {"", "nan", "none", "null", "missing", "character_0"}


@dataclass
class HybridState:
    """In-memory state fitted only when the personalized option is selected."""

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
    graph_attempted: bool = False
    hybrid_state: HybridState | None = None


def print_message(message: str, style: str | None = None) -> None:
    """Print an informational message with Rich when it is available."""
    if RICH_AVAILABLE:
        Console().print(message, style=style)
    else:
        print(message)


def print_title(title: str, subtitle: str | None = None) -> None:
    if RICH_AVAILABLE:
        body = subtitle or ""
        Console().print(Panel(body, title=title, border_style="cyan"))
    else:
        print("\n" + "=" * 60)
        print(title)
        if subtitle:
            print(subtitle)
        print("=" * 60)


def print_table(columns: list[str], rows: Iterable[Iterable[Any]], title: str | None = None) -> None:
    rows_as_text = [["" if value is None else str(value) for value in row] for row in rows]
    if RICH_AVAILABLE:
        table = Table(title=title, show_lines=False)
        for column in columns:
            table.add_column(column, overflow="fold")
        for row in rows_as_text:
            table.add_row(*row)
        Console().print(table)
        return

    if title:
        print(f"\n{title}")
    print(" | ".join(columns))
    print("-" * max(40, len(" | ".join(columns))))
    for row in rows_as_text:
        print(" | ".join(row))


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print_message("\nClosing demo.", "yellow")
        raise SystemExit(0)


def choose_index(max_index: int, prompt: str = "Choose a number (0 to cancel): ") -> int | None:
    while True:
        value = ask(prompt)
        if value == "0" or not value:
            return None
        try:
            selected = int(value)
        except ValueError:
            print_message("Please enter a valid number.", "yellow")
            continue
        if 1 <= selected <= max_index:
            return selected - 1
        print_message(f"Please choose a value between 1 and {max_index}.", "yellow")


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
    """Match the node grain used by the Week 12 ingredient graph."""
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


def bayesian_recipe_score(rating: pd.Series, reviews: pd.Series, global_mean: float, minimum_reviews: int = 10) -> pd.Series:
    """Stabilize sparse aggregate ratings before ranking cluster/graph recipes."""
    rating_numeric = pd.to_numeric(rating, errors="coerce").fillna(global_mean)
    review_numeric = pd.to_numeric(reviews, errors="coerce").fillna(0.0)
    return (review_numeric * rating_numeric + minimum_reviews * global_mean) / (review_numeric + minimum_reviews)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Map a score vector into [0, 1] while handling constant vectors."""
    if scores.size == 0:
        return scores
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum <= minimum:
        return np.ones_like(scores, dtype=float)
    return (scores - minimum) / (maximum - minimum)


def load_optional_csv(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print_message(f"{label} not found at {path}. This feature will be unavailable.", "yellow")
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print_message(f"Could not load {label}: {error}", "yellow")
        return None


def load_context() -> DemoContext:
    if not RECIPES_PATH.exists():
        raise FileNotFoundError(f"Processed recipes were not found at {RECIPES_PATH}.")

    available_columns = pd.read_csv(RECIPES_PATH, nrows=0).columns.tolist()
    missing_columns = sorted(set(RECIPE_COLUMNS) - set(available_columns))
    if missing_columns:
        raise ValueError(f"recipes_processed.csv is missing columns: {', '.join(missing_columns)}")

    recipes = pd.read_csv(RECIPES_PATH, usecols=RECIPE_COLUMNS)
    recipes["RecipeId"] = pd.to_numeric(recipes["RecipeId"], errors="coerce")
    recipes = recipes.dropna(subset=["RecipeId"]).copy()
    recipes["RecipeId"] = recipes["RecipeId"].astype(int)

    assignments = load_optional_csv(CLUSTER_ASSIGNMENTS_PATH, "Cluster assignments")
    profiles = load_optional_csv(CLUSTER_PROFILES_PATH, "Cluster profiles")
    if assignments is not None and not {"RecipeId", "cluster"}.issubset(assignments.columns):
        print_message("Cluster assignments have an unexpected schema. Skipping clustering feature.", "yellow")
        assignments = None
    if profiles is not None and "cluster" not in profiles.columns:
        print_message("Cluster profiles have an unexpected schema. Skipping clustering feature.", "yellow")
        profiles = None

    return DemoContext(recipes=recipes, cluster_assignments=assignments, cluster_profiles=profiles)


def display_recipe_matches(matches: pd.DataFrame, title: str) -> None:
    rows = []
    for index, row in matches.reset_index(drop=True).iterrows():
        rows.append(
            [
                index + 1,
                row["Name"],
                row.get("RecipeCategory", "Unknown"),
                f"{pd.to_numeric(row.get('AggregatedRating'), errors='coerce'):.1f}"
                if pd.notna(pd.to_numeric(row.get("AggregatedRating"), errors="coerce"))
                else "n/a",
                int(pd.to_numeric(row.get("ReviewCount"), errors="coerce"))
                if pd.notna(pd.to_numeric(row.get("ReviewCount"), errors="coerce"))
                else 0,
            ]
        )
    print_table(["#", "Recipe", "Category", "Rating", "Reviews"], rows, title)


def handle_search(context: DemoContext) -> None:
    print_title("Keyword Search and Similar Recipes", "Search the catalog, then explore the selected recipe's cluster.")
    query = ask("Recipe keyword (blank to return): ")
    if not query:
        return

    matches = context.recipes[
        context.recipes["Name"].fillna("").str.contains(query, case=False, regex=False)
    ].head(10).copy()
    if matches.empty:
        print_message(f"No recipe titles matched '{query}'.", "yellow")
        return

    display_recipe_matches(matches, f"Recipes matching '{query}'")
    selected_index = choose_index(len(matches), "Select a recipe to explore its cluster (0 to return): ")
    if selected_index is None:
        return
    selected = matches.iloc[selected_index]

    if context.cluster_assignments is None or context.cluster_profiles is None:
        print_message("Clustering artifacts are unavailable, so similar recipes cannot be shown.", "yellow")
        return

    cluster_row = context.cluster_assignments.loc[
        pd.to_numeric(context.cluster_assignments["RecipeId"], errors="coerce").eq(selected["RecipeId"])
    ]
    if cluster_row.empty:
        print_message("The selected recipe has no cluster assignment.", "yellow")
        return

    cluster_id = int(cluster_row.iloc[0]["cluster"])
    profile = context.cluster_profiles.loc[context.cluster_profiles["cluster"].eq(cluster_id)]
    profile_row = profile.iloc[0] if not profile.empty else None
    profile_name = (
        str(profile_row.get("dominant_category", f"Cluster {cluster_id}"))
        if profile_row is not None
        else f"Cluster {cluster_id}"
    )
    print_message(
        f"Because you liked {selected['Name']}, here are other highly-rated recipes from "
        f"the same profile: {profile_name} (Cluster {cluster_id}).",
        "green",
    )
    if profile_row is not None and pd.notna(profile_row.get("interpretation_summary")):
        print_message(str(profile_row["interpretation_summary"]))

    cluster_recipe_ids = set(
        pd.to_numeric(
            context.cluster_assignments.loc[context.cluster_assignments["cluster"].eq(cluster_id), "RecipeId"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    candidates = context.recipes[
        context.recipes["RecipeId"].isin(cluster_recipe_ids)
        & context.recipes["RecipeId"].ne(selected["RecipeId"])
    ].copy()
    if candidates.empty:
        print_message("No other recipes were available in this cluster.", "yellow")
        return

    global_mean = float(pd.to_numeric(context.recipes["AggregatedRating"], errors="coerce").mean())
    candidates["rank_score"] = bayesian_recipe_score(
        candidates["AggregatedRating"], candidates["ReviewCount"], global_mean
    )
    recommendations = candidates.sort_values(
        ["rank_score", "ReviewCount", "Name"], ascending=[False, False, True]
    ).head(5)
    display_recipe_matches(recommendations, f"Highly-rated recipes from Cluster {cluster_id}")


def load_graph_artifacts(context: DemoContext) -> bool:
    if context.graph_attempted:
        return context.graph_nodes is not None and context.graph_edges is not None
    context.graph_attempted = True
    context.graph_nodes = load_optional_csv(GRAPH_NODES_PATH, "Ingredient graph nodes")
    context.graph_edges = load_optional_csv(GRAPH_EDGES_PATH, "Ingredient graph edges")
    required_nodes = {"ingredient", "pagerank_weighted"}
    required_edges = {"source", "target", "cooccurrence_count"}
    if context.graph_nodes is None or not required_nodes.issubset(context.graph_nodes.columns):
        print_message("Ingredient node data has an unexpected schema. Skipping graph feature.", "yellow")
        context.graph_nodes = None
    if context.graph_edges is None or not required_edges.issubset(context.graph_edges.columns):
        print_message("Ingredient edge data has an unexpected schema. Skipping graph feature.", "yellow")
        context.graph_edges = None
    return context.graph_nodes is not None and context.graph_edges is not None


def resolve_graph_ingredient(nodes: pd.DataFrame, raw_ingredient: str) -> str | None:
    normalized = normalize_ingredient(raw_ingredient)
    if not normalized:
        print_message("Please enter a valid ingredient.", "yellow")
        return None
    exact = nodes.loc[nodes["ingredient"].eq(normalized)]
    if not exact.empty:
        return normalized

    options = nodes[nodes["ingredient"].str.contains(normalized, case=False, regex=False, na=False)].copy()
    if options.empty:
        print_message(f"No graph ingredient matched '{raw_ingredient}'.", "yellow")
        return None
    options = options.sort_values("pagerank_weighted", ascending=False).head(10).reset_index(drop=True)
    print_table(
        ["#", "Graph ingredient", "PageRank", "Recipe frequency"],
        [
            [index + 1, row["ingredient"], f"{row['pagerank_weighted']:.6f}", int(row.get("recipe_count", 0))]
            for index, row in options.iterrows()
        ],
        "Choose the ingredient node you meant",
    )
    selection = choose_index(len(options))
    return None if selection is None else str(options.iloc[selection]["ingredient"])


def handle_graph_recommendation(context: DemoContext) -> None:
    print_title("Graph-Based Ingredient Pairing", "Find recipes that pair your ingredient with central graph neighbors.")
    if not load_graph_artifacts(context):
        return
    assert context.graph_nodes is not None and context.graph_edges is not None

    raw_ingredient = ask("Ingredient you have (blank to return): ")
    if not raw_ingredient:
        return
    ingredient = resolve_graph_ingredient(context.graph_nodes, raw_ingredient)
    if ingredient is None:
        return

    edges = context.graph_edges
    outgoing = edges.loc[edges["source"].eq(ingredient), ["target", "cooccurrence_count"]].rename(
        columns={"target": "neighbor"}
    )
    incoming = edges.loc[edges["target"].eq(ingredient), ["source", "cooccurrence_count"]].rename(
        columns={"source": "neighbor"}
    )
    neighbors = pd.concat([outgoing, incoming], ignore_index=True)
    if neighbors.empty:
        print_message(f"'{ingredient}' has no retained graph neighbors at the current edge threshold.", "yellow")
        return

    node_scores = context.graph_nodes[["ingredient", "pagerank_weighted"]].rename(
        columns={"ingredient": "neighbor", "pagerank_weighted": "neighbor_pagerank"}
    )
    neighbors = neighbors.merge(node_scores, on="neighbor", how="left")
    neighbors = neighbors.sort_values(
        ["neighbor_pagerank", "cooccurrence_count", "neighbor"], ascending=[False, False, True]
    ).head(5)
    print_table(
        ["Partner ingredient", "PageRank", "Co-occurring recipes"],
        [
            [row["neighbor"], f"{row['neighbor_pagerank']:.6f}", int(row["cooccurrence_count"])]
            for _, row in neighbors.iterrows()
        ],
        f"Central ingredients paired with '{ingredient}'",
    )

    # Narrow with a cheap text check before parsing ingredient lists for an exact graph-node match.
    search_phrase = ingredient.replace("_", " ")
    recipe_candidates = context.recipes[
        context.recipes["RecipeIngredientParts"].fillna("").str.contains(
            search_phrase, case=False, regex=False
        )
    ].copy()
    if recipe_candidates.empty:
        print_message("No catalog recipes contained this exact graph ingredient label.", "yellow")
        return

    partner_scores = dict(zip(neighbors["neighbor"], neighbors["neighbor_pagerank"]))
    partner_tokens = set(partner_scores)
    matched_partners: list[str] = []
    for value in recipe_candidates["RecipeIngredientParts"]:
        recipe_ingredients = normalized_recipe_ingredients(value)
        shared = sorted(recipe_ingredients & partner_tokens, key=lambda item: partner_scores[item], reverse=True)
        matched_partners.append(shared[0] if ingredient in recipe_ingredients and shared else "")
    recipe_candidates["graph_partner"] = matched_partners
    recommendations = recipe_candidates.loc[recipe_candidates["graph_partner"].ne("")].copy()
    if recommendations.empty:
        print_message("No recipes contained the selected ingredient with these retained graph partners.", "yellow")
        return

    global_mean = float(pd.to_numeric(context.recipes["AggregatedRating"], errors="coerce").mean())
    recommendations["rank_score"] = bayesian_recipe_score(
        recommendations["AggregatedRating"], recommendations["ReviewCount"], global_mean
    )
    recommendations["partner_pagerank"] = recommendations["graph_partner"].map(partner_scores)
    recommendations = recommendations.sort_values(
        ["partner_pagerank", "rank_score", "ReviewCount"], ascending=[False, False, False]
    ).head(5)
    print_table(
        ["Recipe", "Graph pair", "Rating", "Reviews"],
        [
            [
                row["Name"],
                f"{ingredient} + {row['graph_partner']}",
                f"{pd.to_numeric(row['AggregatedRating'], errors='coerce'):.1f}",
                int(pd.to_numeric(row["ReviewCount"], errors="coerce"))
                if pd.notna(pd.to_numeric(row["ReviewCount"], errors="coerce"))
                else 0,
            ]
            for _, row in recommendations.iterrows()
        ],
        "Recipes suggested by ingredient pairings",
    )


def fit_hybrid_models(context: DemoContext) -> HybridState | None:
    """Fit the existing Week 10 models once and keep them for this CLI session."""
    if context.hybrid_state is not None:
        return context.hybrid_state
    if not all(path.exists() for path in [REVIEWS_PATH, CONTENT_SVD_PATH, CONTENT_RECIPE_IDS_PATH]):
        print_message("Recommendation inputs are missing. Run the Week 5 and Week 10 pipelines first.", "yellow")
        return None

    try:
        from features.run_recommendation_experiments import (
            CollaborativeSVDRecommender,
            ContentSVDRecommender,
            chronological_split,
            get_5_core,
            load_reviews,
        )

        print_message(
            "Preparing the personalized model. This first run may take a while; later requests reuse it.",
            "cyan",
        )
        reviews = load_reviews(REVIEWS_PATH)
        reviews_core = get_5_core(reviews, k=5)
        train_reviews, _ = chronological_split(reviews_core)

        collaborative_model = CollaborativeSVDRecommender(n_factors=50)
        collaborative_model.fit(train_reviews)
        content_model = ContentSVDRecommender(CONTENT_SVD_PATH, CONTENT_RECIPE_IDS_PATH)
        content_model.fit(train_reviews)
    except (OSError, ValueError, ImportError, MemoryError) as error:
        print_message(f"Could not prepare the hybrid recommender: {error}", "yellow")
        return None

    context.hybrid_state = HybridState(
        train_reviews=train_reviews,
        collaborative_model=collaborative_model,
        content_model=content_model,
    )
    return context.hybrid_state


def choose_hybrid_user(state: HybridState) -> int | None:
    raw_user_id = ask("AuthorId (press Enter for a reproducible random active user): ")
    active_users = sorted(state.collaborative_model.user_to_idx)
    if not active_users:
        print_message("The fitted collaborative model has no active users.", "yellow")
        return None
    if not raw_user_id:
        user_id = int(np.random.default_rng(42).choice(active_users))
        print_message(f"Using active AuthorId {user_id}.", "cyan")
        return user_id
    try:
        user_id = int(raw_user_id)
    except ValueError:
        print_message("AuthorId must be an integer.", "yellow")
        return None
    if user_id not in state.collaborative_model.user_to_idx:
        print_message("That AuthorId is not an active user in the trained 5-core data.", "yellow")
        return None
    return user_id


def recommend_hybrid_recipes(context: DemoContext, state: HybridState, author_id: int, top_k: int = 3) -> pd.DataFrame:
    """Rank all eligible unseen recipes with the same CF/content blend used in Week 10."""
    cf_model = state.collaborative_model
    content_model = state.content_model
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


def handle_hybrid_recommendation(context: DemoContext) -> None:
    print_title("Personalized Hybrid Recommendations", "Collaborative SVD (60%) plus content similarity (40%).")
    state = fit_hybrid_models(context)
    if state is None:
        return
    author_id = choose_hybrid_user(state)
    if author_id is None:
        return

    history = state.train_reviews.loc[state.train_reviews["AuthorId"].eq(author_id)].sort_values(
        "DateSubmitted", ascending=False
    ).head(5)
    history_display = history.merge(
        context.recipes[["RecipeId", "Name"]], on="RecipeId", how="left"
    )
    print_table(
        ["Previously rated recipe", "Rating", "Date"],
        [
            [row.get("Name", row["RecipeId"]), f"{row['Rating']:.1f}", row["DateSubmitted"].date()]
            for _, row in history_display.iterrows()
        ],
        f"Recent training-history ratings for AuthorId {author_id}",
    )

    recommendations = recommend_hybrid_recipes(context, state, author_id)
    if recommendations.empty:
        print_message("No eligible unseen recipes were available for this user.", "yellow")
        return
    print_table(
        ["#", "Recommended recipe", "Category", "Hybrid", "CF", "Content", "Catalog rating"],
        [
            [
                index + 1,
                row["Name"],
                row.get("RecipeCategory", "Unknown"),
                f"{row['hybrid_score']:.4f}",
                f"{row['cf_score']:.3f}",
                f"{row['content_score']:.3f}",
                f"{pd.to_numeric(row['AggregatedRating'], errors='coerce'):.1f}",
            ]
            for index, (_, row) in enumerate(recommendations.iterrows())
        ],
        "Top 3 personalized recommendations",
    )


def print_menu() -> None:
    print_title(
        "Recipe Intelligence System",
        "Explore keyword search, clusters, ingredient networks, and personalized ML recommendations.",
    )
    print_table(
        ["Option", "Experience"],
        [
            ["1", "Keyword search and similar recipes from the same cluster"],
            ["2", "Graph-based ingredient pairing"],
            ["3", "Personalized hybrid ML recommendations"],
            ["0", "Exit"],
        ],
    )


def main() -> None:
    try:
        context = load_context()
    except (OSError, ValueError, FileNotFoundError) as error:
        print_message(f"Demo could not start: {error}", "red")
        return

    print_message(f"Loaded {len(context.recipes):,} processed recipes.", "green")
    while True:
        print_menu()
        choice = ask("Choose an option: ")
        if choice == "1":
            handle_search(context)
        elif choice == "2":
            handle_graph_recommendation(context)
        elif choice == "3":
            handle_hybrid_recommendation(context)
        elif choice == "0":
            print_message("Thanks for exploring the Recipe Intelligence System.", "green")
            return
        else:
            print_message("Choose 1, 2, 3, or 0.", "yellow")


if __name__ == "__main__":
    main()
