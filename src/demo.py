#!/usr/bin/env python3
"""Terminal wrapper for the Recipe Recommendation Project demo."""

from __future__ import annotations

import numpy as np

try:
    from demo_core import (
        active_user_ids,
        cluster_recommendations,
        fit_hybrid_models,
        get_cluster_profile,
        graph_ingredient_options,
        graph_neighbors,
        graph_pair_recommendations,
        load_context,
        load_graph_artifacts,
        recommend_hybrid_recipes,
        search_recipes,
        user_history,
    )
except ModuleNotFoundError:
    from src.demo_core import (
        active_user_ids,
        cluster_recommendations,
        fit_hybrid_models,
        get_cluster_profile,
        graph_ingredient_options,
        graph_neighbors,
        graph_pair_recommendations,
        load_context,
        load_graph_artifacts,
        recommend_hybrid_recipes,
        search_recipes,
        user_history,
    )


def show_recipes(recipes, title: str) -> None:
    print(f"\n{title}")
    for index, (_, row) in enumerate(recipes.iterrows(), start=1):
        print(f"{index}. {row['Name']} | {row.get('RecipeCategory', 'Unknown')} | rating={row.get('AggregatedRating', 'n/a')}")


def choose_index(count: int, prompt: str) -> int | None:
    value = input(prompt).strip()
    if not value or value == "0":
        return None
    try:
        selected = int(value) - 1
    except ValueError:
        return None
    return selected if 0 <= selected < count else None


def handle_search(context) -> None:
    query = input("Recipe keyword (blank to return): ").strip()
    matches = search_recipes(context, query)
    if matches.empty:
        print("No recipes found.")
        return
    show_recipes(matches, "Search results")
    selected_index = choose_index(len(matches), "Select a recipe (0 to return): ")
    if selected_index is None:
        return
    selected = matches.iloc[selected_index]
    cluster_info = get_cluster_profile(context, int(selected["RecipeId"]))
    if cluster_info is None:
        print("No cluster assignment is available for this recipe.")
        return
    cluster_id, profile = cluster_info
    name = profile.get("dominant_category", f"Cluster {cluster_id}") if profile is not None else f"Cluster {cluster_id}"
    print(f"Because you liked {selected['Name']}, here are recipes from {name} (Cluster {cluster_id}).")
    show_recipes(cluster_recommendations(context, int(selected["RecipeId"])), "Similar cluster recipes")


def handle_graph(context) -> None:
    try:
        nodes, edges = load_graph_artifacts(context)
    except FileNotFoundError as error:
        print(error)
        return
    raw = input("Ingredient (blank to return): ").strip()
    options = graph_ingredient_options(nodes, raw)
    if options.empty:
        print("No graph ingredient found.")
        return
    ingredient = str(options.iloc[0]["ingredient"])
    neighbors = graph_neighbors(nodes, edges, ingredient)
    recommendations = graph_pair_recommendations(context, ingredient, neighbors)
    print(f"Central partners for {ingredient}: {', '.join(neighbors['neighbor'].tolist())}")
    show_recipes(recommendations, "Recipes suggested by graph pairings")


def handle_hybrid(context, state):
    if state is None:
        print("Training the hybrid model on first use; this can take a moment.")
        try:
            state = fit_hybrid_models()
        except (FileNotFoundError, ImportError, MemoryError, ValueError) as error:
            print(f"Could not prepare the hybrid recommender: {error}")
            return None
    raw = input("AuthorId (blank for a random active user): ").strip()
    users = active_user_ids(state)
    try:
        author_id = int(np.random.default_rng(42).choice(users)) if not raw else int(raw)
        show_recipes(user_history(context, state, author_id), "Recent rated recipes")
        show_recipes(recommend_hybrid_recipes(context, state, author_id), "Top hybrid recommendations")
    except ValueError as error:
        print(f"Could not recommend recipes: {error}")
    return state


def main() -> None:
    context = load_context()
    hybrid_state = None
    while True:
        print("\n1. Search + cluster  2. Ingredient graph  3. Hybrid recommendations  0. Exit")
        choice = input("Choose an option: ").strip()
        if choice == "1":
            handle_search(context)
        elif choice == "2":
            handle_graph(context)
        elif choice == "3":
            hybrid_state = handle_hybrid(context, hybrid_state)
        elif choice == "0":
            return


if __name__ == "__main__":
    main()
