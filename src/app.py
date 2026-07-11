"""Streamlit interface for exploring Recipe Recommendation Project artifacts."""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st

try:
    from demo_core import (
        active_user_ids,
        cluster_recommendations,
        first_recipe_image,
        fit_hybrid_models,
        get_cluster_profile,
        graph_ingredient_options,
        graph_neighbors,
        graph_pair_recommendations,
        load_context,
        load_graph_artifacts,
        recommend_hybrid_recipes,
        safe_parse_list,
        search_recipes,
        user_history,
    )
except ModuleNotFoundError:
    from src.demo_core import (
        active_user_ids,
        cluster_recommendations,
        first_recipe_image,
        fit_hybrid_models,
        get_cluster_profile,
        graph_ingredient_options,
        graph_neighbors,
        graph_pair_recommendations,
        load_context,
        load_graph_artifacts,
        recommend_hybrid_recipes,
        safe_parse_list,
        search_recipes,
        user_history,
    )


st.set_page_config(page_title="Recipe Intelligence", page_icon="R", layout="wide")


@st.cache_data(show_spinner="Loading recipe and clustering artifacts...")
def cached_context():
    return load_context()


@st.cache_resource(show_spinner="Training the hybrid recommender for this app session...")
def cached_hybrid_state():
    return fit_hybrid_models()


def recipe_label(row: pd.Series) -> str:
    return f"{row['Name']} ({row['RecipeCategory'] or 'Uncategorized'})"


def compact_recipe_frame(recipes: pd.DataFrame) -> pd.DataFrame:
    columns = ["Name", "RecipeCategory", "AggregatedRating", "ReviewCount", "TotalTime_Minutes"]
    available = [column for column in columns if column in recipes.columns]
    frame = recipes[available].copy()
    return frame.rename(
        columns={
            "Name": "Recipe",
            "RecipeCategory": "Category",
            "AggregatedRating": "Rating",
            "ReviewCount": "Reviews",
            "TotalTime_Minutes": "Minutes",
        }
    )


def render_recipe_details(recipe: pd.Series) -> None:
    image_url = first_recipe_image(recipe.get("Images"))
    image_column, detail_column = st.columns([1, 2], gap="large")
    with image_column:
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            st.caption("No recipe image is available.")
    with detail_column:
        st.subheader(str(recipe["Name"]))
        st.caption(str(recipe.get("RecipeCategory", "Uncategorized")))
        metric_columns = st.columns(4)
        metric_columns[0].metric("Rating", f"{pd.to_numeric(recipe.get('AggregatedRating'), errors='coerce'):.1f}")
        review_count = pd.to_numeric(recipe.get("ReviewCount"), errors="coerce")
        metric_columns[1].metric("Reviews", int(review_count) if pd.notna(review_count) else 0)
        metric_columns[2].metric("Minutes", f"{pd.to_numeric(recipe.get('TotalTime_Minutes'), errors='coerce'):.0f}")
        metric_columns[3].metric("Calories", f"{pd.to_numeric(recipe.get('Calories'), errors='coerce'):.0f}")
        description = recipe.get("Description")
        if pd.notna(description) and str(description).strip():
            st.write(str(description))

    ingredients = list(zip(recipe_ingredients(recipe), recipe_quantities(recipe)))
    with st.expander("Ingredients", expanded=True):
        if ingredients:
            st.dataframe(
                pd.DataFrame(ingredients, columns=["Ingredient", "Quantity"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Ingredients are not available for this recipe.")
    with st.expander("Instructions"):
        steps = recipe_instructions(recipe)
        if steps:
            for number, step in enumerate(steps, start=1):
                st.markdown(f"{number}. {step}")
        else:
            st.info("Instructions are not available for this recipe.")


def recipe_ingredients(recipe: pd.Series) -> list[str]:
    return safe_parse_list(recipe.get("RecipeIngredientParts"))


def recipe_quantities(recipe: pd.Series) -> list[str]:
    quantities = safe_parse_list(recipe.get("RecipeIngredientQuantities"))
    return quantities + [""] * max(0, len(recipe_ingredients(recipe)) - len(quantities))


def recipe_instructions(recipe: pd.Series) -> list[str]:
    return safe_parse_list(recipe.get("RecipeInstructions"))


def render_recommendation_rows(recipes: pd.DataFrame, heading: str, include_graph_pair: bool = False) -> None:
    st.subheader(heading)
    if recipes.empty:
        st.info("No recipes matched this recommendation path.")
        return
    columns = ["Name", "RecipeCategory", "AggregatedRating", "ReviewCount"]
    if include_graph_pair:
        columns.insert(1, "graph_partner")
    if "hybrid_score" in recipes.columns:
        columns.extend(["hybrid_score", "cf_score", "content_score"])
    st.dataframe(recipes[[column for column in columns if column in recipes.columns]], hide_index=True, use_container_width=True)
    for _, recipe in recipes.head(5).iterrows():
        with st.expander(f"View {recipe['Name']}"):
            render_recipe_details(recipe)


def render_search_tab(context) -> None:
    st.header("Recipe Search")
    st.caption("Find a recipe, inspect its details, then explore similar recipes from its discovered cluster.")
    query = st.text_input("Search recipe titles", placeholder="e.g. chocolate")
    if not query.strip():
        return
    matches = search_recipes(context, query)
    if matches.empty:
        st.warning(f"No recipe titles matched '{query}'.")
        return
    st.dataframe(compact_recipe_frame(matches), hide_index=True, use_container_width=True)
    selected_recipe_id = st.selectbox(
        "Select a recipe", matches["RecipeId"].tolist(), format_func=lambda recipe_id: recipe_label(
            matches.loc[matches["RecipeId"].eq(recipe_id)].iloc[0]
        )
    )
    selected = matches.loc[matches["RecipeId"].eq(selected_recipe_id)].iloc[0]
    render_recipe_details(selected)

    cluster_info = get_cluster_profile(context, int(selected_recipe_id))
    if cluster_info is None:
        st.warning("No clustering assignment is available for this recipe.")
        return
    cluster_id, profile = cluster_info
    profile_name = profile.get("dominant_category", f"Cluster {cluster_id}") if profile is not None else f"Cluster {cluster_id}"
    st.subheader(f"Recipe Group: {profile_name} (Cluster {cluster_id})")
    if profile is not None:
        st.write(str(profile.get("interpretation_summary", "No cluster interpretation is available.")))
        metrics = st.columns(3)
        metrics[0].metric("Recipes", f"{int(profile.get('recipe_count', 0)):,}")
        metrics[1].metric("Average calories", f"{float(profile.get('avg_calories', 0)):.0f}")
        metrics[2].metric("Average minutes", f"{float(profile.get('avg_total_time', 0)):.0f}")
    render_recommendation_rows(
        cluster_recommendations(context, int(selected_recipe_id)),
        f"Because you selected {selected['Name']}, explore these highly rated recipes from the same group",
    )


def render_graph_tab(context) -> None:
    st.header("Ingredient Pairing")
    st.caption("Enter one ingredient. The graph proposes central ingredients that co-occur with it in recipes.")
    try:
        nodes, edges = load_graph_artifacts(context)
    except FileNotFoundError as error:
        st.error(f"Graph data not found: {error}")
        return
    raw_ingredient = st.text_input("Ingredient", placeholder="e.g. chicken", key="graph_ingredient")
    if not raw_ingredient.strip():
        return
    options = graph_ingredient_options(nodes, raw_ingredient)
    if options.empty:
        st.warning(f"No graph ingredient matched '{raw_ingredient}'.")
        return
    selected_ingredient = st.selectbox(
        "Graph ingredient", options["ingredient"].tolist(), format_func=lambda ingredient: ingredient.replace("_", " ").title()
    )
    neighbors = graph_neighbors(nodes, edges, selected_ingredient)
    if neighbors.empty:
        st.warning("This ingredient has no retained graph neighbors at the current edge threshold.")
        return
    st.subheader("Central pairing ingredients")
    st.dataframe(
        neighbors.rename(
            columns={"neighbor": "Ingredient", "neighbor_pagerank": "PageRank", "cooccurrence_count": "Co-occurrences"}
        ),
        hide_index=True,
        use_container_width=True,
    )
    recommendations = graph_pair_recommendations(context, selected_ingredient, neighbors)
    render_recommendation_rows(
        recommendations,
        f"Recipes containing {selected_ingredient.replace('_', ' ')} with one of these graph partners",
        include_graph_pair=True,
    )


def render_hybrid_tab(context) -> None:
    st.header("Personalized Hybrid Recommendations")
    st.caption("Uses Collaborative SVD (60%) and content similarity (40%) to rank unseen recipes for an active AuthorId.")
    load_model = st.button("Load hybrid model", type="primary")
    if load_model:
        try:
            with st.status("Preparing the hybrid model", expanded=True) as status:
                st.write("Loading reviews and creating the 5-core training subset...")
                state = cached_hybrid_state()
                status.update(label="Hybrid model ready", state="complete", expanded=False)
            st.session_state["hybrid_ready"] = True
            st.session_state["hybrid_state"] = state
        except (FileNotFoundError, ImportError, MemoryError, ValueError) as error:
            st.error(f"Could not prepare the hybrid recommender: {error}")
            return
    if not st.session_state.get("hybrid_ready"):
        st.info("The model trains only after you choose Load hybrid model, then stays cached for this app session.")
        return

    state = st.session_state.get("hybrid_state") or cached_hybrid_state()
    users = active_user_ids(state)
    use_random = st.checkbox("Use a reproducible random active user", value=True)
    if use_random:
        author_id = int(np.random.default_rng(42).choice(users))
        st.caption(f"Selected AuthorId: {author_id}")
    else:
        author_id = int(st.number_input("AuthorId", min_value=min(users), max_value=max(users), step=1))
    if author_id not in set(users):
        st.warning("This AuthorId is not an active user in the trained 5-core dataset.")
        return
    if not st.button("Recommend recipes", key="recommend_hybrid"):
        return
    history = user_history(context, state, author_id)
    st.subheader("Recent rated recipes")
    st.dataframe(history[["Name", "Rating", "DateSubmitted"]], hide_index=True, use_container_width=True)
    with st.spinner("Scoring eligible unseen recipes..."):
        recommendations = recommend_hybrid_recipes(context, state, author_id)
    render_recommendation_rows(recommendations, "Top 3 hybrid recommendations")


def main() -> None:
    st.title("Recipe Intelligence Explorer")
    st.caption("Search the Food.com catalog, inspect recipe groups, explore ingredient networks, and test the hybrid recommender.")
    try:
        context = cached_context()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"The app could not load processed recipes: {error}")
        st.stop()
    st.sidebar.metric("Processed recipes", f"{len(context.recipes):,}")
    st.sidebar.caption("Artifacts are read from the existing Week 5, Week 7, and Week 12 pipeline outputs.")
    search_tab, graph_tab, hybrid_tab = st.tabs(["Recipe Search", "Ingredient Pairing", "Hybrid Recommendations"])
    with search_tab:
        render_search_tab(context)
    with graph_tab:
        render_graph_tab(context)
    with hybrid_tab:
        render_hybrid_tab(context)


if __name__ == "__main__":
    main()
