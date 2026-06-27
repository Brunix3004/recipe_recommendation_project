#!/usr/bin/env python3
"""
Build the Week 12 ingredient-ingredient co-occurrence graph deliverable.

The graph is induced only from processed recipe ingredient lists. It does not
use ratings, review counts, aggregated ratings, notebook state, or raw data.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from itertools import combinations
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib-cache"))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/codex-cache"))
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


INGREDIENT_COL = "RecipeIngredientParts"
INVALID_TOKENS = {"", "nan", "none", "null", "missing", "character_0"}

GRAPH_DEFINITION = {
    "graph_name": "Ingredient-Ingredient Co-occurrence Graph",
    "graph_type": "undirected weighted graph",
    "nodes": "Normalized ingredients extracted from RecipeIngredientParts.",
    "node_grain": "One node represents one normalized ingredient token.",
    "edges": (
        "An undirected edge connects two ingredients if they co-occur in at "
        "least one recipe after ingredient normalization and within-recipe "
        "deduplication."
    ),
    "edge_weight": (
        "The number of distinct recipes in which the two ingredients "
        "co-occur."
    ),
    "directionality": (
        "Undirected, because ingredient co-occurrence inside a recipe has no "
        "natural direction."
    ),
    "recipe_contribution_rule": (
        "Each recipe contributes at most one count to an ingredient node and "
        "at most one count to a given ingredient pair."
    ),
    "primary_graph_ranking": "Weighted PageRank and weighted degree.",
    "popularity_baseline": (
        "Ingredient recipe frequency, defined as the number of distinct "
        "recipes containing the ingredient."
    ),
}

GENERIC_INGREDIENTS = {
    "salt",
    "sugar",
    "white_sugar",
    "granulated_sugar",
    "brown_sugar",
    "powdered_sugar",
    "water",
    "butter",
    "unsalted_butter",
    "margarine",
    "flour",
    "all_purpose_flour",
    "eggs",
    "egg",
    "milk",
    "whole_milk",
    "olive_oil",
    "vegetable_oil",
    "canola_oil",
    "pepper",
    "black_pepper",
    "ground_black_pepper",
    "onion",
    "onions",
    "garlic",
    "garlic_clove",
    "garlic_cloves",
    "vanilla",
    "vanilla_extract",
    "baking_powder",
    "baking_soda",
    "lemon_juice",
    "fresh_lemon_juice",
}

ARTIFACT_FILES = [
    "ingredient_nodes.csv",
    "ingredient_edges.csv",
    "ingredient_graph.graphml",
    "ingredient_graph_summary.json",
    "connected_components.csv",
    "graph_validity_checks.csv",
    "comparison_graph_vs_popularity.csv",
    "top_ingredients_by_popularity.csv",
    "top_ingredients_by_weighted_degree.csv",
    "top_ingredients_by_pagerank.csv",
    "top_ingredients_by_log_pagerank.csv",
    "sensitivity_edge_thresholds.csv",
    "sensitivity_top20_pagerank_overlap.csv",
    "graph_pipeline_config.json",
]

FIGURE_FILES = [
    "ingredient_graph_degree_distribution.png",
    "ingredient_graph_weighted_degree_distribution.png",
    "ingredient_graph_component_size_distribution.png",
    "ingredient_graph_pagerank_vs_popularity.png",
    "ingredient_graph_top_pagerank.png",
    "ingredient_graph_top_weighted_degree.png",
    "ingredient_graph_sensitivity_edges.png",
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reproducible ingredient co-occurrence graph."
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=Path("data/processed/recipes_processed.csv"),
        help="CSV or Parquet file containing RecipeIngredientParts.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/week12/ingredient_graph"),
        help="Output directory for graph artifacts.",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("reports/figures"),
        help="Output directory for graph figures.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/week12_graph_analytics_report.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--min-node-recipe-count",
        type=int,
        default=50,
        help="Minimum distinct recipes required to retain an ingredient node.",
    )
    parser.add_argument(
        "--min-edge-recipe-count",
        type=int,
        default=20,
        help="Minimum distinct co-occurring recipes required to retain an edge.",
    )
    parser.add_argument(
        "--sensitivity-edge-thresholds",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Edge thresholds to evaluate in sensitivity analysis.",
    )
    parser.add_argument(
        "--compute-approx-betweenness",
        action="store_true",
        help="Optionally compute sampled approximate betweenness centrality.",
    )
    parser.add_argument(
        "--approx-betweenness-samples",
        type=int,
        default=200,
        help="Number of sampled nodes for approximate betweenness if enabled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for optional sampled centrality.",
    )
    return parser.parse_args()


def require_positive_thresholds(args: argparse.Namespace) -> None:
    if args.min_node_recipe_count < 1:
        raise ValueError("--min-node-recipe-count must be >= 1.")
    if args.min_edge_recipe_count < 1:
        raise ValueError("--min-edge-recipe-count must be >= 1.")
    if any(threshold < 1 for threshold in args.sensitivity_edge_thresholds):
        raise ValueError("All --sensitivity-edge-thresholds values must be >= 1.")


def read_recipes(path: Path) -> pd.DataFrame:
    """Load only the ingredient column from processed CSV or Parquet input."""
    if not path.exists():
        raise FileNotFoundError(f"Recipe input not found: {path}")

    suffix = path.suffix.lower()
    logger.info("Loading recipe ingredient column from %s", path)

    if suffix == ".parquet":
        try:
            return pd.read_parquet(path, columns=[INGREDIENT_COL])
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet requires pyarrow or fastparquet. Install one "
                "of those packages or use data/processed/recipes_processed.csv."
            ) from exc
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Required column {INGREDIENT_COL!r} is missing from {path}."
            ) from exc

    header = pd.read_csv(path, nrows=0)
    if INGREDIENT_COL not in header.columns:
        raise ValueError(f"Required column {INGREDIENT_COL!r} is missing from {path}.")
    return pd.read_csv(path, usecols=[INGREDIENT_COL], dtype={INGREDIENT_COL: "string"})


def safe_parse_list(value: Any) -> List[str]:
    """Parse Food.com list-like ingredient cells without failing malformed rows."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, set):
        return [str(item) for item in value]
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []
    if re.fullmatch(r"character\s*\(\s*0\s*\)", text, flags=re.IGNORECASE):
        return []
    r_vector_match = re.fullmatch(r"c\s*\((.*)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if r_vector_match:
        inner = r_vector_match.group(1).strip()
        if not inner:
            return []
        try:
            parsed_vector = ast.literal_eval(f"[{inner}]")
            if isinstance(parsed_vector, list):
                return [str(item) for item in parsed_vector]
        except (SyntaxError, ValueError):
            pass

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]

    if parsed is None:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, tuple):
        return [str(item) for item in parsed]
    if isinstance(parsed, set):
        return [str(item) for item in sorted(parsed)]
    return [str(parsed)]


def normalize_ingredient(token: str) -> str:
    """Normalize one ingredient token to the graph node grain."""
    token = str(token).strip().lower()
    if token in INVALID_TOKENS:
        return ""

    token = token.replace("&", " and ")
    token = token.replace("'", "")
    token = re.sub(r"[^\w\s-]+", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    token = re.sub(r"[\s-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")

    if token in INVALID_TOKENS:
        return ""
    return token


def parse_normalized_ingredients(value: Any) -> List[str]:
    tokens = [normalize_ingredient(token) for token in safe_parse_list(value)]
    return sorted({token for token in tokens if token})


def collect_ingredient_counts(
    ingredient_values: Iterable[Any],
) -> Tuple[Counter, Counter, Dict[str, int]]:
    """
    Count ingredient recipe frequency and ingredient-pair co-occurrence.

    Within each recipe, ingredients are deduplicated and sorted before pair
    generation so that each recipe contributes at most one count per node and
    one count per unordered ingredient pair.
    """
    node_counter: Counter = Counter()
    pair_counter: Counter = Counter()
    recipes_processed = 0
    recipes_with_one = 0
    recipes_with_two = 0

    for idx, value in enumerate(ingredient_values, start=1):
        ingredients = parse_normalized_ingredients(value)
        recipes_processed += 1

        if ingredients:
            recipes_with_one += 1
            node_counter.update(ingredients)
        if len(ingredients) >= 2:
            recipes_with_two += 1
            pair_counter.update(combinations(ingredients, 2))

        if idx % 50000 == 0:
            logger.info("Parsed %s recipes...", f"{idx:,}")

    stats = {
        "recipes_processed": recipes_processed,
        "recipes_with_at_least_1_valid_ingredient": recipes_with_one,
        "recipes_with_at_least_2_valid_ingredients": recipes_with_two,
        "raw_unique_ingredients_before_filtering": len(node_counter),
        "raw_unique_ingredient_pairs_before_filtering": len(pair_counter),
    }
    return node_counter, pair_counter, stats


def build_graph(
    node_counter: Counter,
    pair_counter: Counter,
    min_node_recipe_count: int,
    min_edge_recipe_count: int,
) -> nx.Graph:
    retained_nodes = {
        ingredient: int(recipe_count)
        for ingredient, recipe_count in node_counter.items()
        if recipe_count >= min_node_recipe_count
    }

    graph = nx.Graph()
    graph.graph["name"] = GRAPH_DEFINITION["graph_name"]
    graph.graph["graph_type"] = GRAPH_DEFINITION["graph_type"]
    graph.graph["min_node_recipe_count"] = int(min_node_recipe_count)
    graph.graph["min_edge_recipe_count"] = int(min_edge_recipe_count)

    for ingredient in sorted(retained_nodes):
        graph.add_node(
            ingredient,
            ingredient=ingredient,
            recipe_count=int(retained_nodes[ingredient]),
        )

    for (source, target), count in sorted(pair_counter.items()):
        if count < min_edge_recipe_count:
            continue
        if source not in retained_nodes or target not in retained_nodes:
            continue

        freq_source = retained_nodes[source]
        freq_target = retained_nodes[target]
        jaccard = float(count / (freq_source + freq_target - count))
        graph.add_edge(
            source,
            target,
            cooccurrence_count=int(count),
            weight=int(count),
            log_weight=float(np.log1p(count)),
            jaccard=jaccard,
        )

    return graph


def sorted_components(graph: nx.Graph) -> List[Set[str]]:
    return sorted(
        (set(component) for component in nx.connected_components(graph)),
        key=lambda component: (-len(component), sorted(component)[0] if component else ""),
    )


def compute_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    density = float(nx.density(graph)) if n_nodes > 1 else 0.0
    sparsity = float(1.0 - density)
    degrees = dict(graph.degree())
    weighted_degrees = dict(graph.degree(weight="weight"))
    components = sorted_components(graph) if n_nodes else []
    largest_component_size = len(components[0]) if components else 0
    isolated_node_count = sum(1 for _, degree in graph.degree() if degree == 0)

    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "density": density,
        "sparsity": sparsity,
        "connected_components": int(len(components)),
        "largest_component_size": int(largest_component_size),
        "largest_component_share": float(largest_component_size / n_nodes) if n_nodes else 0.0,
        "isolated_node_count": int(isolated_node_count),
        "isolated_node_share": float(isolated_node_count / n_nodes) if n_nodes else 0.0,
        "average_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "average_weighted_degree": (
            float(np.mean(list(weighted_degrees.values()))) if weighted_degrees else 0.0
        ),
        "top_component_sizes": [int(len(component)) for component in components[:10]],
    }


def compute_component_lookup(graph: nx.Graph) -> Tuple[Dict[str, int], Dict[int, int]]:
    node_to_component: Dict[str, int] = {}
    component_sizes: Dict[int, int] = {}
    for component_id, component in enumerate(sorted_components(graph), start=1):
        component_sizes[component_id] = len(component)
        for node in component:
            node_to_component[node] = component_id
    return node_to_component, component_sizes


def run_pagerank(graph: nx.Graph, weight: str) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(graph, alpha=0.85, weight=weight, max_iter=200, tol=1e-10)
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank did not converge quickly for weight=%s; retrying.", weight)
        return nx.pagerank(graph, alpha=0.85, weight=weight, max_iter=1000, tol=1e-8)


def compute_node_metrics(
    graph: nx.Graph,
    compute_approx_betweenness: bool = False,
    approx_betweenness_samples: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    log_weighted_degree = dict(graph.degree(weight="log_weight"))
    pagerank_weighted = run_pagerank(graph, weight="weight")
    pagerank_log_weighted = run_pagerank(graph, weight="log_weight")
    node_to_component, component_sizes = compute_component_lookup(graph)

    total_degree = float(sum(degree.values()))
    total_weighted_degree = float(sum(weighted_degree.values()))

    approx_betweenness: Dict[str, float] = {}
    if compute_approx_betweenness and graph.number_of_nodes() > 0:
        k = min(int(approx_betweenness_samples), graph.number_of_nodes())
        logger.info("Computing sampled approximate betweenness with k=%s.", k)
        approx_betweenness = nx.betweenness_centrality(
            graph,
            k=k,
            normalized=True,
            weight=None,
            seed=seed,
        )

    rows = []
    for ingredient in sorted(graph.nodes()):
        component_id = int(node_to_component.get(ingredient, 0))
        row = {
            "ingredient": ingredient,
            "recipe_count": int(graph.nodes[ingredient].get("recipe_count", 0)),
            "degree": int(degree.get(ingredient, 0)),
            "weighted_degree": float(weighted_degree.get(ingredient, 0.0)),
            "log_weighted_degree": float(log_weighted_degree.get(ingredient, 0.0)),
            "pagerank_weighted": float(pagerank_weighted.get(ingredient, 0.0)),
            "pagerank_log_weighted": float(pagerank_log_weighted.get(ingredient, 0.0)),
            "component_id": component_id,
            "component_size": int(component_sizes.get(component_id, 0)),
            "degree_share": (
                float(degree.get(ingredient, 0) / total_degree) if total_degree else 0.0
            ),
            "weighted_degree_share": (
                float(weighted_degree.get(ingredient, 0.0) / total_weighted_degree)
                if total_weighted_degree
                else 0.0
            ),
        }
        if approx_betweenness:
            row["approx_betweenness"] = float(approx_betweenness.get(ingredient, 0.0))
        rows.append(row)

    nodes_df = pd.DataFrame(rows)

    for _, row in nodes_df.iterrows():
        node = row["ingredient"]
        for column in nodes_df.columns:
            value = row[column]
            if isinstance(value, (np.integer, int)):
                graph.nodes[node][column] = int(value)
            elif isinstance(value, (np.floating, float)):
                graph.nodes[node][column] = float(value)
            else:
                graph.nodes[node][column] = str(value)

    return nodes_df


def edges_to_dataframe(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for source, target, attrs in graph.edges(data=True):
        ordered_source, ordered_target = sorted([source, target])
        rows.append(
            {
                "source": ordered_source,
                "target": ordered_target,
                "cooccurrence_count": int(attrs.get("cooccurrence_count", 0)),
                "weight": int(attrs.get("weight", 0)),
                "log_weight": float(attrs.get("log_weight", 0.0)),
                "jaccard": float(attrs.get("jaccard", 0.0)),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["source", "target", "cooccurrence_count", "weight", "log_weight", "jaccard"]
        )
    return pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True)


def add_rank_columns(nodes_df: pd.DataFrame) -> pd.DataFrame:
    ranked = nodes_df.copy()
    rank_specs = [
        ("recipe_count", "popularity_rank"),
        ("weighted_degree", "weighted_degree_rank"),
        ("pagerank_weighted", "pagerank_rank"),
        ("pagerank_log_weighted", "log_pagerank_rank"),
    ]
    for metric, rank_col in rank_specs:
        order = ranked.sort_values([metric, "ingredient"], ascending=[False, True]).index
        ranks = pd.Series(np.arange(1, len(order) + 1), index=order)
        ranked[rank_col] = ranks
    return ranked


def top_by_metric(nodes_df: pd.DataFrame, metric: str, n: int = 100) -> pd.DataFrame:
    columns = [
        "ingredient",
        "recipe_count",
        "degree",
        "weighted_degree",
        "pagerank_weighted",
        "pagerank_log_weighted",
        "component_id",
        "component_size",
    ]
    rank_col = {
        "recipe_count": "popularity_rank",
        "weighted_degree": "weighted_degree_rank",
        "pagerank_weighted": "pagerank_rank",
        "pagerank_log_weighted": "log_pagerank_rank",
    }[metric]
    out = nodes_df.sort_values([metric, "ingredient"], ascending=[False, True]).head(n).copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[["rank"] + columns + [rank_col]]


def build_connected_components_df(graph: nx.Graph, nodes_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_nodes = graph.number_of_nodes()
    for component_id, component in enumerate(sorted_components(graph), start=1):
        component_nodes = nodes_df[nodes_df["ingredient"].isin(component)]
        top_recipe_count = component_nodes.sort_values(
            ["recipe_count", "ingredient"], ascending=[False, True]
        )["ingredient"].head(10)
        top_pagerank = component_nodes.sort_values(
            ["pagerank_weighted", "ingredient"], ascending=[False, True]
        )["ingredient"].head(10)
        rows.append(
            {
                "component_id": int(component_id),
                "component_size": int(len(component)),
                "component_share": float(len(component) / total_nodes) if total_nodes else 0.0,
                "top_ingredients_by_recipe_count": "; ".join(top_recipe_count),
                "top_ingredients_by_pagerank": "; ".join(top_pagerank),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "component_id",
            "component_size",
            "component_share",
            "top_ingredients_by_recipe_count",
            "top_ingredients_by_pagerank",
        ],
    )


def safe_spearman(df: pd.DataFrame, left: str, right: str) -> float:
    if len(df) < 2:
        return float("nan")
    if df[left].nunique(dropna=True) < 2 or df[right].nunique(dropna=True) < 2:
        return float("nan")
    return float(df[left].corr(df[right], method="spearman"))


def top_k_set(nodes_df: pd.DataFrame, metric: str, k: int) -> Set[str]:
    return set(
        nodes_df.sort_values([metric, "ingredient"], ascending=[False, True])
        .head(k)["ingredient"]
        .tolist()
    )


def build_comparison_df(nodes_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    corr_specs = [
        (
            "recipe_count",
            "weighted_degree",
            "Spearman correlation between recipe frequency and weighted degree.",
        ),
        (
            "recipe_count",
            "pagerank_weighted",
            "Spearman correlation between recipe frequency and weighted PageRank.",
        ),
    ]
    for baseline_metric, graph_metric, description in corr_specs:
        rows.append(
            {
                "comparison_type": "spearman_correlation",
                "baseline_metric": baseline_metric,
                "graph_metric": graph_metric,
                "k": np.nan,
                "value": safe_spearman(nodes_df, baseline_metric, graph_metric),
                "overlap_count": np.nan,
                "overlap_share": np.nan,
                "description": description,
            }
        )

    popularity_sets = {k: top_k_set(nodes_df, "recipe_count", k) for k in [10, 20, 50, 100]}
    for graph_metric in ["pagerank_weighted", "weighted_degree"]:
        for k in [10, 20, 50, 100]:
            graph_set = top_k_set(nodes_df, graph_metric, k)
            overlap = popularity_sets[k] & graph_set
            rows.append(
                {
                    "comparison_type": "top_k_overlap",
                    "baseline_metric": "recipe_count",
                    "graph_metric": graph_metric,
                    "k": int(k),
                    "value": float(len(overlap) / k) if k else 0.0,
                    "overlap_count": int(len(overlap)),
                    "overlap_share": float(len(overlap) / k) if k else 0.0,
                    "description": (
                        f"Top-{k} overlap between popularity and {graph_metric} ranking."
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_validity_checks(
    summary: Mapping[str, Any],
    counts_stats: Mapping[str, Any],
    nodes_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    def add_row(check: str, value: Any, status: str, note: str) -> None:
        rows.append({"check": check, "value": value, "status": status, "note": note})

    add_row(
        "recipes_processed",
        counts_stats["recipes_processed"],
        "ok",
        "Number of processed recipe rows loaded from the input file.",
    )
    add_row(
        "recipes_with_at_least_1_valid_ingredient",
        counts_stats["recipes_with_at_least_1_valid_ingredient"],
        "ok",
        "Rows with at least one parsed and normalized ingredient token.",
    )
    add_row(
        "recipes_with_at_least_2_valid_ingredients",
        counts_stats["recipes_with_at_least_2_valid_ingredients"],
        "ok",
        "Rows able to contribute at least one ingredient pair.",
    )
    add_row(
        "raw_unique_ingredients_before_filtering",
        counts_stats["raw_unique_ingredients_before_filtering"],
        "ok",
        "Ingredient vocabulary before min-node filtering.",
    )
    add_row(
        "retained_ingredient_nodes_after_filtering",
        summary["retained_node_count"],
        "ok",
        "Ingredient nodes with recipe_count above the selected threshold.",
    )
    add_row(
        "raw_unique_ingredient_pairs_before_filtering",
        counts_stats["raw_unique_ingredient_pairs_before_filtering"],
        "ok",
        "Unique unordered ingredient pairs before min-edge filtering.",
    )
    add_row(
        "retained_edges_after_filtering",
        summary["retained_edge_count"],
        "ok",
        "Ingredient pairs with co-occurrence counts above the selected threshold.",
    )
    add_row(
        "graph_density",
        summary["density"],
        "warning" if summary["density"] > 0.10 else "ok",
        "A high density would indicate that the edge threshold is too permissive.",
    )
    add_row(
        "graph_sparsity",
        summary["sparsity"],
        "ok",
        "Graph sparsity is 1 - density.",
    )
    add_row(
        "isolated_node_count",
        summary["isolated_node_count"],
        "warning" if summary["isolated_node_share"] > 0.25 else "ok",
        "Retained nodes that have no retained edges after edge filtering.",
    )
    add_row(
        "isolated_node_share",
        summary["isolated_node_share"],
        "warning" if summary["isolated_node_share"] > 0.25 else "ok",
        "Share of retained ingredient nodes that are isolated.",
    )
    add_row(
        "connected_components",
        summary["connected_components"],
        "warning" if summary["largest_component_share"] < 0.60 else "ok",
        "Fragmentation warning triggers if the largest component is below 60% of nodes.",
    )
    add_row(
        "largest_component_share",
        summary["largest_component_share"],
        "warning" if summary["largest_component_share"] < 0.60 else "ok",
        "Share of retained nodes in the largest connected component.",
    )

    top10 = (
        nodes_df.sort_values(["pagerank_weighted", "ingredient"], ascending=[False, True])
        .head(10)["ingredient"]
        .tolist()
    )
    generic_count = sum(ingredient in GENERIC_INGREDIENTS for ingredient in top10)
    generic_share = float(generic_count / len(top10)) if top10 else 0.0
    add_row(
        "generic_ingredient_dominance_top10_pagerank",
        generic_share,
        "warning" if generic_share >= 0.80 else "ok",
        (
            "Warning triggers if at least 80% of top-10 PageRank ingredients "
            "are generic pantry staples."
        ),
    )

    return pd.DataFrame(rows)


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=default)


def save_empty_figure(path: Path, title: str, message: str = "No data") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_degree_distribution(nodes_df: pd.DataFrame, path: Path) -> None:
    values = nodes_df["degree"].to_numpy() if not nodes_df.empty else np.array([])
    if values.size == 0:
        save_empty_figure(path, "Ingredient Graph Degree Distribution")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50)
    ax.set_yscale("log")
    ax.set_title("Ingredient Graph Degree Distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Ingredient count (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_weighted_degree_distribution(nodes_df: pd.DataFrame, path: Path) -> None:
    values = nodes_df["weighted_degree"].to_numpy() if not nodes_df.empty else np.array([])
    if values.size == 0:
        save_empty_figure(path, "Ingredient Graph Weighted Degree Distribution")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=50)
    ax.set_yscale("log")
    ax.set_title("Ingredient Graph Weighted Degree Distribution")
    ax.set_xlabel("Weighted degree")
    ax.set_ylabel("Ingredient count (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_component_size_distribution(components_df: pd.DataFrame, path: Path) -> None:
    if components_df.empty:
        save_empty_figure(path, "Ingredient Graph Component Size Distribution")
        return
    sizes = components_df["component_size"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(sizes) <= 40:
        ax.bar(np.arange(1, len(sizes) + 1), sizes)
        ax.set_xlabel("Component rank by size")
        ax.set_ylabel("Node count")
    else:
        ax.hist(sizes, bins=40)
        ax.set_yscale("log")
        ax.set_xlabel("Component size")
        ax.set_ylabel("Component count (log scale)")
    ax.set_title("Ingredient Graph Component Size Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pagerank_vs_popularity(nodes_df: pd.DataFrame, path: Path) -> None:
    if nodes_df.empty:
        save_empty_figure(path, "Weighted PageRank vs Ingredient Popularity")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.log1p(nodes_df["recipe_count"])
    y = nodes_df["pagerank_weighted"]
    ax.scatter(x, y, alpha=0.55, s=18)
    label_df = (
        nodes_df.assign(rank_gap=nodes_df["popularity_rank"] - nodes_df["pagerank_rank"])
        .sort_values(["rank_gap", "pagerank_weighted"], ascending=[False, False])
        .head(5)
    )
    for _, row in label_df.iterrows():
        ax.annotate(
            row["ingredient"],
            (np.log1p(row["recipe_count"]), row["pagerank_weighted"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title("Weighted PageRank vs Ingredient Popularity")
    ax.set_xlabel("log1p(recipe_count)")
    ax.set_ylabel("Weighted PageRank")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_top_bar(
    top_df: pd.DataFrame,
    metric: str,
    path: Path,
    title: str,
    xlabel: str,
) -> None:
    if top_df.empty:
        save_empty_figure(path, title)
        return
    display = top_df.head(20).sort_values(metric, ascending=True)
    fig_height = max(5.0, 0.35 * len(display) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(display["ingredient"], display[metric])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sensitivity(sensitivity_df: pd.DataFrame, path: Path) -> None:
    if sensitivity_df.empty:
        save_empty_figure(path, "Ingredient Graph Edge Threshold Sensitivity")
        return
    x = sensitivity_df["min_edge_recipe_count"]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, sensitivity_df["n_edges"], marker="o")
    ax1.set_xlabel("Minimum edge recipe count")
    ax1.set_ylabel("Retained edges")
    ax1.set_title("Ingredient Graph Edge Threshold Sensitivity")
    ax2 = ax1.twinx()
    ax2.plot(x, sensitivity_df["largest_component_share"], marker="s", linestyle="--")
    ax2.set_ylabel("Largest component share")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(
    nodes_df: pd.DataFrame,
    components_df: pd.DataFrame,
    top_pagerank_df: pd.DataFrame,
    top_weighted_degree_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_degree_distribution(
        nodes_df, figures_dir / "ingredient_graph_degree_distribution.png"
    )
    plot_weighted_degree_distribution(
        nodes_df, figures_dir / "ingredient_graph_weighted_degree_distribution.png"
    )
    plot_component_size_distribution(
        components_df, figures_dir / "ingredient_graph_component_size_distribution.png"
    )
    plot_pagerank_vs_popularity(
        nodes_df, figures_dir / "ingredient_graph_pagerank_vs_popularity.png"
    )
    plot_top_bar(
        top_pagerank_df,
        "pagerank_weighted",
        figures_dir / "ingredient_graph_top_pagerank.png",
        "Top Ingredients by Weighted PageRank",
        "Weighted PageRank",
    )
    plot_top_bar(
        top_weighted_degree_df,
        "weighted_degree",
        figures_dir / "ingredient_graph_top_weighted_degree.png",
        "Top Ingredients by Weighted Degree",
        "Weighted degree",
    )
    plot_sensitivity(sensitivity_df, figures_dir / "ingredient_graph_sensitivity_edges.png")


def build_sensitivity_outputs(
    node_counter: Counter,
    pair_counter: Counter,
    min_node_recipe_count: int,
    thresholds: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows = []
    top20_by_threshold: Dict[int, Set[str]] = {}

    for threshold in sorted(dict.fromkeys(thresholds)):
        logger.info("Running edge-threshold sensitivity for min_edge_recipe_count=%s", threshold)
        graph = build_graph(node_counter, pair_counter, min_node_recipe_count, threshold)
        stats = compute_graph_statistics(graph)
        nodes_df = compute_node_metrics(graph)
        top20 = top_k_set(nodes_df, "pagerank_weighted", min(20, len(nodes_df)))
        top20_by_threshold[int(threshold)] = top20

        threshold_rows.append(
            {
                "min_edge_recipe_count": int(threshold),
                "n_nodes": int(stats["n_nodes"]),
                "n_edges": int(stats["n_edges"]),
                "density": float(stats["density"]),
                "sparsity": float(stats["sparsity"]),
                "connected_components": int(stats["connected_components"]),
                "largest_component_share": float(stats["largest_component_share"]),
                "isolated_nodes": int(stats["isolated_node_count"]),
                "isolated_node_share": float(stats["isolated_node_share"]),
                "top20_pagerank_ingredients": "; ".join(
                    nodes_df.sort_values(
                        ["pagerank_weighted", "ingredient"], ascending=[False, True]
                    )
                    .head(20)["ingredient"]
                    .tolist()
                ),
            }
        )

    overlap_rows = []
    sorted_thresholds = sorted(top20_by_threshold)
    for i, threshold_a in enumerate(sorted_thresholds):
        for threshold_b in sorted_thresholds[i + 1 :]:
            set_a = top20_by_threshold[threshold_a]
            set_b = top20_by_threshold[threshold_b]
            union = set_a | set_b
            intersection = set_a & set_b
            overlap_rows.append(
                {
                    "threshold_a": int(threshold_a),
                    "threshold_b": int(threshold_b),
                    "overlap_count": int(len(intersection)),
                    "jaccard_overlap": float(len(intersection) / len(union)) if union else 0.0,
                    "shared_ingredients": "; ".join(sorted(intersection)),
                }
            )

    return pd.DataFrame(threshold_rows), pd.DataFrame(overlap_rows)


def inspect_week10_ranked_outputs(week10_dir: Path = Path("artifacts/week10")) -> List[str]:
    """Look for ranked recipe recommendation CSVs without requiring them."""
    if not week10_dir.exists():
        return []

    candidates = []
    for path in sorted(week10_dir.glob("*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        columns = {column.lower() for column in header.columns}
        has_recipe_id = any(column in columns for column in {"recipeid", "recipe_id", "recipe"})
        has_rank_signal = any(
            "rank" in column or "recommend" in column or column.endswith("score")
            for column in columns
        )
        if has_recipe_id and has_rank_signal:
            candidates.append(str(path))
    return candidates


def fmt_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isfinite(number) and number.is_integer() and abs(number) >= 1:
            return f"{int(number):,}"
        if abs(number) != 0 and abs(number) < 0.001:
            return f"{number:.6g}"
        return f"{number:.{digits}f}"
    text = str(value)
    return text.replace("|", "\\|")


def markdown_table(
    df: pd.DataFrame,
    columns: Sequence[str],
    headers: Sequence[str] = None,
    max_rows: int = 10,
) -> str:
    if df.empty:
        return "_No rows available._"
    subset = df.loc[:, list(columns)].head(max_rows).copy()
    headers = list(headers) if headers is not None else list(columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(fmt_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def summary_table(summary: Mapping[str, Any], counts_stats: Mapping[str, Any]) -> str:
    rows = pd.DataFrame(
        [
            ("Recipes processed", counts_stats["recipes_processed"]),
            (
                "Recipes with >= 1 valid ingredient",
                counts_stats["recipes_with_at_least_1_valid_ingredient"],
            ),
            (
                "Recipes with >= 2 valid ingredients",
                counts_stats["recipes_with_at_least_2_valid_ingredients"],
            ),
            (
                "Raw unique ingredients",
                counts_stats["raw_unique_ingredients_before_filtering"],
            ),
            ("Retained ingredient nodes", summary["retained_node_count"]),
            (
                "Raw unique ingredient pairs",
                counts_stats["raw_unique_ingredient_pairs_before_filtering"],
            ),
            ("Retained edges", summary["retained_edge_count"]),
            ("Density", summary["density"]),
            ("Sparsity", summary["sparsity"]),
            ("Connected components", summary["connected_components"]),
            ("Largest component share", summary["largest_component_share"]),
            ("Isolated nodes", summary["isolated_node_count"]),
            ("Isolated node share", summary["isolated_node_share"]),
            ("Average degree", summary["average_degree"]),
            ("Average weighted degree", summary["average_weighted_degree"]),
        ],
        columns=["metric", "value"],
    )
    return markdown_table(rows, ["metric", "value"], ["Metric", "Value"], max_rows=len(rows))


def comparison_report_tables(comparison_df: pd.DataFrame) -> Tuple[str, str]:
    corr_df = comparison_df[
        comparison_df["comparison_type"].eq("spearman_correlation")
    ].copy()
    overlap_df = comparison_df[comparison_df["comparison_type"].eq("top_k_overlap")].copy()
    corr_table = markdown_table(
        corr_df,
        ["baseline_metric", "graph_metric", "value"],
        ["Baseline metric", "Graph metric", "Spearman rho"],
        max_rows=10,
    )
    overlap_table = markdown_table(
        overlap_df,
        ["graph_metric", "k", "overlap_count", "overlap_share"],
        ["Graph metric", "K", "Overlap count", "Overlap share"],
        max_rows=20,
    )
    return corr_table, overlap_table


def rank_gap_examples(nodes_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pagerank_beyond_frequency = (
        nodes_df.assign(rank_gap=nodes_df["popularity_rank"] - nodes_df["pagerank_rank"])
        .query("rank_gap > 0")
        .sort_values(["rank_gap", "pagerank_weighted"], ascending=[False, False])
        .head(5)
    )
    popularity_less_distinctive = (
        nodes_df.assign(rank_gap=nodes_df["pagerank_rank"] - nodes_df["popularity_rank"])
        .query("rank_gap > 0")
        .sort_values(["rank_gap", "recipe_count"], ascending=[False, False])
        .head(5)
    )
    return pagerank_beyond_frequency, popularity_less_distinctive


def generate_report(
    report_path: Path,
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    counts_stats: Mapping[str, Any],
    nodes_df: pd.DataFrame,
    components_df: pd.DataFrame,
    validity_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    sensitivity_overlap_df: pd.DataFrame,
    top_popularity_df: pd.DataFrame,
    top_weighted_degree_df: pd.DataFrame,
    top_pagerank_df: pd.DataFrame,
    ranked_week10_files: Sequence[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    corr_table, overlap_table = comparison_report_tables(comparison_df)
    pagerank_beyond_frequency, popularity_less_distinctive = rank_gap_examples(nodes_df)

    model_artifact_note = (
        "Model-based recipe ranking artifacts were not available, so the comparison uses "
        "ingredient recipe frequency as the popularity baseline."
    )
    if ranked_week10_files:
        model_artifact_note = (
            "Potential Week 10 ranked recipe artifacts were detected, but the Week 12 "
            "primary comparison remains ingredient-grain graph centrality versus "
            "ingredient recipe frequency. Detected files: "
            + ", ".join(ranked_week10_files)
        )

    dense_note = (
        "The graph is sparse, which is expected for a thresholded ingredient network."
        if summary["density"] <= 0.10
        else "The graph is relatively dense; this suggests the edge threshold may be permissive."
    )
    component_note = (
        "Most retained ingredients participate in a shared co-occurrence space."
        if summary["largest_component_share"] >= 0.60
        else "The graph is fragmented, so the edge threshold may be excluding many bridges."
    )

    command = """python src/graphs/build_ingredient_graph.py \\
  --recipes data/processed/recipes_processed.csv \\
  --out artifacts/week12/ingredient_graph \\
  --figures reports/figures \\
  --report reports/week12_graph_analytics_report.md \\
  --min-node-recipe-count 50 \\
  --min-edge-recipe-count 20 \\
  --sensitivity-edge-thresholds 5 10 20 50"""

    report = f"""# Week 12 Graph Analytics and Centrality Report

## 1. Objective

This deliverable formalizes a domain graph for the Recipe Recommendation System and uses it for structural graph analysis. The Week 12 graph complements, but does not replace, the Week 10 recommender: the recommender asks what recipes a user may like, while this graph asks which ingredients are structurally central in the recipe corpus.

## 2. Formal Graph Definition

| Choice | Definition |
| --- | --- |
| Graph name | {GRAPH_DEFINITION["graph_name"]} |
| Graph type | {GRAPH_DEFINITION["graph_type"]} |
| Nodes | A node is a normalized ingredient. |
| Node grain | One node represents one normalized ingredient token, not a recipe, user, or category. |
| Edges | An undirected edge connects two ingredients if they co-occur in at least one recipe. |
| Edge weight | The edge weight is the number of distinct recipes in which the two ingredients co-occur. |
| Directionality | The graph is undirected because ingredient co-occurrence has no natural direction. |
| Filtering thresholds | Nodes require recipe_count >= {summary["min_node_recipe_count"]}; edges require cooccurrence_count >= {summary["min_edge_recipe_count"]}. |
| Recipe contribution rule | Each recipe contributes at most one count to an ingredient node and at most one count to a given ingredient pair. |

This graph is meaningful for the project because it captures ingredient relationship structure. It can reveal pantry staples, bridge ingredients, and ingredient neighborhoods that support ingredient exploration, substitution hypotheses, and future graph-aware recipe intelligence.

## 3. Graph Construction Pipeline

Input data came from `{summary["input_path"]}`. The pipeline parses `RecipeIngredientParts`, normalizes ingredient strings, deduplicates ingredients within each recipe, sorts them for deterministic pair generation, counts ingredient frequencies and pair co-occurrences, applies node and edge thresholds, then persists the graph as CSV, GraphML, JSON, figures, and this report.

Regeneration command:

```bash
{command}
```

## 4. Graph Summary and Validity Checks

{summary_table(summary, counts_stats)}

{dense_note} {component_note} The selected edge threshold keeps repeated co-occurrence relationships while filtering one-off pair noise.

Validity checks:

{markdown_table(validity_df, ["check", "value", "status", "note"], ["Check", "Value", "Status", "Note"], max_rows=len(validity_df))}

Figures:
- `reports/figures/ingredient_graph_degree_distribution.png`
- `reports/figures/ingredient_graph_weighted_degree_distribution.png`
- `reports/figures/ingredient_graph_component_size_distribution.png`

## 5. Connected Components

The graph contains {summary["connected_components"]:,} connected components. The largest component contains {summary["largest_component_size"]:,} ingredients, or {summary["largest_component_share"]:.4f} of retained nodes.

{markdown_table(components_df, ["component_id", "component_size", "component_share", "top_ingredients_by_recipe_count", "top_ingredients_by_pagerank"], ["Component", "Size", "Share", "Top by recipe count", "Top by PageRank"], max_rows=10)}

A large component means many ingredients participate in a shared culinary co-occurrence space. Smaller components may represent niche ingredients, rare cuisines, highly specialized recipe families, or noisy tokens. Isolated nodes are ingredients frequent enough to survive node filtering but without strong enough co-occurrence edges after the edge threshold.

## 6. Degree and Weighted Degree Analysis

Degree is the number of distinct ingredient neighbors. Weighted degree is the sum of co-occurrence counts across retained neighbors. A high degree ingredient appears with many different ingredients; a high weighted degree ingredient co-occurs frequently across the recipe corpus.

Top ingredients by weighted degree:

{markdown_table(top_weighted_degree_df, ["rank", "ingredient", "recipe_count", "degree", "weighted_degree", "pagerank_weighted"], ["Rank", "Ingredient", "Recipe count", "Degree", "Weighted degree", "PageRank"], max_rows=10)}

Top ingredients by popularity baseline:

{markdown_table(top_popularity_df, ["rank", "ingredient", "recipe_count", "degree", "weighted_degree", "pagerank_weighted"], ["Rank", "Ingredient", "Recipe count", "Degree", "Weighted degree", "PageRank"], max_rows=10)}

Figure: `reports/figures/ingredient_graph_top_weighted_degree.png`

## 7. PageRank / Centrality Analysis

In this undirected weighted graph, PageRank models a random walk over the ingredient network. Ingredients connected to other important ingredients receive higher scores, and weighted PageRank uses co-occurrence strength as the transition weight.

Top ingredients by weighted PageRank:

{markdown_table(top_pagerank_df, ["rank", "ingredient", "recipe_count", "degree", "weighted_degree", "pagerank_weighted"], ["Rank", "Ingredient", "Recipe count", "Degree", "Weighted degree", "PageRank"], max_rows=15)}

High PageRank should be interpreted as structural centrality, not as "best ingredient." Generic staples may rank highly because they connect many parts of the corpus. More specific bridge ingredients are useful because they can connect ingredient neighborhoods that would otherwise be less directly linked.

Figures:
- `reports/figures/ingredient_graph_top_pagerank.png`
- `reports/figures/ingredient_graph_pagerank_vs_popularity.png`

## 8. Comparison: Graph Ranking vs Popularity Baseline

The popularity baseline is `recipe_count`: the number of distinct recipes containing an ingredient. The graph rankings are weighted PageRank and weighted degree.

{model_artifact_note}

Spearman correlations:

{corr_table}

Top-K overlap with popularity:

{overlap_table}

Examples where PageRank ranks ingredients higher than raw frequency:

{markdown_table(pagerank_beyond_frequency, ["ingredient", "recipe_count", "popularity_rank", "pagerank_rank", "pagerank_weighted", "rank_gap"], ["Ingredient", "Recipe count", "Popularity rank", "PageRank rank", "PageRank", "Rank gain"], max_rows=5)}

Examples where popularity is high but PageRank is less distinctive:

{markdown_table(popularity_less_distinctive, ["ingredient", "recipe_count", "popularity_rank", "pagerank_rank", "pagerank_weighted", "rank_gap"], ["Ingredient", "Recipe count", "Popularity rank", "PageRank rank", "PageRank", "Rank loss"], max_rows=5)}

If the correlations are high, graph centrality is partially driven by ingredient popularity. Differences between the rankings show where PageRank captures network position beyond simple frequency.

## 9. Sensitivity Analysis

Edge threshold matters because it controls whether weak one-off co-occurrences are retained. Lower thresholds keep more edges and usually create a denser, more connected graph. Higher thresholds emphasize stable co-occurrences but can isolate nodes and fragment components.

{markdown_table(sensitivity_df, ["min_edge_recipe_count", "n_nodes", "n_edges", "density", "sparsity", "connected_components", "largest_component_share", "isolated_nodes"], ["Edge threshold", "Nodes", "Edges", "Density", "Sparsity", "Components", "Largest share", "Isolates"], max_rows=len(sensitivity_df))}

Top-20 PageRank overlap between thresholds:

{markdown_table(sensitivity_overlap_df, ["threshold_a", "threshold_b", "overlap_count", "jaccard_overlap"], ["Threshold A", "Threshold B", "Overlap count", "Jaccard overlap"], max_rows=len(sensitivity_overlap_df))}

Stable top central ingredients indicate robust graph structure. Large overlap changes indicate sensitivity to the edge-definition threshold. Figure: `reports/figures/ingredient_graph_sensitivity_edges.png`

## 10. Interpretation Note: What the Graph Means and Does Not Mean

Graph structure means ingredient co-occurrence patterns in the Food.com dataset. It can reflect culinary compatibility inside the dataset, pantry-staple centrality, bridge ingredients between culinary styles, and structural ingredient importance.

Graph structure does not mean user preference, nutritional quality, causal compatibility, substitution equivalence, personalized recommendation, or universal cultural importance. It also does not prove that two ingredients taste good together outside the dataset context. Food.com may overrepresent American and Western comfort food, so central ingredients reflect the dataset's cuisine distribution and contributor behavior.

## 11. Relationship to Previous Deliverables

Week 5 produced content embeddings from ingredient, category, keyword, and numeric features. Week 7 clustered recipes using semantic and numeric recipe representations. Week 10 ranked recipes using user behavior and content similarity. Week 12 adds a structural ingredient-network perspective that can support future cluster-aware and graph-aware recommendation.

## 12. Limitations and Future Work

Ingredient normalization may not merge all synonyms. Raw co-occurrence favors common ingredients. The edge threshold affects graph density and component structure. Quantities, preparation instructions, and cooking order are ignored. The graph is undirected and does not capture preparation sequence. High centrality may be dominated by staples.

Future work could use PMI, PPMI, or Jaccard-weighted graphs; recipe-recipe graphs; user-recipe bipartite graphs; synonym dictionaries; or graph-aware recommendation features.

## 13. Reproducibility

Exact command:

```bash
{command}
```

Generated artifacts under `{args.out}`:

{chr(10).join(f"- `{args.out / artifact}`" for artifact in ARTIFACT_FILES)}

Generated figures:

{chr(10).join(f"- `{args.figures / figure}`" for figure in FIGURE_FILES)}
"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_positive_thresholds(args)
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    recipes_df = read_recipes(args.recipes)
    if INGREDIENT_COL not in recipes_df.columns:
        raise ValueError(f"Required column {INGREDIENT_COL!r} is missing.")

    node_counter, pair_counter, counts_stats = collect_ingredient_counts(
        recipes_df[INGREDIENT_COL]
    )

    logger.info("Building main ingredient graph.")
    graph = build_graph(
        node_counter,
        pair_counter,
        args.min_node_recipe_count,
        args.min_edge_recipe_count,
    )
    graph_stats = compute_graph_statistics(graph)
    nodes_df = compute_node_metrics(
        graph,
        compute_approx_betweenness=args.compute_approx_betweenness,
        approx_betweenness_samples=args.approx_betweenness_samples,
        seed=args.seed,
    )
    nodes_df = add_rank_columns(nodes_df)
    edges_df = edges_to_dataframe(graph)
    components_df = build_connected_components_df(graph, nodes_df)

    top_popularity_df = top_by_metric(nodes_df, "recipe_count")
    top_weighted_degree_df = top_by_metric(nodes_df, "weighted_degree")
    top_pagerank_df = top_by_metric(nodes_df, "pagerank_weighted")
    top_log_pagerank_df = top_by_metric(nodes_df, "pagerank_log_weighted")
    comparison_df = build_comparison_df(nodes_df)

    summary = {
        "graph_definition": GRAPH_DEFINITION,
        "input_path": str(args.recipes),
        "output_path": str(args.out),
        "min_node_recipe_count": int(args.min_node_recipe_count),
        "min_edge_recipe_count": int(args.min_edge_recipe_count),
        "recipes_processed": int(counts_stats["recipes_processed"]),
        "raw_node_count": int(counts_stats["raw_unique_ingredients_before_filtering"]),
        "retained_node_count": int(graph.number_of_nodes()),
        "raw_edge_count": int(counts_stats["raw_unique_ingredient_pairs_before_filtering"]),
        "retained_edge_count": int(graph.number_of_edges()),
        "density": float(graph_stats["density"]),
        "sparsity": float(graph_stats["sparsity"]),
        "connected_components": int(graph_stats["connected_components"]),
        "largest_component_size": int(graph_stats["largest_component_size"]),
        "largest_component_share": float(graph_stats["largest_component_share"]),
        "isolated_node_count": int(graph_stats["isolated_node_count"]),
        "isolated_node_share": float(graph_stats["isolated_node_share"]),
        "average_degree": float(graph_stats["average_degree"]),
        "average_weighted_degree": float(graph_stats["average_weighted_degree"]),
        "top_component_sizes": graph_stats["top_component_sizes"],
    }

    validity_df = build_validity_checks(summary, counts_stats, nodes_df)
    sensitivity_df, sensitivity_overlap_df = build_sensitivity_outputs(
        node_counter,
        pair_counter,
        args.min_node_recipe_count,
        args.sensitivity_edge_thresholds,
    )
    ranked_week10_files = inspect_week10_ranked_outputs()

    logger.info("Writing graph artifacts to %s", args.out)
    nodes_df.to_csv(args.out / "ingredient_nodes.csv", index=False)
    edges_df.to_csv(args.out / "ingredient_edges.csv", index=False)
    nx.write_graphml(graph, args.out / "ingredient_graph.graphml")
    save_json(args.out / "ingredient_graph_summary.json", summary)
    components_df.to_csv(args.out / "connected_components.csv", index=False)
    validity_df.to_csv(args.out / "graph_validity_checks.csv", index=False)
    comparison_df.to_csv(args.out / "comparison_graph_vs_popularity.csv", index=False)
    top_popularity_df.to_csv(args.out / "top_ingredients_by_popularity.csv", index=False)
    top_weighted_degree_df.to_csv(
        args.out / "top_ingredients_by_weighted_degree.csv", index=False
    )
    top_pagerank_df.to_csv(args.out / "top_ingredients_by_pagerank.csv", index=False)
    top_log_pagerank_df.to_csv(args.out / "top_ingredients_by_log_pagerank.csv", index=False)
    sensitivity_df.to_csv(args.out / "sensitivity_edge_thresholds.csv", index=False)
    sensitivity_overlap_df.to_csv(
        args.out / "sensitivity_top20_pagerank_overlap.csv", index=False
    )

    pipeline_config = {
        "script": "src/graphs/build_ingredient_graph.py",
        "recipes": str(args.recipes),
        "out": str(args.out),
        "figures": str(args.figures),
        "report": str(args.report),
        "min_node_recipe_count": int(args.min_node_recipe_count),
        "min_edge_recipe_count": int(args.min_edge_recipe_count),
        "sensitivity_edge_thresholds": [
            int(threshold) for threshold in args.sensitivity_edge_thresholds
        ],
        "compute_approx_betweenness": bool(args.compute_approx_betweenness),
        "approx_betweenness_samples": int(args.approx_betweenness_samples),
        "seed": int(args.seed),
        "graph_definition": GRAPH_DEFINITION,
        "week10_ranked_recipe_artifacts_detected": list(ranked_week10_files),
    }
    save_json(args.out / "graph_pipeline_config.json", pipeline_config)

    logger.info("Writing figures to %s", args.figures)
    make_figures(
        nodes_df,
        components_df,
        top_pagerank_df,
        top_weighted_degree_df,
        sensitivity_df,
        args.figures,
    )

    logger.info("Writing Week 12 report to %s", args.report)
    generate_report(
        args.report,
        args,
        summary,
        counts_stats,
        nodes_df,
        components_df,
        validity_df,
        comparison_df,
        sensitivity_df,
        sensitivity_overlap_df,
        top_popularity_df,
        top_weighted_degree_df,
        top_pagerank_df,
        ranked_week10_files,
    )
    logger.info("Week 12 graph deliverable complete.")


if __name__ == "__main__":
    main()
