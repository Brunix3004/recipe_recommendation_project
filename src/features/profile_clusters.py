#!/usr/bin/env python3
"""
Generate interpretable cluster profiles for a selected Week 7 clustering run.

This script does not interpret latent dimensions directly. Instead, it explains
cluster behavior through original human-readable recipe metadata (nutrition,
time, categories, keywords, and ingredients). That mapping is essential because
latent axes from SVD/PCA embeddings are not intrinsically semantic labels.

Cluster profiling is a key bridge from latent recipe structure to practical
recommendation systems: clusters represent embedding-space neighborhoods, and
metadata profiling makes those neighborhoods understandable and actionable.

Outlier points were hidden in boxplot visualizations to improve readability
due to the extremely heavy-tailed nature of recipe nutritional and preparation-time
distributions.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOP_N_TOKENS = 5
TOP_N_CATEGORIES = 3
TOP_N_REPRESENTATIVE = 10

DEFAULT_RECIPES_PATH = Path("data/processed/recipes_processed.csv")
DEFAULT_ASSIGNMENTS_PATH = Path("artifacts/week7/clustering_models/cluster_assignments_k5.csv")
DEFAULT_EVALUATION_SUMMARY_PATH = Path(
    "artifacts/week7/clustering_models/clustering_evaluation_summary.csv"
)
DEFAULT_CLUSTERING_MATRIX_PATH = Path("artifacts/week7/clustering_matrix/X_recipe_clustering.npy")
DEFAULT_CLUSTERING_RECIPE_IDS_PATH = Path(
    "artifacts/week7/clustering_matrix/clustering_recipe_ids.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/week7/clustering_reports")
DEFAULT_FIGURES_DIR = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate interpretable profiles for a selected clustering assignment "
            "by combining cluster labels with original recipe metadata."
        )
    )
    parser.add_argument(
        "--recipes",
        type=Path,
        default=DEFAULT_RECIPES_PATH,
        help=f"Path to recipes_processed.csv (default: {DEFAULT_RECIPES_PATH}).",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_PATH,
        help=f"Path to cluster_assignments_k*.csv (default: {DEFAULT_ASSIGNMENTS_PATH}).",
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=DEFAULT_EVALUATION_SUMMARY_PATH,
        help=(
            "Path to clustering_evaluation_summary.csv "
            f"(default: {DEFAULT_EVALUATION_SUMMARY_PATH})."
        ),
    )
    parser.add_argument(
        "--clustering-matrix",
        type=Path,
        default=DEFAULT_CLUSTERING_MATRIX_PATH,
        help=f"Path to X_recipe_clustering.npy (default: {DEFAULT_CLUSTERING_MATRIX_PATH}).",
    )
    parser.add_argument(
        "--clustering-recipe-ids",
        type=Path,
        default=DEFAULT_CLUSTERING_RECIPE_IDS_PATH,
        help=(
            "Path to clustering_recipe_ids.csv used for matrix row alignment "
            f"(default: {DEFAULT_CLUSTERING_RECIPE_IDS_PATH})."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for profiling CSVs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Output directory for figures (default: {DEFAULT_FIGURES_DIR}).",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def infer_selected_k_from_filename(assignments_path: Path) -> int:
    match = re.search(r"_k(\d+)\.csv$", assignments_path.name)
    if match is None:
        raise ValueError(
            "Unable to infer selected k from assignment filename. "
            f"Expected pattern like cluster_assignments_k10.csv, got: {assignments_path.name}"
        )
    return int(match.group(1))


def validate_required_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def validate_matrix_2d_finite(matrix: np.ndarray, name: str) -> None:
    if not isinstance(matrix, np.ndarray):
        raise ValueError(f"{name} must be a numpy array.")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    recipes_path = resolve_existing_path(args.recipes, "Recipes dataset")
    assignments_path = resolve_existing_path(args.assignments, "Cluster assignments CSV")
    evaluation_summary_path = resolve_existing_path(
        args.evaluation_summary, "Clustering evaluation summary CSV"
    )
    matrix_path = resolve_existing_path(args.clustering_matrix, "Clustering matrix")
    clustering_recipe_ids_path = resolve_existing_path(
        args.clustering_recipe_ids, "Clustering recipe IDs CSV"
    )

    return {
        "recipes_path": recipes_path,
        "assignments_path": assignments_path,
        "evaluation_summary_path": evaluation_summary_path,
        "matrix_path": matrix_path,
        "clustering_recipe_ids_path": clustering_recipe_ids_path,
        "recipes": pd.read_csv(recipes_path, low_memory=False),
        "assignments": pd.read_csv(assignments_path),
        "evaluation_summary": pd.read_csv(evaluation_summary_path),
        "X_clustering": np.load(matrix_path),
        "clustering_recipe_ids": pd.read_csv(clustering_recipe_ids_path),
    }


def resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of the candidate columns for {label} exists: {candidates}")


def resolve_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def validate_inputs(inputs: dict[str, Any], selected_k: int) -> None:
    recipes: pd.DataFrame = inputs["recipes"]
    assignments: pd.DataFrame = inputs["assignments"]
    evaluation_summary: pd.DataFrame = inputs["evaluation_summary"]
    X_clustering: np.ndarray = inputs["X_clustering"]
    clustering_recipe_ids: pd.DataFrame = inputs["clustering_recipe_ids"]

    validate_required_columns(recipes, ["RecipeId"], "recipes_processed.csv")
    validate_required_columns(assignments, ["RecipeId", "cluster"], "cluster assignments CSV")
    validate_required_columns(
        evaluation_summary,
        ["k", "inertia", "silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"],
        "clustering_evaluation_summary.csv",
    )
    validate_required_columns(
        clustering_recipe_ids,
        ["row_index", "RecipeId"],
        "clustering_recipe_ids.csv",
    )

    validate_matrix_2d_finite(X_clustering, "X_recipe_clustering")

    if len(clustering_recipe_ids) != X_clustering.shape[0]:
        raise ValueError(
            "clustering_recipe_ids.csv row count must match clustering matrix rows: "
            f"{len(clustering_recipe_ids)} != {X_clustering.shape[0]}"
        )
    expected_index = np.arange(X_clustering.shape[0], dtype=np.int64)
    if not np.array_equal(clustering_recipe_ids["row_index"].to_numpy(), expected_index):
        raise ValueError("clustering_recipe_ids row_index must be contiguous from 0 to n_rows-1.")

    if assignments["RecipeId"].duplicated().any():
        dup_count = int(assignments["RecipeId"].duplicated().sum())
        raise ValueError(f"cluster assignments contain duplicate RecipeId values: {dup_count}")
    if recipes["RecipeId"].duplicated().any():
        dup_count = int(recipes["RecipeId"].duplicated().sum())
        raise ValueError(f"recipes_processed.csv contains duplicate RecipeId values: {dup_count}")

    eval_row = evaluation_summary.loc[evaluation_summary["k"].eq(selected_k)]
    if eval_row.empty:
        raise ValueError(
            f"Selected k={selected_k} from assignment filename is missing in evaluation summary."
        )

    cluster_unique = sorted(assignments["cluster"].dropna().astype(int).unique().tolist())
    if len(cluster_unique) != selected_k:
        raise ValueError(
            f"Assignment file implies k={selected_k}, but found {len(cluster_unique)} unique clusters."
        )


def normalize_token(token: Any) -> str:
    text = str(token).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s/-]+", " ", text)
    text = re.sub(r"[\s-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return "" if text in {"", "nan", "none", "null"} else text


def parse_token_list(value: Any) -> list[str]:
    # Robust parsing for malformed list-like fields (e.g., stringified lists,
    # comma-separated strings, or inconsistent row formats).
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    elif value is None or pd.isna(value):
        raw_items = []
    else:
        text = str(value).strip()
        if not text:
            raw_items = []
        else:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    raw_items = list(parsed)
                elif parsed is None:
                    raw_items = []
                else:
                    raw_items = [parsed]
            except (SyntaxError, ValueError):
                cleaned = re.sub(r"^[\[\(\{]+|[\]\)\}]+$", "", text)
                raw_items = re.split(r"[|,;]", cleaned)

    tokens = [normalize_token(item) for item in raw_items]
    return [token for token in tokens if token]


def top_from_counter(counter: Counter[str], n: int) -> list[str]:
    return [token for token, _ in counter.most_common(n)]


def join_top(tokens: list[str]) -> str:
    return " | ".join(tokens) if tokens else ""


def merge_metadata_assignments(
    recipes: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    merged = recipes.merge(
        assignments[["RecipeId", "cluster"]],
        on="RecipeId",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(assignments):
        raise ValueError(
            "Row consistency check failed: merged rows do not match assignment rows. "
            f"merged={len(merged)}, assignments={len(assignments)}"
        )
    if merged["RecipeId"].duplicated().any():
        dup_count = int(merged["RecipeId"].duplicated().sum())
        raise ValueError(f"Duplicate RecipeId values detected after merge: {dup_count}")
    return merged


def build_row_cluster_mapping(
    clustering_recipe_ids: pd.DataFrame,
    assignments: pd.DataFrame,
    n_rows: int,
) -> pd.DataFrame:
    mapping = clustering_recipe_ids.merge(
        assignments[["RecipeId", "cluster"]],
        on="RecipeId",
        how="left",
        validate="one_to_one",
    )
    if mapping["cluster"].isna().any():
        missing = int(mapping["cluster"].isna().sum())
        raise ValueError(
            f"Cluster labels do not fully align with clustering recipe IDs: {missing} rows missing."
        )
    if len(mapping) != n_rows:
        raise ValueError(
            "Row consistency check failed for clustering matrix mapping: "
            f"{len(mapping)} != {n_rows}"
        )
    return mapping


def resolve_profile_columns(merged: pd.DataFrame) -> dict[str, str | None]:
    return {
        "name": resolve_optional_column(merged, ["Name", "RecipeName", "title"]),
        "category": resolve_optional_column(merged, ["RecipeCategory_clean", "RecipeCategory"]),
        "keywords": resolve_optional_column(merged, ["Keywords", "RecipeKeywords"]),
        "ingredients": resolve_optional_column(
            merged, ["RecipeIngredientParts", "Ingredients", "RecipeIngredients"]
        ),
        "calories": resolve_column(merged, ["Calories"], "Calories"),
        "protein": resolve_column(merged, ["ProteinContent"], "ProteinContent"),
        "fat": resolve_column(merged, ["FatContent"], "FatContent"),
        "carbs": resolve_column(merged, ["CarbohydrateContent"], "CarbohydrateContent"),
        "total_time": resolve_column(
            merged, ["TotalTime_Minutes", "TotalTime_Minutes_imputed"], "TotalTime_Minutes"
        ),
        "cook_time": resolve_column(
            merged, ["CookTime_Minutes", "CookTime_Minutes_imputed"], "CookTime_Minutes"
        ),
        "prep_time": resolve_column(
            merged, ["PrepTime_Minutes", "PrepTime_Minutes_imputed"], "PrepTime_Minutes"
        ),
    }


def add_parsed_token_columns(
    merged: pd.DataFrame,
    category_col: str | None,
    keywords_col: str | None,
    ingredients_col: str | None,
) -> pd.DataFrame:
    prof = merged.copy()
    if keywords_col is None:
        prof["_keyword_tokens"] = [[] for _ in range(len(prof))]
    else:
        prof["_keyword_tokens"] = prof[keywords_col].apply(parse_token_list)

    if ingredients_col is None:
        prof["_ingredient_tokens"] = [[] for _ in range(len(prof))]
    else:
        prof["_ingredient_tokens"] = prof[ingredients_col].apply(parse_token_list)

    if category_col is None:
        prof["_category_token"] = ""
    else:
        prof["_category_token"] = prof[category_col].fillna("").map(normalize_token)

    prof["_semantic_tokens"] = prof.apply(
        lambda row: (
            row["_keyword_tokens"]
            + row["_ingredient_tokens"]
            + ([row["_category_token"]] if row["_category_token"] else [])
        ),
        axis=1,
    )
    return prof


def compute_cluster_centroids(X: np.ndarray, row_mapping: pd.DataFrame) -> dict[int, np.ndarray]:
    tick_labels = row_mapping["cluster"].to_numpy(dtype=np.int64)
    centroids: dict[int, np.ndarray] = {}
    for cluster in sorted(np.unique(tick_labels).tolist()):
        row_indices = row_mapping.loc[
            row_mapping["cluster"].astype(np.int64).eq(cluster), "row_index"
        ].to_numpy(dtype=np.int64)
        if len(row_indices) == 0:
            raise ValueError(f"Empty cluster detected for centroid computation: {cluster}")
        centroids[int(cluster)] = X[row_indices].mean(axis=0)
    return centroids


def build_representative_recipes(
    X: np.ndarray,
    row_mapping: pd.DataFrame,
    profiles: pd.DataFrame,
    centroids: dict[int, np.ndarray],
    name_col: str | None,
) -> pd.DataFrame:
    representatives: list[pd.DataFrame] = []

    for cluster, centroid in centroids.items():
        row_indices = row_mapping.loc[
            row_mapping["cluster"].astype(np.int64).eq(cluster), "row_index"
        ].to_numpy(dtype=np.int64)
        X_cluster = X[row_indices]
        distances = np.linalg.norm(X_cluster - centroid, axis=1)
        cluster_rows = pd.DataFrame(
            {
                "row_index": row_indices,
                "cluster": int(cluster),
                "distance_to_centroid": distances.astype(np.float64),
            }
        )
        cluster_rows = cluster_rows.merge(
            row_mapping[["row_index", "RecipeId"]],
            on="row_index",
            how="left",
            validate="one_to_one",
        )
        cluster_rows = cluster_rows.sort_values(
            ["distance_to_centroid", "RecipeId", "row_index"],
            ascending=[True, True, True],
        ).head(TOP_N_REPRESENTATIVE)
        cluster_rows["rank_within_cluster"] = np.arange(1, len(cluster_rows) + 1, dtype=np.int64)
        representatives.append(cluster_rows)

    reps = pd.concat(representatives, ignore_index=True).sort_values(
        ["cluster", "rank_within_cluster"]
    )

    if name_col is None:
        reps["recipe_name"] = reps["RecipeId"].astype(str)
    else:
        recipe_names = profiles[["RecipeId", name_col]].drop_duplicates(subset=["RecipeId"])
        reps = reps.merge(recipe_names, on="RecipeId", how="left", validate="many_to_one")
        reps["recipe_name"] = reps[name_col].fillna(reps["RecipeId"].astype(str))

    return reps[
        [
            "cluster",
            "rank_within_cluster",
            "row_index",
            "RecipeId",
            "recipe_name",
            "distance_to_centroid",
        ]
    ]


def classify_level(value: float, reference: float) -> str:
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return "moderate"
    ratio = value / reference
    if ratio >= 1.15:
        return "high"
    if ratio <= 0.85:
        return "low"
    return "moderate"


def classify_time_level(value: float, reference: float) -> str:
    if not np.isfinite(value) or not np.isfinite(reference) or reference <= 0:
        return "moderate"
    ratio = value / reference
    if ratio >= 1.15:
        return "long"
    if ratio <= 0.85:
        return "short"
    return "moderate"


def build_interpretation_summary(
    cluster: int,
    dominant_category: str,
    top_keywords: list[str],
    top_ingredients: list[str],
    avg_protein: float,
    avg_calories: float,
    avg_total_time: float,
    protein_ref: float,
    calories_ref: float,
    total_time_ref: float,
) -> str:
    protein_level = classify_level(avg_protein, protein_ref)
    calorie_level = classify_level(avg_calories, calories_ref)
    time_level = classify_time_level(avg_total_time, total_time_ref)

    semantic_bits = top_ingredients[:2] + top_keywords[:2]
    semantic_phrase = ", ".join(semantic_bits) if semantic_bits else "general recipe patterns"
    category_phrase = dominant_category if dominant_category else "mixed-category"

    return (
        f"Cluster {cluster} contains primarily {category_phrase} recipes with "
        f"{protein_level}-protein and {calorie_level}-calorie structure, "
        f"{time_level} preparation time, and semantic signals around {semantic_phrase}."
    )


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def profile_clusters(
    profiles: pd.DataFrame,
    representatives: pd.DataFrame,
    cols: dict[str, str | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = len(profiles)
    groupby_cluster = profiles.groupby("cluster", sort=True)

    calories_all = numeric_series(profiles, cols["calories"])  # type: ignore[arg-type]
    protein_all = numeric_series(profiles, cols["protein"])  # type: ignore[arg-type]
    total_time_all = numeric_series(profiles, cols["total_time"])  # type: ignore[arg-type]

    reference_calories = float(calories_all.median(skipna=True))
    reference_protein = float(protein_all.median(skipna=True))
    reference_total_time = float(total_time_all.median(skipna=True))

    summary_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []

    for cluster, group in groupby_cluster:
        recipe_count = int(len(group))
        dataset_percentage = float(recipe_count / total_rows * 100) if total_rows else 0.0

        category_counter: Counter[str] = Counter()
        if cols["category"] is not None:
            category_counter.update(
                normalize_token(v) for v in group[cols["category"]].dropna().astype(str).tolist()
            )
        category_counter.pop("", None)

        keyword_counter: Counter[str] = Counter()
        ingredient_counter: Counter[str] = Counter()
        semantic_counter: Counter[str] = Counter()
        for tokens in group["_keyword_tokens"]:
            keyword_counter.update(tokens)
        for tokens in group["_ingredient_tokens"]:
            ingredient_counter.update(tokens)
        for tokens in group["_semantic_tokens"]:
            semantic_counter.update(tokens)

        top_categories = top_from_counter(category_counter, TOP_N_CATEGORIES)
        top_keywords = top_from_counter(keyword_counter, TOP_N_TOKENS)
        top_ingredients = top_from_counter(ingredient_counter, TOP_N_TOKENS)
        top_semantic = top_from_counter(semantic_counter, TOP_N_TOKENS)
        dominant_category = top_categories[0] if top_categories else ""

        calories = numeric_series(group, cols["calories"])  # type: ignore[arg-type]
        protein = numeric_series(group, cols["protein"])  # type: ignore[arg-type]
        fat = numeric_series(group, cols["fat"])  # type: ignore[arg-type]
        carbs = numeric_series(group, cols["carbs"])  # type: ignore[arg-type]
        total_time = numeric_series(group, cols["total_time"])  # type: ignore[arg-type]
        cook_time = numeric_series(group, cols["cook_time"])  # type: ignore[arg-type]
        prep_time = numeric_series(group, cols["prep_time"])  # type: ignore[arg-type]

        rep_names = representatives.loc[
            representatives["cluster"].eq(cluster), "recipe_name"
        ].head(3)
        representative_name_text = " | ".join(rep_names.astype(str).tolist())

        avg_calories = float(calories.mean(skipna=True))
        avg_protein = float(protein.mean(skipna=True))
        avg_total_time = float(total_time.mean(skipna=True))

        interpretation = build_interpretation_summary(
            cluster=int(cluster),
            dominant_category=dominant_category,
            top_keywords=top_keywords,
            top_ingredients=top_ingredients,
            avg_protein=avg_protein,
            avg_calories=avg_calories,
            avg_total_time=avg_total_time,
            protein_ref=reference_protein,
            calories_ref=reference_calories,
            total_time_ref=reference_total_time,
        )

        summary_rows.append(
            {
                "cluster": int(cluster),
                "recipe_count": recipe_count,
                "dataset_percentage": dataset_percentage,
                "dominant_category": dominant_category,
                "dominant_keywords": join_top(top_keywords[:3]),
                "dominant_ingredients": join_top(top_ingredients[:3]),
                "avg_calories": avg_calories,
                "avg_protein": avg_protein,
                "avg_total_time": avg_total_time,
                "interpretation_summary": interpretation,
            }
        )

        detailed_rows.append(
            {
                "cluster": int(cluster),
                "recipe_count": recipe_count,
                "dataset_percentage": dataset_percentage,
                "mean_calories": avg_calories,
                "median_calories": float(calories.median(skipna=True)),
                "mean_protein": avg_protein,
                "median_protein": float(protein.median(skipna=True)),
                "mean_fat": float(fat.mean(skipna=True)),
                "median_fat": float(fat.median(skipna=True)),
                "mean_carbohydrate": float(carbs.mean(skipna=True)),
                "median_carbohydrate": float(carbs.median(skipna=True)),
                "mean_total_time": avg_total_time,
                "median_total_time": float(total_time.median(skipna=True)),
                "mean_cook_time": float(cook_time.mean(skipna=True)),
                "median_cook_time": float(cook_time.median(skipna=True)),
                "mean_prep_time": float(prep_time.mean(skipna=True)),
                "median_prep_time": float(prep_time.median(skipna=True)),
                "top_categories": join_top(top_categories),
                "top_keywords": join_top(top_keywords),
                "top_ingredients": join_top(top_ingredients),
                "most_frequent_semantic_tokens": join_top(top_semantic),
                "most_representative_recipe_names": representative_name_text,
                "interpretation_summary": interpretation,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster").reset_index(drop=True)
    detailed_df = pd.DataFrame(detailed_rows).sort_values("cluster").reset_index(drop=True)
    return summary_df, detailed_df


def plot_cluster_size_distribution(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["cluster"].astype(str), summary_df["recipe_count"])
    plt.xlabel("Cluster")
    plt.ylabel("Recipe count")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric_by_cluster(
    profiles: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    plot_df = profiles[["cluster", metric_col]].copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])

    plt.figure(figsize=(8, 5))
    if plot_df.empty:
        plt.text(0.5, 0.5, "No valid data available", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        groups = [
            plot_df.loc[plot_df["cluster"].eq(cluster), metric_col].to_numpy()
            for cluster in sorted(plot_df["cluster"].unique().tolist())
        ]
        labels = [str(cluster) for cluster in sorted(plot_df["cluster"].unique().tolist())]
        # Outlier-robust display only: raw data are unchanged, but extreme fliers
        # are hidden to keep interquartile structure readable across clusters.
        # plt.boxplot(groups, labels=labels, showfliers=False)
        plt.boxplot(groups, showfliers=False)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.xlabel("Cluster")
        plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_outputs(
    out_dir: Path,
    figures_dir: Path,
    selected_k: int,
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    representatives: pd.DataFrame,
    profiles: pd.DataFrame,
    cols: dict[str, str | None],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "cluster_profile_summary": out_dir / "cluster_profile_summary.csv",
        "cluster_numeric_statistics": out_dir / "cluster_numeric_statistics.csv",
        "representative_recipes": out_dir / f"representative_recipes_k{selected_k}.csv",
        "cluster_size_distribution_plot": figures_dir / "cluster_size_distribution.png",
        "calories_by_cluster_plot": figures_dir / "calories_by_cluster.png",
        "protein_by_cluster_plot": figures_dir / "protein_by_cluster.png",
        "total_time_by_cluster_plot": figures_dir / "total_time_by_cluster.png",
    }

    summary_df.to_csv(paths["cluster_profile_summary"], index=False)
    detailed_df.to_csv(paths["cluster_numeric_statistics"], index=False)
    representatives.to_csv(paths["representative_recipes"], index=False)

    plot_cluster_size_distribution(summary_df, paths["cluster_size_distribution_plot"])
    plot_metric_by_cluster(
        profiles=profiles,
        metric_col=cols["calories"],  # type: ignore[arg-type]
        title="Calories by Cluster",
        y_label="Calories",
        out_path=paths["calories_by_cluster_plot"],
    )
    plot_metric_by_cluster(
        profiles=profiles,
        metric_col=cols["protein"],  # type: ignore[arg-type]
        title="Protein by Cluster",
        y_label="Protein",
        out_path=paths["protein_by_cluster_plot"],
    )
    plot_metric_by_cluster(
        profiles=profiles,
        metric_col=cols["total_time"],  # type: ignore[arg-type]
        title="Total Time by Cluster",
        y_label="Total Time (minutes)",
        out_path=paths["total_time_by_cluster_plot"],
    )
    return paths


def print_final_summary(
    selected_k: int,
    summary_df: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    largest_row = summary_df.loc[summary_df["recipe_count"].idxmax()]
    smallest_row = summary_df.loc[summary_df["recipe_count"].idxmin()]

    print("Cluster profiling complete.")
    print(f"Selected k: {selected_k}")
    print(f"Number of clusters: {len(summary_df)}")
    print(f"Largest cluster: {int(largest_row['cluster'])} ({int(largest_row['recipe_count'])} recipes)")
    print(
        f"Smallest cluster: {int(smallest_row['cluster'])} "
        f"({int(smallest_row['recipe_count'])} recipes)"
    )
    print("Output paths:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def main() -> None:
    args = parse_args()
    selected_k = infer_selected_k_from_filename(args.assignments)
    inputs = load_inputs(args)
    validate_inputs(inputs, selected_k)

    recipes: pd.DataFrame = inputs["recipes"]
    assignments: pd.DataFrame = inputs["assignments"]
    X_clustering: np.ndarray = inputs["X_clustering"]
    clustering_recipe_ids: pd.DataFrame = inputs["clustering_recipe_ids"]

    profiles = merge_metadata_assignments(recipes, assignments)
    cols = resolve_profile_columns(profiles)
    profiles = add_parsed_token_columns(
        merged=profiles,
        category_col=cols["category"],
        keywords_col=cols["keywords"],
        ingredients_col=cols["ingredients"],
    )

    row_mapping = build_row_cluster_mapping(
        clustering_recipe_ids=clustering_recipe_ids,
        assignments=assignments,
        n_rows=X_clustering.shape[0],
    )
    centroids = compute_cluster_centroids(X_clustering, row_mapping)
    representatives = build_representative_recipes(
        X=X_clustering,
        row_mapping=row_mapping,
        profiles=profiles,
        centroids=centroids,
        name_col=cols["name"],
    )

    summary_df, detailed_df = profile_clusters(
        profiles=profiles,
        representatives=representatives,
        cols=cols,
    )
    paths = save_outputs(
        out_dir=args.out,
        figures_dir=args.figures_out,
        selected_k=selected_k,
        summary_df=summary_df,
        detailed_df=detailed_df,
        representatives=representatives,
        profiles=profiles,
        cols=cols,
    )

    # Persist a small run metadata snapshot for traceability.
    metadata_path = args.out / "cluster_profile_run_metadata.json"
    metadata = {
        "selected_k": selected_k,
        "n_clusters_profiled": int(len(summary_df)),
        "input_paths": {
            "recipes": str(inputs["recipes_path"]),
            "assignments": str(inputs["assignments_path"]),
            "evaluation_summary": str(inputs["evaluation_summary_path"]),
            "clustering_matrix": str(inputs["matrix_path"]),
            "clustering_recipe_ids": str(inputs["clustering_recipe_ids_path"]),
        },
        "output_paths": {name: str(path) for name, path in paths.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print_final_summary(
        selected_k=selected_k,
        summary_df=summary_df,
        paths={**paths, "run_metadata": metadata_path},
    )


if __name__ == "__main__":
    main()
