#!/usr/bin/env python3
"""
Generate remaining Week 7 clustering evaluation/reporting deliverables.

This script reuses existing Week 7 clustering and profiling artifacts to produce
rubric-ready evaluation tables, sweep plots, and markdown summaries without
changing the clustering pipeline itself.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EVALUATION_SUMMARY = Path(
    "artifacts/week7/clustering_models/clustering_evaluation_summary.csv"
)
DEFAULT_SELECTED_ASSIGNMENTS = Path("artifacts/week7/clustering_models/cluster_assignments_k5.csv")
DEFAULT_CLUSTER_PROFILE_SUMMARY = Path(
    "artifacts/week7/clustering_reports/cluster_profile_summary.csv"
)
DEFAULT_CLUSTER_NUMERIC_STATS = Path(
    "artifacts/week7/clustering_reports/cluster_numeric_statistics.csv"
)
DEFAULT_CLUSTERING_MATRIX_CONFIG = Path(
    "artifacts/week7/clustering_matrix/clustering_matrix_config.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/week7/clustering_reports")
DEFAULT_FIGURES_DIR = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Week 7 validation/reporting artifacts from existing clustering "
            "and profiling outputs."
        )
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=DEFAULT_EVALUATION_SUMMARY,
        help=f"Path to clustering_evaluation_summary.csv (default: {DEFAULT_EVALUATION_SUMMARY}).",
    )
    parser.add_argument(
        "--selected-assignments",
        type=Path,
        default=DEFAULT_SELECTED_ASSIGNMENTS,
        help=f"Path to selected cluster_assignments_k*.csv (default: {DEFAULT_SELECTED_ASSIGNMENTS}).",
    )
    parser.add_argument(
        "--cluster-profile-summary",
        type=Path,
        default=DEFAULT_CLUSTER_PROFILE_SUMMARY,
        help=f"Path to cluster_profile_summary.csv (default: {DEFAULT_CLUSTER_PROFILE_SUMMARY}).",
    )
    parser.add_argument(
        "--cluster-numeric-stats",
        type=Path,
        default=DEFAULT_CLUSTER_NUMERIC_STATS,
        help=f"Path to cluster_numeric_statistics.csv (default: {DEFAULT_CLUSTER_NUMERIC_STATS}).",
    )
    parser.add_argument(
        "--representative-recipes",
        type=Path,
        default=None,
        help=(
            "Optional path to representative_recipes_k*.csv. If omitted, inferred "
            "from selected k in the output directory."
        ),
    )
    parser.add_argument(
        "--clustering-matrix-config",
        type=Path,
        default=DEFAULT_CLUSTERING_MATRIX_CONFIG,
        help=(
            "Optional path to clustering_matrix_config.json for representation details "
            f"(default: {DEFAULT_CLUSTERING_MATRIX_CONFIG})."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for reporting artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Output directory for sweep plots (default: {DEFAULT_FIGURES_DIR}).",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def infer_selected_k(assignments_path: Path) -> int:
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


def load_inputs(args: argparse.Namespace, selected_k: int) -> dict[str, Any]:
    evaluation_path = resolve_existing_path(args.evaluation_summary, "Evaluation summary CSV")
    selected_assignments_path = resolve_existing_path(
        args.selected_assignments, "Selected assignments CSV"
    )
    profile_summary_path = resolve_existing_path(
        args.cluster_profile_summary, "Cluster profile summary CSV"
    )
    numeric_stats_path = resolve_existing_path(args.cluster_numeric_stats, "Cluster numeric stats CSV")

    if args.representative_recipes is None:
        representative_path = args.out / f"representative_recipes_k{selected_k}.csv"
    else:
        representative_path = args.representative_recipes
    representative_path = resolve_existing_path(representative_path, "Representative recipes CSV")

    matrix_config_path = args.clustering_matrix_config
    matrix_config: dict[str, Any] | None = None
    if matrix_config_path.exists():
        matrix_config = json.loads(matrix_config_path.read_text(encoding="utf-8"))

    return {
        "evaluation_path": evaluation_path,
        "selected_assignments_path": selected_assignments_path,
        "profile_summary_path": profile_summary_path,
        "numeric_stats_path": numeric_stats_path,
        "representative_path": representative_path,
        "matrix_config_path": matrix_config_path,
        "evaluation": pd.read_csv(evaluation_path),
        "selected_assignments": pd.read_csv(selected_assignments_path),
        "profile_summary": pd.read_csv(profile_summary_path),
        "numeric_stats": pd.read_csv(numeric_stats_path),
        "representative": pd.read_csv(representative_path),
        "matrix_config": matrix_config,
    }


def validate_inputs(inputs: dict[str, Any], selected_k: int) -> None:
    evaluation = inputs["evaluation"]
    selected_assignments = inputs["selected_assignments"]
    profile_summary = inputs["profile_summary"]
    numeric_stats = inputs["numeric_stats"]
    representative = inputs["representative"]

    validate_required_columns(
        evaluation,
        ["k", "inertia", "silhouette_score", "davies_bouldin_score"],
        "clustering_evaluation_summary.csv",
    )
    validate_required_columns(selected_assignments, ["RecipeId", "cluster"], "selected assignments")
    validate_required_columns(
        profile_summary,
        [
            "cluster",
            "recipe_count",
            "dataset_percentage",
            "dominant_category",
            "dominant_keywords",
            "dominant_ingredients",
            "avg_calories",
            "avg_protein",
            "avg_total_time",
        ],
        "cluster_profile_summary.csv",
    )
    validate_required_columns(
        numeric_stats,
        ["cluster", "mean_calories", "mean_protein", "mean_total_time"],
        "cluster_numeric_statistics.csv",
    )
    validate_required_columns(
        representative,
        ["cluster", "rank_within_cluster", "RecipeId"],
        "representative_recipes CSV",
    )

    tested_k = sorted(evaluation["k"].dropna().astype(int).unique().tolist())
    if not tested_k:
        raise ValueError("Evaluation summary contains no tested k values.")
    if selected_k not in tested_k:
        raise ValueError(
            f"Selected k={selected_k} from assignments is not present in evaluation summary."
        )

    selected_cluster_count = selected_assignments["cluster"].dropna().astype(int).nunique()
    if selected_cluster_count != selected_k:
        raise ValueError(
            f"Selected assignments imply k={selected_k}, but found {selected_cluster_count} clusters."
        )

    profile_clusters = sorted(profile_summary["cluster"].dropna().astype(int).unique().tolist())
    if len(profile_clusters) != selected_k:
        raise ValueError(
            f"cluster_profile_summary contains {len(profile_clusters)} clusters, expected {selected_k}."
        )

    stats_clusters = sorted(numeric_stats["cluster"].dropna().astype(int).unique().tolist())
    if stats_clusters != profile_clusters:
        raise ValueError("cluster_numeric_statistics clusters do not match cluster_profile_summary.")


def build_validation_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    eval_sorted = evaluation.copy()
    eval_sorted["k"] = eval_sorted["k"].astype(int)
    eval_sorted = eval_sorted.sort_values("k").drop_duplicates(subset=["k"], keep="last")

    runtime = (
        eval_sorted["runtime_seconds"].astype(float)
        if "runtime_seconds" in eval_sorted.columns
        else pd.Series(np.nan, index=eval_sorted.index, dtype=float)
    )

    validation = pd.DataFrame(
        {
            "k": eval_sorted["k"].astype(int),
            "inertia": eval_sorted["inertia"].astype(float),
            "silhouette_score": eval_sorted["silhouette_score"].astype(float),
            "davies_bouldin_score": eval_sorted["davies_bouldin_score"].astype(float),
            "cluster_count": eval_sorted["k"].astype(int),
            "runtime_seconds": runtime,
        }
    )
    return validation.reset_index(drop=True)


def plot_metric_sweep(
    validation: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(validation["k"], validation[y_col], marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def split_top_tokens(text: str, limit: int = 3) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = [part.strip() for part in text.split("|")]
    return [token for token in tokens if token][:limit]


def classify_level(value: float, baseline: float) -> str:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0:
        return "moderate"
    ratio = value / baseline
    if ratio >= 1.15:
        return "high"
    if ratio <= 0.85:
        return "low"
    return "moderate"


def classify_time(value: float, baseline: float) -> str:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0:
        return "moderate"
    ratio = value / baseline
    if ratio >= 1.15:
        return "long"
    if ratio <= 0.85:
        return "short"
    return "moderate"


def build_failure_analysis_markdown(
    validation: pd.DataFrame,
    profile_summary: pd.DataFrame,
    selected_k: int,
) -> str:
    best_silhouette_row = validation.loc[validation["silhouette_score"].idxmax()]
    best_davies_row = validation.loc[validation["davies_bouldin_score"].idxmin()]
    selected_row = validation.loc[validation["k"].eq(selected_k)].iloc[0]

    inertia_drop = float(validation["inertia"].iloc[0] - validation["inertia"].iloc[-1])
    silhouette_span = float(validation["silhouette_score"].max() - validation["silhouette_score"].min())
    davies_span = float(
        validation["davies_bouldin_score"].max() - validation["davies_bouldin_score"].min()
    )
    selected_sil = float(selected_row["silhouette_score"])
    selected_db = float(selected_row["davies_bouldin_score"])

    disagreement = int(best_silhouette_row["k"]) != int(best_davies_row["k"])
    dominant_categories = profile_summary["dominant_category"].astype(str).tolist()
    category_overlap_count = len(dominant_categories) - len(set(dominant_categories))

    max_cluster = int(profile_summary["recipe_count"].max())
    min_cluster = int(profile_summary["recipe_count"].min())
    size_ratio = float(max_cluster / max(min_cluster, 1))

    lines = [
        "# Week 7 Failure Analysis Summary",
        "",
        "## Sensitivity across k values",
        (
            f"- Tested k values: {validation['k'].tolist()}."
            f" Inertia decreases by {inertia_drop:.2f} across the sweep,"
            f" silhouette spans {silhouette_span:.6f}, and Davies-Bouldin spans {davies_span:.6f}."
        ),
        (
            f"- Selected configuration k={selected_k} has silhouette={selected_sil:.6f} "
            f"and Davies-Bouldin={selected_db:.6f}."
        ),
        "",
        "## Disagreement between validation metrics",
        (
            f"- Best silhouette occurs at k={int(best_silhouette_row['k'])}, while best "
            f"Davies-Bouldin occurs at k={int(best_davies_row['k'])}."
        ),
        (
            "- This indicates metric disagreement and suggests that cluster quality "
            "is multi-objective rather than optimized by a single scalar criterion."
            if disagreement
            else "- Silhouette and Davies-Bouldin agree on the same k in this run."
        ),
        "",
        "## High-dimensional feature and model assumptions",
        (
            "- The representation compresses high-dimensional sparse content signals "
            "with SVD plus numeric PCA; residual sparsity and anisotropy can still "
            "stress K-means spherical-cluster assumptions."
        ),
        (
            "- K-means assumes roughly convex, similarly scaled clusters under "
            "Euclidean distance; recipe domains may violate these assumptions."
        ),
        "",
        "## Outliers and cluster distribution imbalance",
        (
            f"- Cluster size imbalance remains material (largest={max_cluster}, "
            f"smallest={min_cluster}, ratio={size_ratio:.2f})."
        ),
        (
            "- Extreme nutritional/time outliers can inflate within-cluster variance. "
            "For readability, visualization boxplots hide extreme fliers only; raw rows are unchanged."
        ),
        "",
        "## Visualization limitations",
        (
            "- PCA/t-SNE plots are exploratory diagnostics only and should not be treated "
            "as proof of globally separable clusters."
        ),
        (
            "- t-SNE preserves local neighborhoods but distorts global geometry; "
            "apparent distances between distant groups are not metrically reliable."
        ),
        "",
        "## Category overlap risk",
        (
            f"- Dominant-category overlap across clusters is non-zero "
            f"(overlap count={category_overlap_count}), suggesting category semantics "
            "can span multiple latent groups."
        ),
    ]
    return "\n".join(lines) + "\n"


def build_cluster_interpretation_markdown(
    profile_summary: pd.DataFrame,
    representative: pd.DataFrame,
    selected_k: int,
) -> str:
    total_recipes = int(profile_summary["recipe_count"].sum())
    baseline_calories = float(np.average(profile_summary["avg_calories"], weights=profile_summary["recipe_count"]))
    baseline_protein = float(np.average(profile_summary["avg_protein"], weights=profile_summary["recipe_count"]))
    baseline_total_time = float(
        np.average(profile_summary["avg_total_time"], weights=profile_summary["recipe_count"])
    )

    lines = [
        "# Week 7 Cluster Interpretation Summary",
        "",
        (
            f"Selected configuration uses **k={selected_k}** with approximately "
            f"{total_recipes:,} recipes profiled."
        ),
        "",
    ]

    reps = representative.copy()
    recipe_name_col = "recipe_name" if "recipe_name" in reps.columns else "RecipeId"

    for _, row in profile_summary.sort_values("cluster").iterrows():
        cluster = int(row["cluster"])
        count = int(row["recipe_count"])
        pct = float(row["dataset_percentage"])

        cal_level = classify_level(float(row["avg_calories"]), baseline_calories)
        protein_level = classify_level(float(row["avg_protein"]), baseline_protein)
        time_level = classify_time(float(row["avg_total_time"]), baseline_total_time)

        keywords = split_top_tokens(str(row["dominant_keywords"]))
        ingredients = split_top_tokens(str(row["dominant_ingredients"]))
        category = str(row["dominant_category"]).strip() or "mixed-category"

        rep_names = (
            reps.loc[reps["cluster"].astype(int).eq(cluster)]
            .sort_values("rank_within_cluster")
            .head(3)[recipe_name_col]
            .astype(str)
            .tolist()
        )
        rep_text = ", ".join(rep_names) if rep_names else "No representative recipe names available."

        tendency_parts = [category]
        tendency_parts.extend(ingredients[:2])
        tendency_parts.extend(keywords[:2])
        tendency_text = ", ".join([part for part in tendency_parts if part]) or "general recipe style"

        operational = (
            f"Use Cluster {cluster} as a candidate pool for {category.lower()}-leaning recommendations; "
            "re-rank within-cluster items by user preferences and constraints."
        )

        lines.extend(
            [
                f"## Cluster {cluster}",
                f"- **Approximate size:** {count:,} recipes ({pct:.2f}% of dataset).",
                (
                    f"- **Dominant nutritional/complexity characteristics:** "
                    f"{protein_level}-protein, {cal_level}-calorie, {time_level} preparation time."
                ),
                f"- **Ingredient/style tendencies:** {tendency_text}.",
                f"- **Example representative recipes:** {rep_text}.",
                f"- **Operational interpretation:** {operational}",
                "",
            ]
        )

    return "\n".join(lines)


def build_week7_summary_markdown(
    validation: pd.DataFrame,
    profile_summary: pd.DataFrame,
    selected_k: int,
    best_silhouette_k: int,
    best_davies_k: int,
    matrix_config: dict[str, Any] | None,
) -> str:
    selected_row = validation.loc[validation["k"].eq(selected_k)].iloc[0]
    selected_sil = float(selected_row["silhouette_score"])
    selected_db = float(selected_row["davies_bouldin_score"])
    selected_inertia = float(selected_row["inertia"])

    if matrix_config is not None:
        weights = matrix_config.get("representation_weights", {})
        content_w = weights.get("CONTENT_WEIGHT", "unknown")
        numeric_w = weights.get("NUMERIC_WEIGHT", "unknown")
        shape = matrix_config.get("matrix_shapes", {}).get("X_recipe_clustering", "unknown")
        representation_text = (
            "Canonical Week 7 clustering matrix combining semantic SVD content and scaled "
            f"numeric PCA features (weights: content={content_w}, numeric={numeric_w}, shape={shape})."
        )
    else:
        representation_text = (
            "Canonical Week 7 clustering matrix combining semantic SVD content and scaled "
            "numeric PCA features (details unavailable because clustering_matrix_config.json was not found)."
        )

    k_justification = (
        f"Selected k={selected_k} because it balances metric quality and interpretability."
    )
    if selected_k == best_silhouette_k == best_davies_k:
        k_justification = (
            f"Selected k={selected_k} because it is jointly optimal for silhouette and Davies-Bouldin."
        )
    elif selected_k == best_silhouette_k:
        k_justification = (
            f"Selected k={selected_k} because it maximizes silhouette while remaining comparable on Davies-Bouldin."
        )
    elif selected_k == best_davies_k:
        k_justification = (
            f"Selected k={selected_k} because it minimizes Davies-Bouldin with acceptable silhouette."
        )

    n_clusters = int(profile_summary["cluster"].nunique())

    lines = [
        "# Week 7 Summary",
        "",
        "## Chosen representation",
        f"- {representation_text}",
        "",
        "## Why K-means (MiniBatchKMeans) was used",
        (
            "- K-means provides a scalable baseline for large recipe corpora, offers "
            "fast assignment for downstream retrieval, and works naturally with latent "
            "Euclidean embeddings."
        ),
        "",
        "## Selected k and justification",
        f"- {k_justification}",
        (
            f"- Selected-k metrics: inertia={selected_inertia:.2f}, "
            f"silhouette={selected_sil:.6f}, Davies-Bouldin={selected_db:.6f}."
        ),
        (
            f"- Reference optima: best silhouette k={best_silhouette_k}, "
            f"best Davies-Bouldin k={best_davies_k}."
        ),
        "",
        "## Validation metric interpretation",
        (
            "- Inertia decreases with higher k as expected; silhouette and "
            "Davies-Bouldin provide complementary quality signals and may disagree."
        ),
        "",
        "## Cluster usefulness for recommendation",
        (
            f"- {n_clusters} interpreted clusters provide structured candidate pools "
            "that can be re-ranked by user preferences, dietary constraints, and context."
        ),
        "",
        "## Known limitations",
        "- K-means assumes roughly spherical clusters under Euclidean distance.",
        "- Cluster size imbalance and outliers can affect centroid stability.",
        "- PCA/t-SNE plots are exploratory only and should not be treated as definitive geometry.",
        (
            "- For readability, cluster boxplots hide extreme outlier markers only "
            "(raw data remain unchanged)."
        ),
        "",
    ]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    selected_k = infer_selected_k(args.selected_assignments)

    inputs = load_inputs(args, selected_k)
    validate_inputs(inputs, selected_k)

    args.out.mkdir(parents=True, exist_ok=True)
    args.figures_out.mkdir(parents=True, exist_ok=True)

    validation = build_validation_table(inputs["evaluation"])
    validation_path = args.out / "cluster_validation_metrics.csv"
    validation.to_csv(validation_path, index=False)

    inertia_plot = args.figures_out / "inertia_vs_k.png"
    silhouette_plot = args.figures_out / "silhouette_vs_k.png"
    davies_plot = args.figures_out / "davies_bouldin_vs_k.png"

    plot_metric_sweep(
        validation=validation,
        y_col="inertia",
        y_label="Inertia",
        title="Inertia vs Number of Clusters (k)",
        out_path=inertia_plot,
    )
    plot_metric_sweep(
        validation=validation,
        y_col="silhouette_score",
        y_label="Silhouette Score",
        title="Silhouette Score vs Number of Clusters (k)",
        out_path=silhouette_plot,
    )
    plot_metric_sweep(
        validation=validation,
        y_col="davies_bouldin_score",
        y_label="Davies-Bouldin Score",
        title="Davies-Bouldin Score vs Number of Clusters (k)",
        out_path=davies_plot,
    )

    failure_md = build_failure_analysis_markdown(
        validation=validation,
        profile_summary=inputs["profile_summary"],
        selected_k=selected_k,
    )
    failure_path = args.out / "failure_analysis_summary.md"
    write_text(failure_path, failure_md)

    interpretation_md = build_cluster_interpretation_markdown(
        profile_summary=inputs["profile_summary"],
        representative=inputs["representative"],
        selected_k=selected_k,
    )
    interpretation_path = args.out / "cluster_interpretation_summary.md"
    write_text(interpretation_path, interpretation_md)

    best_silhouette_k = int(validation.loc[validation["silhouette_score"].idxmax(), "k"])
    best_davies_k = int(validation.loc[validation["davies_bouldin_score"].idxmin(), "k"])

    week7_md = build_week7_summary_markdown(
        validation=validation,
        profile_summary=inputs["profile_summary"],
        selected_k=selected_k,
        best_silhouette_k=best_silhouette_k,
        best_davies_k=best_davies_k,
        matrix_config=inputs["matrix_config"],
    )
    week7_summary_path = args.out / "week7_summary.md"
    write_text(week7_summary_path, week7_md)

    generated_paths = {
        "cluster_validation_metrics": validation_path,
        "failure_analysis_summary": failure_path,
        "cluster_interpretation_summary": interpretation_path,
        "week7_summary": week7_summary_path,
        "inertia_vs_k_plot": inertia_plot,
        "silhouette_vs_k_plot": silhouette_plot,
        "davies_bouldin_vs_k_plot": davies_plot,
    }

    print("Week 7 reporting artifacts generated.")
    print("Generated artifact paths:")
    for path in generated_paths.values():
        print(f"- {path}")
    print(f"Number of tested k values: {len(validation)}")
    print(f"Selected k: {selected_k}")
    print(f"Best silhouette k: {best_silhouette_k}")
    print(f"Best Davies-Bouldin k: {best_davies_k}")


if __name__ == "__main__":
    main()
