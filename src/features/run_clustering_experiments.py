#!/usr/bin/env python3
"""
Run reproducible Week 7 clustering experiments on the canonical latent recipe matrix.

This script consumes the already-built Week 7 clustering representation and runs
MiniBatchKMeans sweeps across multiple k values. Clustering is performed on latent
embeddings (content SVD + numeric PCA) because that space is compact, denoised,
and semantically structured for recipe similarity.

Euclidean distance is appropriate in this latent space because PCA/SVD embeddings
are continuous vector representations where geometric proximity reflects shared
structure. Semantic content features dominate the representation by design, while
numeric dimensions refine boundaries, supporting future recommendation workflows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
BATCH_SIZE = 4096
N_INIT = 10
MAX_SILHOUETTE_SAMPLE = 50_000
K_VALUES = [5, 8, 10, 12, 15, 20, 25]

DEFAULT_CLUSTER_MATRIX_PATH = Path("artifacts/week7/clustering_matrix/X_recipe_clustering.npy")
DEFAULT_FEATURE_NAMES_PATH = Path("artifacts/week7/clustering_matrix/clustering_feature_names.csv")
DEFAULT_RECIPE_IDS_PATH = Path("artifacts/week7/clustering_matrix/clustering_recipe_ids.csv")
DEFAULT_MODELS_DIR = Path("artifacts/week7/clustering_models")
DEFAULT_FIGURES_DIR = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MiniBatchKMeans clustering experiments on Week 7 latent recipe "
            "embeddings without rebuilding upstream preprocessing or reductions."
        )
    )
    parser.add_argument(
        "--clustering-matrix",
        type=Path,
        default=DEFAULT_CLUSTER_MATRIX_PATH,
        help=f"Path to X_recipe_clustering.npy (default: {DEFAULT_CLUSTER_MATRIX_PATH}).",
    )
    parser.add_argument(
        "--feature-metadata",
        type=Path,
        default=DEFAULT_FEATURE_NAMES_PATH,
        help=(
            "Path to clustering_feature_names.csv "
            f"(default: {DEFAULT_FEATURE_NAMES_PATH})."
        ),
    )
    parser.add_argument(
        "--recipe-ids",
        type=Path,
        default=DEFAULT_RECIPE_IDS_PATH,
        help=f"Path to clustering_recipe_ids.csv (default: {DEFAULT_RECIPE_IDS_PATH}).",
    )
    parser.add_argument(
        "--models-out",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory for models and experiment artifacts (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Directory for plots (default: {DEFAULT_FIGURES_DIR}).",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


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
    matrix_path = resolve_existing_path(args.clustering_matrix, "Clustering matrix")
    feature_path = resolve_existing_path(args.feature_metadata, "Feature metadata CSV")
    recipe_ids_path = resolve_existing_path(args.recipe_ids, "Recipe IDs CSV")

    X = np.load(matrix_path)
    feature_metadata = pd.read_csv(feature_path)
    recipe_ids = pd.read_csv(recipe_ids_path)

    return {
        "matrix_path": matrix_path,
        "feature_path": feature_path,
        "recipe_ids_path": recipe_ids_path,
        "X": X,
        "feature_metadata": feature_metadata,
        "recipe_ids": recipe_ids,
    }


def validate_inputs(inputs: dict[str, Any]) -> None:
    X: np.ndarray = inputs["X"]
    feature_metadata: pd.DataFrame = inputs["feature_metadata"]
    recipe_ids: pd.DataFrame = inputs["recipe_ids"]

    validate_matrix_2d_finite(X, "X_recipe_clustering")

    if len(feature_metadata) != X.shape[1]:
        raise ValueError(
            "Feature metadata row count does not match matrix columns: "
            f"{len(feature_metadata)} != {X.shape[1]}"
        )
    if len(recipe_ids) != X.shape[0]:
        raise ValueError(
            "Recipe IDs row count does not match matrix rows: "
            f"{len(recipe_ids)} != {X.shape[0]}"
        )

    validate_required_columns(recipe_ids, ["row_index", "RecipeId"], "clustering_recipe_ids.csv")
    if recipe_ids["RecipeId"].isna().any():
        raise ValueError("clustering_recipe_ids.csv contains null RecipeId values.")
    if X.shape[0] < 2:
        raise ValueError("Clustering matrix must contain at least 2 rows.")
    if max(K_VALUES) >= X.shape[0]:
        raise ValueError(
            "All k values must be smaller than the number of recipes. "
            f"max(k)={max(K_VALUES)}, n_rows={X.shape[0]}"
        )


def compute_silhouette_with_sampling(
    X: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
) -> float:
    sampled_labels = labels[sample_indices]
    # Sampling is deterministic but can occasionally under-cover tiny clusters.
    # If that happens, fall back to full labels to keep the metric valid.
    if np.unique(sampled_labels).size < 2:
        return float(silhouette_score(X, labels, metric="euclidean"))
    return float(
        silhouette_score(
            X[sample_indices],
            sampled_labels,
            metric="euclidean",
        )
    )


def deterministic_sample_indices(
    n_rows: int, max_sample_size: int = MAX_SILHOUETTE_SAMPLE
) -> np.ndarray:
    # Silhouette is O(n^2)-like in pairwise distance behavior and is expensive
    # at Week-7 scale (~500k recipes). We therefore compute it on a deterministic
    # random subset capped at 50k rows to keep experiments tractable and repeatable.
    sample_size = min(n_rows, max_sample_size)
    if sample_size == n_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(RANDOM_STATE)
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


def validate_cluster_labels(labels: np.ndarray, n_rows: int, k: int) -> np.ndarray:
    if labels.shape[0] != n_rows:
        raise ValueError(f"Cluster labels row count mismatch: {labels.shape[0]} != {n_rows}")
    cluster_sizes = np.bincount(labels, minlength=k)
    if len(cluster_sizes) != k:
        raise ValueError("Unexpected cluster size vector length.")
    if (cluster_sizes == 0).any():
        empty_clusters = np.where(cluster_sizes == 0)[0].tolist()
        raise ValueError(f"Empty clusters detected for k={k}: {empty_clusters}")
    return cluster_sizes


def validate_metrics(
    k: int,
    inertia: float,
    silhouette: float,
    davies_bouldin: float,
    calinski_harabasz: float,
) -> None:
    metric_values = {
        "inertia": inertia,
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "calinski_harabasz_score": calinski_harabasz,
    }
    for name, value in metric_values.items():
        if not np.isfinite(value):
            raise ValueError(f"Invalid non-finite metric for k={k}: {name}={value}")

    if inertia < 0:
        raise ValueError(f"Inertia must be non-negative for k={k}, got {inertia}.")
    if not (-1.0 <= silhouette <= 1.0):
        raise ValueError(
            f"Silhouette score out of expected range [-1, 1] for k={k}, got {silhouette}."
        )
    if davies_bouldin < 0:
        raise ValueError(
            f"Davies-Bouldin score must be non-negative for k={k}, got {davies_bouldin}."
        )
    if calinski_harabasz <= 0:
        raise ValueError(
            "Calinski-Harabasz score must be positive "
            f"for k={k}, got {calinski_harabasz}."
        )


def export_assignments(
    recipe_ids: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    output_dir: Path,
) -> Path:
    if len(recipe_ids) != len(labels):
        raise ValueError("Cluster labels must align with recipe IDs before export.")

    path = output_dir / f"cluster_assignments_k{k}.csv"
    assignments = recipe_ids[["row_index", "RecipeId"]].copy()
    assignments["cluster"] = labels.astype(np.int64)
    assignments.to_csv(path, index=False)
    return path


def export_centroids(model: MiniBatchKMeans, k: int, output_dir: Path) -> Path:
    centers = model.cluster_centers_
    centroid_cols = [f"latent_dim_{idx + 1:03d}" for idx in range(centers.shape[1])]
    centroids = pd.DataFrame(centers, columns=centroid_cols)
    centroids.insert(0, "cluster", np.arange(k, dtype=np.int64))

    path = output_dir / f"cluster_centroids_k{k}.csv"
    centroids.to_csv(path, index=False)
    return path


def run_experiment_for_k(
    X: np.ndarray,
    recipe_ids: pd.DataFrame,
    sample_indices: np.ndarray,
    k: int,
    model_dir: Path,
) -> dict[str, Any]:
    # MiniBatchKMeans is used instead of standard KMeans because full-batch Lloyd
    # updates are expensive at large scale (~500k recipes). Mini-batch updates
    # offer substantial runtime/memory benefits while preserving useful cluster
    # structure in latent Euclidean embedding space.
    start = time.perf_counter()
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        n_init=N_INIT,
    )
    labels = model.fit_predict(X)
    runtime_seconds = float(time.perf_counter() - start)

    cluster_sizes = validate_cluster_labels(labels, n_rows=X.shape[0], k=k)
    model_path = model_dir / f"minibatch_kmeans_k{k}.joblib"
    joblib.dump(model, model_path)

    silhouette = compute_silhouette_with_sampling(X, labels, sample_indices)
    davies_bouldin = float(davies_bouldin_score(X, labels))
    calinski_harabasz = float(calinski_harabasz_score(X, labels))
    inertia = float(model.inertia_)

    validate_metrics(
        k=k,
        inertia=inertia,
        silhouette=silhouette,
        davies_bouldin=davies_bouldin,
        calinski_harabasz=calinski_harabasz,
    )

    assignment_path = export_assignments(recipe_ids, labels, k, model_dir)
    centroid_path = export_centroids(model, k, model_dir)

    return {
        "k": k,
        "inertia": inertia,
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "calinski_harabasz_score": calinski_harabasz,
        "runtime_seconds": runtime_seconds,
        "min_cluster_size": int(cluster_sizes.min()),
        "max_cluster_size": int(cluster_sizes.max()),
        "mean_cluster_size": float(cluster_sizes.mean()),
        "std_cluster_size": float(cluster_sizes.std(ddof=0)),
        "model_path": model_path,
        "assignment_path": assignment_path,
        "centroid_path": centroid_path,
    }


def evaluation_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    ordered_cols = [
        "k",
        "inertia",
        "silhouette_score",
        "davies_bouldin_score",
        "calinski_harabasz_score",
        "runtime_seconds",
        "min_cluster_size",
        "max_cluster_size",
        "mean_cluster_size",
        "std_cluster_size",
    ]
    return summary[ordered_cols].sort_values("k").reset_index(drop=True)


def plot_metric_vs_k(
    summary: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(summary["k"], summary[y_col], marker="o")
    plt.xlabel("k")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plots(summary: pd.DataFrame, figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elbow_curve": figures_dir / "clustering_elbow_curve.png",
        "silhouette_vs_k": figures_dir / "clustering_silhouette_vs_k.png",
        "davies_bouldin_vs_k": figures_dir / "clustering_davies_bouldin_vs_k.png",
        "cluster_size_variability_vs_k": figures_dir
        / "clustering_cluster_size_variability_vs_k.png",
    }
    plot_metric_vs_k(
        summary=summary,
        y_col="inertia",
        title="Elbow Curve: Inertia vs k",
        y_label="Inertia",
        out_path=paths["elbow_curve"],
    )
    plot_metric_vs_k(
        summary=summary,
        y_col="silhouette_score",
        title="Silhouette Score vs k",
        y_label="Silhouette Score",
        out_path=paths["silhouette_vs_k"],
    )
    plot_metric_vs_k(
        summary=summary,
        y_col="davies_bouldin_score",
        title="Davies-Bouldin Score vs k",
        y_label="Davies-Bouldin Score",
        out_path=paths["davies_bouldin_vs_k"],
    )
    plot_metric_vs_k(
        summary=summary,
        y_col="std_cluster_size",
        title="Cluster Size Variability vs k",
        y_label="Std Cluster Size",
        out_path=paths["cluster_size_variability_vs_k"],
    )
    return paths


def estimate_mb(matrix: np.ndarray) -> float:
    return float(matrix.nbytes / (1024**2))


def build_config(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    summary: pd.DataFrame,
    sample_size: int,
    total_runtime_seconds: float,
    summary_path: Path,
    plot_paths: dict[str, Path],
    model_paths: dict[int, Path],
    assignment_paths: dict[int, Path],
    centroid_paths: dict[int, Path],
) -> dict[str, Any]:
    return {
        "k_values": K_VALUES,
        "random_state": RANDOM_STATE,
        "batch_size": BATCH_SIZE,
        "n_init": N_INIT,
        "sampled_silhouette_size": sample_size,
        "input_paths": {
            "X_recipe_clustering": str(inputs["matrix_path"]),
            "clustering_feature_names": str(inputs["feature_path"]),
            "clustering_recipe_ids": str(inputs["recipe_ids_path"]),
        },
        "input_shapes": {
            "X_recipe_clustering": list(inputs["X"].shape),
            "clustering_feature_names": [len(inputs["feature_metadata"]), inputs["feature_metadata"].shape[1]],
            "clustering_recipe_ids": [len(inputs["recipe_ids"]), inputs["recipe_ids"].shape[1]],
        },
        "runtime_metadata": {
            "total_runtime_seconds": total_runtime_seconds,
            "per_k_runtime_seconds": {
                str(int(row["k"])): float(row["runtime_seconds"])
                for _, row in summary.iterrows()
            },
        },
        "output_paths": {
            "models_dir": str(args.models_out),
            "figures_dir": str(args.figures_out),
            "evaluation_summary_csv": str(summary_path),
            "config_json": str(args.models_out / "clustering_experiment_config.json"),
            "models_by_k": {str(k): str(path) for k, path in model_paths.items()},
            "assignments_by_k": {str(k): str(path) for k, path in assignment_paths.items()},
            "centroids_by_k": {str(k): str(path) for k, path in centroid_paths.items()},
            "plots": {name: str(path) for name, path in plot_paths.items()},
        },
        "evaluation_metrics_generated": [
            "inertia",
            "silhouette_score",
            "davies_bouldin_score",
            "calinski_harabasz_score",
            "cluster_size_distribution",
            "runtime_seconds",
        ],
        "matrix_memory_estimates_mb": {
            "X_recipe_clustering": estimate_mb(inputs["X"]),
        },
    }


def print_final_summary(
    summary: pd.DataFrame,
    total_runtime_seconds: float,
    models_dir: Path,
    figures_dir: Path,
    matrix_shape: tuple[int, int],
) -> None:
    best_silhouette_row = summary.loc[summary["silhouette_score"].idxmax()]
    best_davies_row = summary.loc[summary["davies_bouldin_score"].idxmin()]

    print("Clustering experiments complete.")
    print(f"Best silhouette k: {int(best_silhouette_row['k'])}")
    print(f"Best Davies-Bouldin k: {int(best_davies_row['k'])}")
    print(f"Total runtime (seconds): {total_runtime_seconds:.4f}")
    print(f"Models/output directory: {models_dir}")
    print(f"Figures directory: {figures_dir}")
    print(f"Clustering matrix shape: {matrix_shape}")


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    inputs = load_inputs(args)
    validate_inputs(inputs)

    X: np.ndarray = inputs["X"]
    recipe_ids: pd.DataFrame = inputs["recipe_ids"]

    args.models_out.mkdir(parents=True, exist_ok=True)
    args.figures_out.mkdir(parents=True, exist_ok=True)

    sample_indices = deterministic_sample_indices(X.shape[0], MAX_SILHOUETTE_SAMPLE)
    sample_size = int(len(sample_indices))

    experiment_rows: list[dict[str, Any]] = []
    model_paths: dict[int, Path] = {}
    assignment_paths: dict[int, Path] = {}
    centroid_paths: dict[int, Path] = {}

    for k in K_VALUES:
        result = run_experiment_for_k(
            X=X,
            recipe_ids=recipe_ids,
            sample_indices=sample_indices,
            k=k,
            model_dir=args.models_out,
        )
        model_paths[k] = result.pop("model_path")
        assignment_paths[k] = result.pop("assignment_path")
        centroid_paths[k] = result.pop("centroid_path")
        experiment_rows.append(result)

    summary = evaluation_summary_frame(experiment_rows)
    summary_path = args.models_out / "clustering_evaluation_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_paths = save_plots(summary, args.figures_out)
    total_runtime_seconds = float(time.perf_counter() - total_start)

    config = build_config(
        args=args,
        inputs=inputs,
        summary=summary,
        sample_size=sample_size,
        total_runtime_seconds=total_runtime_seconds,
        summary_path=summary_path,
        plot_paths=plot_paths,
        model_paths=model_paths,
        assignment_paths=assignment_paths,
        centroid_paths=centroid_paths,
    )
    config_path = args.models_out / "clustering_experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print_final_summary(
        summary=summary,
        total_runtime_seconds=total_runtime_seconds,
        models_dir=args.models_out,
        figures_dir=args.figures_out,
        matrix_shape=tuple(X.shape),
    )


if __name__ == "__main__":
    main()
