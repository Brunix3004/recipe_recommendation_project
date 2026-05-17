#!/usr/bin/env python3
"""
Create exploratory 2D visualizations of recipe clusters from reduced embeddings.

This script is strictly visualization-oriented. It does not build reusable
recommendation embeddings and does not rerun feature engineering or clustering.

Important interpretation notes:
1. PCA here is for plotting only and is separate from the earlier representation
   PCA used in Week 5 dimensionality reduction.
2. t-SNE is visualization-only: it emphasizes local neighborhood structure but
   does not preserve global geometry faithfully.
3. Sampling is required at this dataset scale (>500k rows) to keep projection
   methods computationally tractable and memory efficient.
4. Visualization is still valuable because cluster separation/overlap patterns
   help interpret learned latent recipe structure.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MATRIX_PATH = Path("artifacts/week5/pca_svd/X_recipe_reduced.npy")
DEFAULT_ASSIGNMENTS_PATH = Path("artifacts/week7/clustering_models/cluster_assignments_k5.csv")
DEFAULT_FEATURE_NAMES_PATH = Path("artifacts/week5/pca_svd/reduced_feature_names.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/week7/cluster_visualizations")
DEFAULT_FIGURES_DIR = Path("reports/figures")

DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_RANDOM_STATE = 42
DEFAULT_TSNE_PERPLEXITY = 40.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PCA/t-SNE cluster visualizations from the reduced recipe "
            "embedding matrix using deterministic sampling."
        )
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help=f"Path to X_recipe_reduced.npy (default: {DEFAULT_MATRIX_PATH}).",
    )
    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_PATH,
        help=f"Path to cluster_assignments_k*.csv (default: {DEFAULT_ASSIGNMENTS_PATH}).",
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=DEFAULT_FEATURE_NAMES_PATH,
        help=f"Path to reduced_feature_names.csv (default: {DEFAULT_FEATURE_NAMES_PATH}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for visualization artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--figures-out",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help=f"Output directory for plots (default: {DEFAULT_FIGURES_DIR}).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Sample size for visualization projections (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Deterministic random state (default: {DEFAULT_RANDOM_STATE}).",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=DEFAULT_TSNE_PERPLEXITY,
        help=f"t-SNE perplexity in [30, 50] (default: {DEFAULT_TSNE_PERPLEXITY}).",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def infer_selected_k_from_filename(path: Path) -> int | None:
    match = re.search(r"_k(\d+)\.csv$", path.name)
    return int(match.group(1)) if match is not None else None


def validate_matrix_2d_finite(matrix: np.ndarray, name: str) -> None:
    if not isinstance(matrix, np.ndarray):
        raise ValueError(f"{name} must be a numpy array.")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


def validate_required_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    matrix_path = resolve_existing_path(args.matrix, "Reduced recipe matrix")
    assignments_path = resolve_existing_path(args.assignments, "Cluster assignments CSV")
    feature_names_path = resolve_existing_path(args.feature_names, "Reduced feature names CSV")

    return {
        "matrix_path": matrix_path,
        "assignments_path": assignments_path,
        "feature_names_path": feature_names_path,
        "X_reduced": np.load(matrix_path),
        "assignments": pd.read_csv(assignments_path),
        "feature_names": pd.read_csv(feature_names_path),
    }


def validate_inputs(inputs: dict[str, Any]) -> None:
    X_reduced: np.ndarray = inputs["X_reduced"]
    assignments: pd.DataFrame = inputs["assignments"]
    feature_names: pd.DataFrame = inputs["feature_names"]

    validate_matrix_2d_finite(X_reduced, "X_recipe_reduced")
    validate_required_columns(assignments, ["row_index", "RecipeId", "cluster"], "Assignments CSV")

    if len(feature_names) != X_reduced.shape[1]:
        raise ValueError(
            "Reduced feature names row count does not match matrix columns: "
            f"{len(feature_names)} != {X_reduced.shape[1]}"
        )
    if len(assignments) != X_reduced.shape[0]:
        raise ValueError(
            "Assignments row count does not match matrix rows: "
            f"{len(assignments)} != {X_reduced.shape[0]}"
        )
    if assignments["RecipeId"].duplicated().any():
        dup_count = int(assignments["RecipeId"].duplicated().sum())
        raise ValueError(f"Assignments contain duplicate RecipeId values: {dup_count}")
    if assignments["row_index"].duplicated().any():
        dup_count = int(assignments["row_index"].duplicated().sum())
        raise ValueError(f"Assignments contain duplicate row_index values: {dup_count}")
    if assignments["cluster"].isna().any():
        raise ValueError("Assignments contain null cluster labels.")
    try:
        assignments["cluster"].astype(np.int64)
    except ValueError as exc:
        raise ValueError("Assignments cluster labels must be integer-like.") from exc

    # Matching row ordering is enforced by row_index consistency.
    ordered = assignments.sort_values("row_index").reset_index(drop=True)
    expected_index = np.arange(X_reduced.shape[0], dtype=np.int64)
    if not np.array_equal(ordered["row_index"].to_numpy(dtype=np.int64), expected_index):
        raise ValueError(
            "Assignments row_index must be contiguous [0, n_rows-1] to match matrix ordering."
        )


def sample_row_indices(
    assignments: pd.DataFrame,
    n_rows: int,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError(f"sample_size must be > 0, got {sample_size}.")
    size = min(n_rows, sample_size)
    if size == n_rows:
        return np.arange(n_rows, dtype=np.int64)

    ordered = assignments.sort_values("row_index").reset_index(drop=True)
    clusters = sorted(ordered["cluster"].astype(np.int64).unique().tolist())
    if size < len(clusters):
        raise ValueError(
            "sample_size must be at least the number of clusters to ensure cluster coverage. "
            f"sample_size={size}, n_clusters={len(clusters)}"
        )

    # Ensure at least one sampled row per cluster so all clusters appear in
    # visualization centroids and legends.
    mandatory = []
    for cluster in clusters:
        cluster_rows = ordered.loc[ordered["cluster"].astype(np.int64).eq(cluster), "row_index"]
        mandatory.append(int(cluster_rows.iloc[0]))
    mandatory_idx = np.array(sorted(set(mandatory)), dtype=np.int64)

    remaining_needed = size - len(mandatory_idx)
    if remaining_needed <= 0:
        return mandatory_idx

    pool = np.setdiff1d(np.arange(n_rows, dtype=np.int64), mandatory_idx, assume_unique=True)
    rng = np.random.default_rng(random_state)
    extra = np.sort(rng.choice(pool, size=remaining_needed, replace=False))
    return np.sort(np.concatenate([mandatory_idx, extra]))


def build_sample_frame(assignments: pd.DataFrame, sample_idx: np.ndarray) -> pd.DataFrame:
    ordered = assignments.sort_values("row_index").reset_index(drop=True)
    sample_meta = ordered.loc[sample_idx, ["row_index", "RecipeId", "cluster"]].copy()
    sample_meta["row_index"] = sample_meta["row_index"].astype(np.int64)
    sample_meta["cluster"] = sample_meta["cluster"].astype(np.int64)
    return sample_meta


def run_pca_2d(X_sample: np.ndarray, random_state: int) -> tuple[np.ndarray, PCA]:
    # This PCA is an additional projection for 2D plotting. It is not the same as
    # the representation-learning PCA used earlier in the pipeline.
    pca = PCA(n_components=2, random_state=random_state)
    X_pca_2d = pca.fit_transform(X_sample).astype(np.float32)
    return X_pca_2d, pca


def run_tsne_2d(
    X_sample: np.ndarray,
    random_state: int,
    perplexity: float,
) -> tuple[np.ndarray, TSNE]:
    if not (30.0 <= perplexity <= 50.0):
        raise ValueError(f"t-SNE perplexity must be in [30, 50], got {perplexity}.")

    # t-SNE is intentionally run only on sampled data.
    # It preserves local neighborhoods but does not preserve global geometry.
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    X_tsne_2d = tsne.fit_transform(X_sample).astype(np.float32)
    return X_tsne_2d, tsne


def build_coordinates_df(
    sample_meta: pd.DataFrame,
    coords: np.ndarray,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    out = sample_meta.copy()
    out[x_col] = coords[:, 0]
    out[y_col] = coords[:, 1]
    return out


def get_feature_labels(feature_names: pd.DataFrame, n_dims: int) -> list[str]:
    if "feature_name" in feature_names.columns:
        labels = feature_names["feature_name"].astype(str).tolist()
    else:
        labels = [f"latent_dim_{i + 1:03d}" for i in range(len(feature_names))]
    if len(labels) != n_dims:
        raise ValueError(
            f"Feature label length mismatch: {len(labels)} != {n_dims} (matrix dims)."
        )
    return labels


def compute_visual_centroids(
    X_sample: np.ndarray,
    X_pca_2d: np.ndarray,
    X_tsne_2d: np.ndarray,
    sample_meta: pd.DataFrame,
    feature_labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clusters = sorted(sample_meta["cluster"].unique().tolist())

    for cluster in clusters:
        mask = sample_meta["cluster"].eq(cluster).to_numpy()
        if not mask.any():
            raise ValueError(f"Empty sampled cluster encountered: {cluster}")

        centroid_original = X_sample[mask].mean(axis=0)
        centroid_pca = X_pca_2d[mask].mean(axis=0)
        centroid_tsne = X_tsne_2d[mask].mean(axis=0)

        for dim_idx, value in enumerate(centroid_original, start=1):
            rows.append(
                {
                    "cluster": int(cluster),
                    "space": "original_reduced_space",
                    "dimension_index": int(dim_idx),
                    "dimension_name": feature_labels[dim_idx - 1],
                    "centroid_value": float(value),
                }
            )
        rows.append(
            {
                "cluster": int(cluster),
                "space": "pca_2d",
                "dimension_index": 1,
                "dimension_name": "pca_1",
                "centroid_value": float(centroid_pca[0]),
            }
        )
        rows.append(
            {
                "cluster": int(cluster),
                "space": "pca_2d",
                "dimension_index": 2,
                "dimension_name": "pca_2",
                "centroid_value": float(centroid_pca[1]),
            }
        )
        rows.append(
            {
                "cluster": int(cluster),
                "space": "tsne_2d",
                "dimension_index": 1,
                "dimension_name": "tsne_1",
                "centroid_value": float(centroid_tsne[0]),
            }
        )
        rows.append(
            {
                "cluster": int(cluster),
                "space": "tsne_2d",
                "dimension_index": 2,
                "dimension_name": "tsne_2",
                "centroid_value": float(centroid_tsne[1]),
            }
        )

    return pd.DataFrame(rows).sort_values(["cluster", "space", "dimension_index"])


def scatter_plot_by_cluster(
    coords_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    clusters = sorted(coords_df["cluster"].unique().tolist())
    cmap = plt.get_cmap("tab20", max(len(clusters), 1))

    for i, cluster in enumerate(clusters):
        subset = coords_df.loc[coords_df["cluster"].eq(cluster)]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            s=9,
            alpha=0.45,  # alpha helps show density in crowded regions.
            color=cmap(i),
            label=f"cluster {cluster}",
            edgecolors="none",
        )

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(markerscale=1.8, fontsize=8, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_outputs(
    out_dir: Path,
    figures_dir: Path,
    pca_coords: pd.DataFrame,
    tsne_coords: pd.DataFrame,
    centroids_df: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "pca_coordinates": out_dir / "pca_2d_coordinates.csv",
        "tsne_coordinates": out_dir / "tsne_2d_coordinates.csv",
        "visual_centroids": out_dir / "cluster_visual_centroids.csv",
        "diagnostics": out_dir / "visualization_diagnostics.json",
        "pca_plot": figures_dir / "pca_cluster_scatterplot.png",
        "tsne_plot": figures_dir / "tsne_cluster_scatterplot.png",
    }

    pca_coords.to_csv(paths["pca_coordinates"], index=False)
    tsne_coords.to_csv(paths["tsne_coordinates"], index=False)
    centroids_df.to_csv(paths["visual_centroids"], index=False)
    paths["diagnostics"].write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    scatter_plot_by_cluster(
        coords_df=pca_coords,
        x_col="pca_1",
        y_col="pca_2",
        title="PCA 2D Cluster Scatterplot (Sampled Recipes)",
        out_path=paths["pca_plot"],
    )
    scatter_plot_by_cluster(
        coords_df=tsne_coords,
        x_col="tsne_1",
        y_col="tsne_2",
        title="t-SNE 2D Cluster Scatterplot (Sampled Recipes)",
        out_path=paths["tsne_plot"],
    )
    return paths


def print_final_summary(
    sample_size_used: int,
    pca_variance_ratio: list[float],
    pca_total_variance: float,
    tsne_kl_divergence: float,
    output_paths: dict[str, Path],
) -> None:
    print("Cluster visualization export complete.")
    print(f"Sample size used: {sample_size_used}")
    print(
        "PCA explained variance ratio (2D): "
        f"[{pca_variance_ratio[0]:.6f}, {pca_variance_ratio[1]:.6f}]"
    )
    print(f"PCA cumulative explained variance (2D): {pca_total_variance:.6f}")
    print(f"t-SNE KL divergence: {tsne_kl_divergence:.6f}")
    print("Output artifact paths:")
    print(f"- {output_paths['pca_coordinates']}")
    print(f"- {output_paths['tsne_coordinates']}")
    print(f"- {output_paths['visual_centroids']}")
    print(f"- {output_paths['diagnostics']}")
    print("Figure paths:")
    print(f"- {output_paths['pca_plot']}")
    print(f"- {output_paths['tsne_plot']}")


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    inputs = load_inputs(args)
    validate_inputs(inputs)

    X_reduced: np.ndarray = inputs["X_reduced"]
    assignments: pd.DataFrame = inputs["assignments"]
    feature_names: pd.DataFrame = inputs["feature_names"]

    sample_idx = sample_row_indices(
        assignments=assignments,
        n_rows=X_reduced.shape[0],
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    X_sample = X_reduced[sample_idx]
    sample_meta = build_sample_frame(assignments, sample_idx)

    pca_start = time.perf_counter()
    X_pca_2d, pca_model = run_pca_2d(X_sample, random_state=args.random_state)
    pca_runtime_seconds = float(time.perf_counter() - pca_start)

    tsne_start = time.perf_counter()
    X_tsne_2d, tsne_model = run_tsne_2d(
        X_sample=X_sample,
        random_state=args.random_state,
        perplexity=args.tsne_perplexity,
    )
    tsne_runtime_seconds = float(time.perf_counter() - tsne_start)

    pca_coords = build_coordinates_df(sample_meta, X_pca_2d, "pca_1", "pca_2")
    tsne_coords = build_coordinates_df(sample_meta, X_tsne_2d, "tsne_1", "tsne_2")

    feature_labels = get_feature_labels(feature_names, X_reduced.shape[1])
    centroids_df = compute_visual_centroids(
        X_sample=X_sample,
        X_pca_2d=X_pca_2d,
        X_tsne_2d=X_tsne_2d,
        sample_meta=sample_meta,
        feature_labels=feature_labels,
    )

    pca_var_ratio = pca_model.explained_variance_ratio_.astype(float).tolist()
    pca_var_sum = float(np.sum(pca_model.explained_variance_ratio_))
    tsne_kl = float(tsne_model.kl_divergence_)
    total_runtime_seconds = float(time.perf_counter() - total_start)

    diagnostics = {
        "selected_k_inferred_from_assignment_filename": infer_selected_k_from_filename(
            inputs["assignments_path"]
        ),
        "input_paths": {
            "matrix": str(inputs["matrix_path"]),
            "assignments": str(inputs["assignments_path"]),
            "feature_names": str(inputs["feature_names_path"]),
        },
        "input_shape": list(X_reduced.shape),
        "sample_size_requested": int(args.sample_size),
        "sample_size_used": int(len(sample_idx)),
        "sampling_random_state": int(args.random_state),
        "pca_2d_explained_variance_ratio": pca_var_ratio,
        "pca_2d_cumulative_explained_variance_ratio": pca_var_sum,
        "tsne_kl_divergence": tsne_kl,
        "tsne_perplexity": float(args.tsne_perplexity),
        "runtime_seconds": {
            "pca_projection": pca_runtime_seconds,
            "tsne_projection": tsne_runtime_seconds,
            "total": total_runtime_seconds,
        },
        "notes": [
            "PCA projection here is for 2D visualization and is separate from representation PCA.",
            "t-SNE is visualization-only and should not be used as a reusable embedding.",
            "t-SNE preserves local neighborhoods better than global geometry.",
            "Sampling is used to control runtime and memory on very large datasets.",
            "Sampling also guarantees at least one row per cluster for visualization coverage.",
        ],
    }

    paths = save_outputs(
        out_dir=args.out,
        figures_dir=args.figures_out,
        pca_coords=pca_coords,
        tsne_coords=tsne_coords,
        centroids_df=centroids_df,
        diagnostics=diagnostics,
    )
    print_final_summary(
        sample_size_used=len(sample_idx),
        pca_variance_ratio=pca_var_ratio,
        pca_total_variance=pca_var_sum,
        tsne_kl_divergence=tsne_kl,
        output_paths=paths,
    )


if __name__ == "__main__":
    main()
