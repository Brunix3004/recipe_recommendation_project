#!/usr/bin/env python3
"""
Build the canonical Week 7 clustering matrix from Week 5 reduced artifacts.

This script combines semantic and numeric latent embeddings that were already
produced in Week 5 (TruncatedSVD for content and PCA for numeric features). It
intentionally preserves semantic geometry by leaving the SVD representation
unscaled, then applies controlled weighting before concatenation.

The resulting matrix is the reusable clustering representation for Week 7 and
is intended to feed KMeans, cluster evaluation, and recommendation experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA


DEFAULT_CONTENT_SVD_PATH = Path("artifacts/week5/pca_svd/X_content_svd.npy")
DEFAULT_NUMERIC_PCA_PATH = Path("artifacts/week5/pca_svd/X_numeric_pca.npy")
DEFAULT_RECIPE_IDS_PATH = Path("artifacts/week5/pca_svd/reduced_recipe_ids.csv")
DEFAULT_REDUCED_FEATURE_NAMES_PATH = Path("artifacts/week5/pca_svd/reduced_feature_names.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/week7/clustering_matrix")

CONTENT_WEIGHT = 1.0
NUMERIC_WEIGHT = 1.0
N_CONTENT_KEPT = 50
N_JOINT_COMPONENTS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Week 7 clustering matrix from existing Week 5 PCA/SVD outputs "
            "(without rerunning preprocessing or dimensionality reduction)."
        )
    )
    parser.add_argument(
        "--content-svd",
        type=Path,
        default=DEFAULT_CONTENT_SVD_PATH,
        help=f"Path to X_content_svd.npy (default: {DEFAULT_CONTENT_SVD_PATH}).",
    )
    parser.add_argument(
        "--numeric-pca",
        type=Path,
        default=DEFAULT_NUMERIC_PCA_PATH,
        help=f"Path to X_numeric_pca.npy (default: {DEFAULT_NUMERIC_PCA_PATH}).",
    )
    parser.add_argument(
        "--recipe-ids",
        type=Path,
        default=DEFAULT_RECIPE_IDS_PATH,
        help=f"Path to reduced_recipe_ids.csv (default: {DEFAULT_RECIPE_IDS_PATH}).",
    )
    parser.add_argument(
        "--reduced-feature-names",
        type=Path,
        default=DEFAULT_REDUCED_FEATURE_NAMES_PATH,
        help=(
            "Path to reduced_feature_names.csv used for upstream consistency checks "
            f"(default: {DEFAULT_REDUCED_FEATURE_NAMES_PATH})."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def estimate_matrix_mb(matrix: np.ndarray) -> float:
    return float(matrix.nbytes / (1024**2))


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
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
    content_svd_path = resolve_existing_path(args.content_svd, "Content SVD matrix")
    numeric_pca_path = resolve_existing_path(args.numeric_pca, "Numeric PCA matrix")
    recipe_ids_path = resolve_existing_path(args.recipe_ids, "Reduced recipe IDs CSV")
    reduced_feature_names_path = resolve_existing_path(
        args.reduced_feature_names, "Reduced feature names CSV"
    )

    X_content_svd = np.load(content_svd_path)
    X_numeric_pca = np.load(numeric_pca_path)
    recipe_ids = pd.read_csv(recipe_ids_path)
    reduced_feature_names = pd.read_csv(reduced_feature_names_path)

    return {
        "content_svd_path": content_svd_path,
        "numeric_pca_path": numeric_pca_path,
        "recipe_ids_path": recipe_ids_path,
        "reduced_feature_names_path": reduced_feature_names_path,
        "X_content_svd": X_content_svd,
        "X_numeric_pca": X_numeric_pca,
        "recipe_ids": recipe_ids,
        "reduced_feature_names": reduced_feature_names,
    }


def validate_inputs(inputs: dict[str, Any]) -> None:
    X_content_svd: np.ndarray = inputs["X_content_svd"]
    X_numeric_pca: np.ndarray = inputs["X_numeric_pca"]
    recipe_ids: pd.DataFrame = inputs["recipe_ids"]
    reduced_feature_names: pd.DataFrame = inputs["reduced_feature_names"]

    validate_matrix_2d_finite(X_content_svd, "X_content_svd")
    validate_matrix_2d_finite(X_numeric_pca, "X_numeric_pca")

    if X_content_svd.shape[0] != X_numeric_pca.shape[0]:
        raise ValueError(
            "Row count mismatch between X_content_svd and X_numeric_pca: "
            f"{X_content_svd.shape[0]} != {X_numeric_pca.shape[0]}"
        )

    validate_required_columns(recipe_ids, ["row_index", "RecipeId"], "reduced_recipe_ids.csv")
    if len(recipe_ids) != X_content_svd.shape[0]:
        raise ValueError(
            "reduced_recipe_ids.csv row count does not match matrix rows: "
            f"{len(recipe_ids)} != {X_content_svd.shape[0]}"
        )
    if recipe_ids["RecipeId"].isna().any():
        raise ValueError("reduced_recipe_ids.csv contains null RecipeId values.")
    if recipe_ids["RecipeId"].duplicated().any():
        dup_count = int(recipe_ids["RecipeId"].duplicated().sum())
        raise ValueError(f"reduced_recipe_ids.csv contains duplicate RecipeId values: {dup_count}")

    expected_feature_count = X_content_svd.shape[1] + X_numeric_pca.shape[1]
    if len(reduced_feature_names) != expected_feature_count:
        raise ValueError(
            "reduced_feature_names.csv row count does not match reduced dimensions: "
            f"{len(reduced_feature_names)} != {expected_feature_count}"
        )


def build_clustering_matrix(
    X_content_svd: np.ndarray,
    X_numeric_pca: np.ndarray,
    n_content_kept: int = N_CONTENT_KEPT,
    n_joint_components: int = N_JOINT_COMPONENTS,
    content_weight: float = CONTENT_WEIGHT,
    numeric_weight: float = NUMERIC_WEIGHT,
) -> np.ndarray:
    X_content = X_content_svd[:, :n_content_kept].astype(np.float64, copy=True)
    X_numeric = X_numeric_pca.astype(np.float64, copy=True)

    X_content = StandardScaler().fit_transform(X_content)
    X_numeric = StandardScaler().fit_transform(X_numeric)

    X_content = normalize(X_content, norm="l2", axis=1)

    content_energy = np.linalg.norm(X_content, ord="fro")
    numeric_energy = np.linalg.norm(X_numeric, ord="fro")
    X_content *= (content_weight / content_energy)
    X_numeric *= (numeric_weight / numeric_energy)

    X_joint = np.hstack([X_content, X_numeric])
    X_joint = PCA(n_components=n_joint_components, whiten=True,
                  random_state=42).fit_transform(X_joint)

    return X_joint.astype(np.float32, copy=False)


def build_feature_metadata(n_joint_components: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(n_joint_components):
        rows.append(
            {
                "feature_index": i,
                "feature_name": f"latent_joint_pca_{i + 1:03d}",
                "feature_group": "joint_pca",
                "source_representation": "X_joint",
                "scaling_applied": True,
                "representation_weight": 1.0,
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    X_clustering: np.ndarray,
    feature_metadata: pd.DataFrame,
    recipe_ids: pd.DataFrame,
    expected_cols: int,
) -> None:
    validate_matrix_2d_finite(X_clustering, "X_clustering")

    if X_clustering.shape[1] != expected_cols:
        raise ValueError(
            "Final clustering matrix column count is inconsistent: "
            f"{X_clustering.shape[1]} != {expected_cols}"
        )

    if len(feature_metadata) != X_clustering.shape[1]:
        raise ValueError(
            "Feature metadata row count does not match final matrix columns: "
            f"{len(feature_metadata)} != {X_clustering.shape[1]}"
        )
    expected_index = np.arange(len(feature_metadata), dtype=np.int64)
    if not np.array_equal(feature_metadata["feature_index"].to_numpy(), expected_index):
        raise ValueError("feature_index must be contiguous from 0 to n_features - 1.")

    if len(recipe_ids) != X_clustering.shape[0]:
        raise ValueError(
            "clustering_recipe_ids row count does not match final matrix rows: "
            f"{len(recipe_ids)} != {X_clustering.shape[0]}"
        )


def build_config(
    inputs: dict[str, Any],
    X_clustering: np.ndarray,
    output_paths: dict[str, Path],
    n_content_kept: int,
    n_joint_components: int,
) -> dict[str, Any]:
    return {
        "input_paths": {
            "X_content_svd": str(inputs["content_svd_path"]),
            "X_numeric_pca": str(inputs["numeric_pca_path"]),
            "reduced_recipe_ids": str(inputs["recipe_ids_path"]),
            "reduced_feature_names": str(inputs["reduced_feature_names_path"]),
        },
        "matrix_shapes": {
            "X_content_svd": list(inputs["X_content_svd"].shape),
            "X_numeric_pca": list(inputs["X_numeric_pca"].shape),
            "X_recipe_clustering": list(X_clustering.shape),
        },
        "scaling_strategy": {
            "numeric_representation": "StandardScaler on X_numeric_pca",
            "content_representation": "StandardScaler and L2 normalization on truncated X_content_svd",
            "joint_representation": "PCA with whitening",
        },
        "representation_weights": {
            "CONTENT_WEIGHT": CONTENT_WEIGHT,
            "NUMERIC_WEIGHT": NUMERIC_WEIGHT,
        },
        "parameters": {
            "n_content_kept": n_content_kept,
            "n_joint_components": n_joint_components,
        },
        "memory_estimates_mb": {
            "X_content_svd": estimate_matrix_mb(inputs["X_content_svd"]),
            "X_numeric_pca": estimate_matrix_mb(inputs["X_numeric_pca"]),
            "X_recipe_clustering": estimate_matrix_mb(X_clustering),
        },
        "rationale": "Truncated SVD tail, normalized rows/columns, block-equalized Frobenius norm, joint PCA applied.",
        "random_state": 42,
        "artifact_paths": {name: str(path) for name, path in output_paths.items()},
    }


def save_outputs(
    output_paths: dict[str, Path],
    X_clustering: np.ndarray,
    recipe_ids: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    np.save(
        output_paths["X_recipe_clustering"],
        X_clustering.astype(np.float32, copy=False),
    )
    recipe_ids[["row_index", "RecipeId"]].to_csv(
        output_paths["clustering_recipe_ids"], index=False
    )
    feature_metadata.to_csv(output_paths["clustering_feature_names"], index=False)
    output_paths["clustering_matrix_config"].write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )


def build_output_paths(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "X_recipe_clustering": output_dir / "X_recipe_clustering.npy",
        "clustering_feature_names": output_dir / "clustering_feature_names.csv",
        "clustering_recipe_ids": output_dir / "clustering_recipe_ids.csv",
        "clustering_matrix_config": output_dir / "clustering_matrix_config.json",
    }


def print_summary(
    X_content_svd: np.ndarray,
    X_numeric_pca: np.ndarray,
    X_clustering: np.ndarray,
    output_paths: dict[str, Path],
) -> None:
    print("Week 7 clustering matrix build complete.")
    print(f"Content representation shape (X_content_svd): {X_content_svd.shape}")
    print(f"Numeric representation shape (X_numeric_pca): {X_numeric_pca.shape}")
    print(f"Final clustering matrix shape: {X_clustering.shape}")
    print(f"Applied weights -> content: {CONTENT_WEIGHT}, numeric: {NUMERIC_WEIGHT}")
    print(
        "Scaling strategy -> StandardScaler on both blocks, L2 normalization on content, "
        "Frobenius equalization, and joint PCA + whitening."
    )
    print("Output artifacts:")
    print(f"- X_recipe_clustering.npy: {output_paths['X_recipe_clustering']}")
    print(f"- clustering_feature_names.csv: {output_paths['clustering_feature_names']}")
    print(f"- clustering_recipe_ids.csv: {output_paths['clustering_recipe_ids']}")
    print(f"- clustering_matrix_config.json: {output_paths['clustering_matrix_config']}")
    print("Estimated memory usage (MB):")
    print(f"- X_content_svd: {estimate_matrix_mb(X_content_svd):.2f}")
    print(f"- X_numeric_pca: {estimate_matrix_mb(X_numeric_pca):.2f}")
    print(f"- X_recipe_clustering: {estimate_matrix_mb(X_clustering):.2f}")


def main() -> None:
    args = parse_args()
    output_dir = args.out

    inputs = load_inputs(args)
    validate_inputs(inputs)

    X_content_svd: np.ndarray = inputs["X_content_svd"]
    X_numeric_pca: np.ndarray = inputs["X_numeric_pca"]
    recipe_ids: pd.DataFrame = inputs["recipe_ids"]

    X_clustering = build_clustering_matrix(X_content_svd, X_numeric_pca)
    feature_metadata = build_feature_metadata(n_joint_components=X_clustering.shape[1])

    validate_outputs(
        X_clustering=X_clustering,
        feature_metadata=feature_metadata,
        recipe_ids=recipe_ids,
        expected_cols=N_JOINT_COMPONENTS,
    )

    output_paths = build_output_paths(output_dir)
    config = build_config(
        inputs=inputs,
        X_clustering=X_clustering,
        output_paths=output_paths,
        n_content_kept=N_CONTENT_KEPT,
        n_joint_components=N_JOINT_COMPONENTS,
    )
    save_outputs(
        output_paths=output_paths,
        X_clustering=X_clustering,
        recipe_ids=recipe_ids,
        feature_metadata=feature_metadata,
        config=config,
    )

    print_summary(
        X_content_svd=X_content_svd,
        X_numeric_pca=X_numeric_pca,
        X_clustering=X_clustering,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()
