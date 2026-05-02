#!/usr/bin/env python3
"""
Apply Week 5 dimensionality reduction to existing numeric and content matrices.

This script is Step 5 only. It does not rebuild features, cluster recipes,
create recommendations, or run t-SNE. PCA is used for the dense, scaled numeric
matrix, while TruncatedSVD is used for the high-dimensional sparse TF-IDF
content matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error


MPL_CACHE_DIR = Path("/private/tmp/recipe_recommendation_matplotlib")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANDOM_STATE = 42
PCA_TARGET_VARIANCE = 0.90
SVD_CANDIDATES = [50, 100, 200, 300]
DEFAULT_SELECTED_SVD_COMPONENTS = 200
SVD_RECONSTRUCTION_MAX_ROWS = 20_000
SVD_RECONSTRUCTION_MAX_DENSE_ENTRIES = 25_000_000


def resolve_existing_path(path: Path, fallback: Path | None = None) -> Path:
    if path.exists():
        return path
    if fallback is not None and fallback.exists():
        print(f"WARNING: input path does not exist: {path}. Using fallback: {fallback}")
        return fallback
    raise FileNotFoundError(f"Input path does not exist: {path}")


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    numeric_matrix_path = resolve_existing_path(
        args.numeric_matrix,
        Path("artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy"),
    )
    numeric_feature_names_path = resolve_existing_path(
        args.numeric_feature_names,
        Path("artifacts/week5/numeric_matrix_outputs/numeric_feature_names.csv"),
    )
    numeric_recipe_ids_path = resolve_existing_path(
        args.numeric_recipe_ids,
        Path("artifacts/week5/numeric_matrix_outputs/recipe_ids.csv"),
    )

    X_numeric = np.load(numeric_matrix_path)
    numeric_features = pd.read_csv(numeric_feature_names_path)
    numeric_ids = pd.read_csv(numeric_recipe_ids_path)

    X_content = load_npz(args.content_matrix).tocsr()
    content_features = pd.read_csv(args.content_feature_names)
    content_ids = pd.read_csv(args.content_recipe_ids)

    return {
        "numeric_matrix_path": numeric_matrix_path,
        "numeric_feature_names_path": numeric_feature_names_path,
        "numeric_recipe_ids_path": numeric_recipe_ids_path,
        "content_matrix_path": args.content_matrix,
        "content_feature_names_path": args.content_feature_names,
        "content_recipe_ids_path": args.content_recipe_ids,
        "X_numeric": X_numeric,
        "numeric_features": numeric_features,
        "numeric_ids": numeric_ids,
        "X_content": X_content,
        "content_features": content_features,
        "content_ids": content_ids,
    }


def validate_required_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def numeric_feature_labels(numeric_features: pd.DataFrame) -> list[str]:
    if "original_feature" in numeric_features.columns:
        return numeric_features["original_feature"].astype(str).tolist()
    if "feature" in numeric_features.columns:
        return numeric_features["feature"].astype(str).tolist()
    raise ValueError("Numeric feature names CSV needs original_feature or feature column.")


def validate_inputs(inputs: dict[str, Any]) -> None:
    X_numeric = inputs["X_numeric"]
    X_content = inputs["X_content"]
    numeric_features = inputs["numeric_features"]
    content_features = inputs["content_features"]
    numeric_ids = inputs["numeric_ids"]
    content_ids = inputs["content_ids"]

    if not isinstance(X_numeric, np.ndarray) or X_numeric.ndim != 2:
        raise ValueError("Numeric matrix must be a dense 2D numpy array.")
    if not sparse.issparse(X_content):
        raise ValueError("Content matrix must be scipy sparse.")
    if X_numeric.shape[0] != X_content.shape[0]:
        raise ValueError("Numeric and content matrices have different row counts.")

    validate_required_columns(numeric_ids, ["row_index", "RecipeId"], "numeric recipe IDs")
    validate_required_columns(content_ids, ["row_index", "RecipeId"], "content recipe IDs")
    if not numeric_ids["RecipeId"].reset_index(drop=True).equals(
        content_ids["RecipeId"].reset_index(drop=True)
    ):
        raise ValueError("Numeric and content RecipeId order does not match.")

    if len(numeric_feature_labels(numeric_features)) != X_numeric.shape[1]:
        raise ValueError("Numeric feature count does not match numeric matrix columns.")
    if len(content_features) != X_content.shape[1]:
        raise ValueError("Content feature count does not match content matrix columns.")
    if "feature_name" not in content_features.columns:
        raise ValueError("Content feature names CSV must include feature_name.")

    if not np.isfinite(X_numeric).all():
        raise ValueError("Numeric matrix contains NaN or infinite values.")
    if not np.isfinite(X_content.data).all():
        raise ValueError("Content matrix data contains NaN or infinite values.")


def save_recipe_ids(numeric_ids: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "reduced_recipe_ids.csv"
    numeric_ids[["row_index", "RecipeId"]].to_csv(path, index=False)
    return path


def run_pca(
    X_numeric: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    # PCA is appropriate for dense, scaled numeric features because these
    # variables have meaningful covariance structure after preprocessing.
    pca = PCA(random_state=RANDOM_STATE)
    X_all = pca.fit_transform(X_numeric)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    selected_k = int(np.searchsorted(cumulative, PCA_TARGET_VARIANCE) + 1)
    achieved = float(cumulative[selected_k - 1])

    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    explained = pd.DataFrame(
        {
            "component": components,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )

    loading_rows = []
    top_rows = []
    for comp_idx, ratio in enumerate(pca.explained_variance_ratio_, start=1):
        loadings = pca.components_[comp_idx - 1]
        for feature, loading in zip(feature_names, loadings):
            loading_rows.append(
                {
                    "component": comp_idx,
                    "feature": feature,
                    "loading": float(loading),
                    "abs_loading": float(abs(loading)),
                    "explained_variance_ratio": float(ratio),
                }
            )

        order = np.argsort(np.abs(loadings))[::-1]
        for rank, feature_idx in enumerate(order[: min(10, len(order))], start=1):
            feature = feature_names[feature_idx]
            loading = float(loadings[feature_idx])
            top_rows.append(
                {
                    "component": comp_idx,
                    "rank": rank,
                    "feature": feature,
                    "loading": loading,
                    "abs_loading": abs(loading),
                    "interpretation_hint": (
                        f"Component {comp_idx} is strongly associated with {feature}."
                    ),
                }
            )

    reconstruction_rows = []
    for k in range(1, X_numeric.shape[1] + 1):
        X_reduced_k = X_all[:, :k]
        X_hat_k = X_reduced_k @ pca.components_[:k, :] + pca.mean_
        mse = float(mean_squared_error(X_numeric, X_hat_k))
        reconstruction_rows.append(
            {
                "n_components": k,
                "cumulative_explained_variance_ratio": float(cumulative[k - 1]),
                "reconstruction_mse": mse,
                "reconstruction_rmse": float(np.sqrt(mse)),
            }
        )

    X_numeric_pca = X_all[:, :selected_k].astype(np.float32)
    X_numeric_pca_2d = X_all[:, :2].astype(np.float32)

    paths = {
        "model": output_dir / "pca_model_full.joblib",
        "explained": output_dir / "pca_explained_variance.csv",
        "loadings": output_dir / "pca_component_loadings.csv",
        "top_loadings": output_dir / "pca_top_loadings.csv",
        "reconstruction": output_dir / "pca_reconstruction_metrics.csv",
        "matrix": output_dir / "X_numeric_pca.npy",
        "matrix_2d": output_dir / "X_numeric_pca_2d.npy",
        "config": output_dir / "pca_selected_config.json",
        "plot_cumulative": figures_dir / "pca_cumulative_variance.png",
        "plot_scree": figures_dir / "pca_scree_plot.png",
        "plot_reconstruction": figures_dir / "pca_reconstruction_error.png",
    }

    joblib.dump(pca, paths["model"])
    explained.to_csv(paths["explained"], index=False)
    pd.DataFrame(loading_rows).to_csv(paths["loadings"], index=False)
    pd.DataFrame(top_rows).to_csv(paths["top_loadings"], index=False)
    pd.DataFrame(reconstruction_rows).to_csv(paths["reconstruction"], index=False)
    np.save(paths["matrix"], X_numeric_pca)
    np.save(paths["matrix_2d"], X_numeric_pca_2d)

    config = {
        "selected_pca_components": selected_k,
        "target_variance_threshold": PCA_TARGET_VARIANCE,
        "achieved_cumulative_variance": achieved,
        "numeric_input_shape": list(X_numeric.shape),
        "numeric_output_shape": list(X_numeric_pca.shape),
        "pca_method": "PCA on scaled dense numeric matrix",
        "reason": "Numeric features are dense, scaled, and correlated, making PCA appropriate.",
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 5))
    plt.plot(components, cumulative, marker="o")
    plt.axhline(PCA_TARGET_VARIANCE, linestyle="--")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA cumulative explained variance")
    plt.tight_layout()
    plt.savefig(paths["plot_cumulative"], dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(components, pca.explained_variance_ratio_, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA scree plot")
    plt.tight_layout()
    plt.savefig(paths["plot_scree"], dpi=150)
    plt.close()

    recon = pd.DataFrame(reconstruction_rows)
    plt.figure(figsize=(8, 5))
    plt.plot(recon["n_components"], recon["reconstruction_rmse"], marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Reconstruction RMSE")
    plt.title("PCA reconstruction error")
    plt.tight_layout()
    plt.savefig(paths["plot_reconstruction"], dpi=150)
    plt.close()

    return {
        "pca": pca,
        "X_numeric_pca": X_numeric_pca,
        "X_numeric_pca_2d": X_numeric_pca_2d,
        "selected_k": selected_k,
        "achieved_variance": achieved,
        "explained": explained,
        "reconstruction": pd.DataFrame(reconstruction_rows),
        "paths": paths,
    }


def valid_svd_candidates(n_features: int) -> list[int]:
    if n_features < 2:
        raise ValueError("TruncatedSVD requires at least 2 content features.")
    valid = [k for k in SVD_CANDIDATES if k < n_features]
    if valid:
        return valid
    fallback = max(1, min(10, n_features - 1))
    return [fallback]


def deterministic_sample_indices(n_rows: int, n_cols: int) -> tuple[np.ndarray, bool, str]:
    max_rows_by_memory = max(1, SVD_RECONSTRUCTION_MAX_DENSE_ENTRIES // max(n_cols, 1))
    sample_size = min(n_rows, SVD_RECONSTRUCTION_MAX_ROWS, max_rows_by_memory)
    if sample_size <= 0:
        return np.array([], dtype=np.int64), False, "Reconstruction skipped because sample size is zero."

    rng = np.random.default_rng(RANDOM_STATE)
    if n_rows <= sample_size:
        indices = np.arange(n_rows, dtype=np.int64)
    else:
        indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))

    note = (
        f"Reconstruction evaluated on {sample_size} sampled rows; dense sample "
        f"limit is {SVD_RECONSTRUCTION_MAX_DENSE_ENTRIES} entries."
    )
    return indices, True, note


def run_svd(
    X_content: sparse.csr_matrix,
    content_features: pd.DataFrame,
    output_dir: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    # TruncatedSVD is appropriate for high-dimensional sparse TF-IDF content
    # features because dense PCA would be computationally inefficient and would
    # require densifying the full sparse matrix.
    candidates = valid_svd_candidates(X_content.shape[1])
    selected_k = (
        DEFAULT_SELECTED_SVD_COMPONENTS
        if DEFAULT_SELECTED_SVD_COMPONENTS in candidates
        else max(candidates)
    )

    sample_indices, can_reconstruct, reconstruction_note = deterministic_sample_indices(
        X_content.shape[0], X_content.shape[1]
    )
    X_sample_sparse = X_content[sample_indices] if can_reconstruct else None
    X_sample_dense = (
        X_sample_sparse.toarray().astype(np.float32) if X_sample_sparse is not None else None
    )

    sweep_rows = []
    models: dict[int, TruncatedSVD] = {}
    selected_matrix = None
    selected_model = None

    for k in candidates:
        start = time.perf_counter()
        svd = TruncatedSVD(n_components=k, random_state=RANDOM_STATE)
        Z = svd.fit_transform(X_content).astype(np.float32)
        runtime = time.perf_counter() - start
        retained_energy = float(svd.explained_variance_ratio_.sum())

        mse = np.nan
        rmse = np.nan
        if can_reconstruct and X_sample_dense is not None:
            Z_sample = svd.transform(X_sample_sparse)
            X_hat_sample = svd.inverse_transform(Z_sample)
            mse = float(mean_squared_error(X_sample_dense, X_hat_sample))
            rmse = float(np.sqrt(mse))

        sweep_rows.append(
            {
                "n_components": k,
                "retained_energy": retained_energy,
                "reconstruction_mse_sample": mse,
                "reconstruction_rmse_sample": rmse,
                "input_shape_rows": X_content.shape[0],
                "input_shape_cols": X_content.shape[1],
                "output_shape_rows": Z.shape[0],
                "output_shape_cols": Z.shape[1],
                "random_state": RANDOM_STATE,
                "runtime_seconds": round(runtime, 4),
            }
        )
        models[k] = svd
        joblib.dump(svd, output_dir / f"svd_model_k{k}.joblib")

        if k == selected_k:
            selected_matrix = Z
            selected_model = svd

    if selected_matrix is None or selected_model is None:
        raise ValueError("Selected SVD model was not fitted.")

    joblib.dump(selected_model, output_dir / "svd_model_selected.joblib")
    np.save(output_dir / "X_content_svd.npy", selected_matrix.astype(np.float32))

    cumulative = np.cumsum(selected_model.explained_variance_ratio_)
    component_numbers = np.arange(1, selected_k + 1)
    explained = pd.DataFrame(
        {
            "component": component_numbers,
            "explained_variance": selected_model.explained_variance_,
            "explained_variance_ratio": selected_model.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )
    explained_path = output_dir / "svd_explained_variance_selected.csv"
    explained.to_csv(explained_path, index=False)

    top_terms = build_svd_top_terms(selected_model, content_features)
    top_terms_path = output_dir / "svd_top_terms.csv"
    top_terms.to_csv(top_terms_path, index=False)

    sweep = pd.DataFrame(sweep_rows)
    sweep_path = output_dir / "svd_component_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    selected_retained = float(selected_model.explained_variance_ratio_.sum())
    config = {
        "selected_svd_components": selected_k,
        "candidate_components": SVD_CANDIDATES,
        "valid_candidate_components": candidates,
        "selected_retained_energy": selected_retained,
        "content_input_shape": list(X_content.shape),
        "content_output_shape": list(selected_matrix.shape),
        "svd_method": "TruncatedSVD on sparse TF-IDF content matrix",
        "reason": (
            "Content features are high-dimensional and sparse, making TruncatedSVD "
            "appropriate instead of dense PCA."
        ),
        "random_state": RANDOM_STATE,
        "reconstruction_note": reconstruction_note,
        "reconstruction_sample_size": int(len(sample_indices)),
    }
    config_path = output_dir / "svd_selected_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    plot_paths = {
        "retained_energy": figures_dir / "svd_retained_energy_sweep.png",
        "selected_cumulative": figures_dir / "svd_selected_cumulative_energy.png",
        "reconstruction": figures_dir / "svd_reconstruction_error_sweep.png",
    }
    plt.figure(figsize=(8, 5))
    plt.plot(sweep["n_components"], sweep["retained_energy"], marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Retained energy")
    plt.title("TruncatedSVD retained energy sweep")
    plt.tight_layout()
    plt.savefig(plot_paths["retained_energy"], dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(component_numbers, cumulative)
    plt.xlabel("Component")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("Selected TruncatedSVD cumulative energy")
    plt.tight_layout()
    plt.savefig(plot_paths["selected_cumulative"], dpi=150)
    plt.close()

    recon_available = sweep["reconstruction_rmse_sample"].notna().any()
    if recon_available:
        plt.figure(figsize=(8, 5))
        plt.plot(sweep["n_components"], sweep["reconstruction_rmse_sample"], marker="o")
        plt.xlabel("Number of components")
        plt.ylabel("Sample reconstruction RMSE")
        plt.title("TruncatedSVD sampled reconstruction error")
        plt.tight_layout()
        plt.savefig(plot_paths["reconstruction"], dpi=150)
        plt.close()
    else:
        print("WARNING: SVD reconstruction error plot skipped because reconstruction was not computed.")

    return {
        "X_content_svd": selected_matrix.astype(np.float32),
        "selected_model": selected_model,
        "selected_k": selected_k,
        "selected_retained_energy": selected_retained,
        "sweep": sweep,
        "explained": explained,
        "paths": {
            "matrix": output_dir / "X_content_svd.npy",
            "selected_model": output_dir / "svd_model_selected.joblib",
            "sweep": sweep_path,
            "explained": explained_path,
            "top_terms": top_terms_path,
            "config": config_path,
            **{f"plot_{key}": value for key, value in plot_paths.items()},
        },
        "reconstruction_note": reconstruction_note,
    }


def build_svd_top_terms(
    svd: TruncatedSVD, content_features: pd.DataFrame, top_n: int = 15
) -> pd.DataFrame:
    rows = []
    feature_names = content_features["feature_name"].astype(str).to_numpy()
    feature_groups = content_features["feature_group"].astype(str).to_numpy()

    for comp_idx, loadings in enumerate(svd.components_, start=1):
        positive_order = np.argsort(loadings)[::-1][:top_n]
        negative_order = np.argsort(loadings)[:top_n]
        for side, order in [("positive", positive_order), ("negative", negative_order)]:
            for rank, feature_idx in enumerate(order, start=1):
                loading = float(loadings[feature_idx])
                rows.append(
                    {
                        "component": comp_idx,
                        "side": side,
                        "rank": rank,
                        "feature_name": feature_names[feature_idx],
                        "feature_group": feature_groups[feature_idx],
                        "loading": loading,
                        "abs_loading": abs(loading),
                    }
                )
    return pd.DataFrame(rows)


def build_combined_representation(
    X_content_svd: np.ndarray,
    X_numeric_pca: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    # The combined reduced matrix is a reusable recipe embedding for Week 7
    # clustering and Week 10 content-based recommendation experiments.
    X_recipe_reduced = np.hstack([X_content_svd, X_numeric_pca]).astype(np.float32)
    rows = []
    feature_index = 0
    for idx in range(X_content_svd.shape[1]):
        rows.append(
            {
                "feature_index": feature_index,
                "feature_name": f"content_svd_{idx + 1:03d}",
                "feature_group": "content_svd",
                "source_representation": "X_content_svd",
            }
        )
        feature_index += 1
    for idx in range(X_numeric_pca.shape[1]):
        rows.append(
            {
                "feature_index": feature_index,
                "feature_name": f"numeric_pca_{idx + 1:03d}",
                "feature_group": "numeric_pca",
                "source_representation": "X_numeric_pca",
            }
        )
        feature_index += 1

    feature_names = pd.DataFrame(rows)
    matrix_path = output_dir / "X_recipe_reduced.npy"
    features_path = output_dir / "reduced_feature_names.csv"
    np.save(matrix_path, X_recipe_reduced)
    feature_names.to_csv(features_path, index=False)
    return {
        "X_recipe_reduced": X_recipe_reduced,
        "feature_names": feature_names,
        "matrix_path": matrix_path,
        "features_path": features_path,
    }


def validate_reduced_outputs(
    n_recipes: int,
    X_numeric_pca: np.ndarray,
    X_content_svd: np.ndarray,
    X_recipe_reduced: np.ndarray,
    reduced_feature_names: pd.DataFrame,
    selected_pca_components: int,
    selected_svd_components: int,
    pca_achieved_variance: float,
) -> None:
    if selected_pca_components <= 0:
        raise ValueError("Selected PCA components must be > 0.")
    if selected_svd_components <= 0:
        raise ValueError("Selected SVD components must be > 0.")
    if pca_achieved_variance < PCA_TARGET_VARIANCE:
        raise ValueError("Selected PCA components did not reach target variance.")

    for name, matrix in [
        ("X_numeric_pca", X_numeric_pca),
        ("X_content_svd", X_content_svd),
        ("X_recipe_reduced", X_recipe_reduced),
    ]:
        if matrix.shape[0] != n_recipes:
            raise ValueError(f"{name} row count does not match n_recipes.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinite values.")

    if len(reduced_feature_names) != X_recipe_reduced.shape[1]:
        raise ValueError("reduced_feature_names rows do not match combined matrix columns.")


def estimate_mb(matrix: np.ndarray) -> float:
    return float(matrix.nbytes / (1024**2))


def build_dimensionality_comparison(
    X_numeric: np.ndarray,
    X_content: sparse.csr_matrix,
    X_numeric_pca: np.ndarray,
    X_content_svd: np.ndarray,
    X_recipe_reduced: np.ndarray,
    pca_result: dict[str, Any],
    svd_result: dict[str, Any],
) -> pd.DataFrame:
    pca_recon = pca_result["reconstruction"]
    pca_selected_recon = pca_recon.loc[
        pca_recon["n_components"].eq(pca_result["selected_k"])
    ].iloc[0]
    svd_sweep = svd_result["sweep"]
    svd_selected_recon = svd_sweep.loc[
        svd_sweep["n_components"].eq(svd_result["selected_k"])
    ].iloc[0]

    rows = [
        {
            "representation": "numeric_scaled",
            "input_shape": str(tuple(X_numeric.shape)),
            "output_shape": str(tuple(X_numeric.shape)),
            "original_dimensions": X_numeric.shape[1],
            "reduced_dimensions": X_numeric.shape[1],
            "method": "StandardScaler output",
            "sparse_or_dense": "dense",
            "preprocessing_source": "Step 3 numeric matrix",
            "selected_components": "",
            "variance_or_energy_retained": "",
            "reconstruction_mse": "",
            "reconstruction_rmse": "",
            "intended_downstream_use": "Input to numeric PCA.",
            "notes": "Dense scaled numeric matrix after clipping/log1p/standardization.",
        },
        {
            "representation": "numeric_pca_selected",
            "input_shape": str(tuple(X_numeric.shape)),
            "output_shape": str(tuple(X_numeric_pca.shape)),
            "original_dimensions": X_numeric.shape[1],
            "reduced_dimensions": X_numeric_pca.shape[1],
            "method": "PCA",
            "sparse_or_dense": "dense",
            "preprocessing_source": "Step 3 numeric matrix",
            "selected_components": pca_result["selected_k"],
            "variance_or_energy_retained": pca_result["achieved_variance"],
            "reconstruction_mse": float(pca_selected_recon["reconstruction_mse"]),
            "reconstruction_rmse": float(pca_selected_recon["reconstruction_rmse"]),
            "intended_downstream_use": "Compact numeric recipe structure.",
            "notes": "PCA-selected numeric representation using 90% cumulative explained variance threshold.",
        },
        {
            "representation": "content_tfidf",
            "input_shape": str(tuple(X_content.shape)),
            "output_shape": str(tuple(X_content.shape)),
            "original_dimensions": X_content.shape[1],
            "reduced_dimensions": X_content.shape[1],
            "method": "TF-IDF sparse matrix",
            "sparse_or_dense": "sparse",
            "preprocessing_source": "Step 4 content TF-IDF matrix",
            "selected_components": "",
            "variance_or_energy_retained": "",
            "reconstruction_mse": "",
            "reconstruction_rmse": "",
            "intended_downstream_use": "Input to content SVD.",
            "notes": "Sparse TF-IDF matrix from ingredients, keywords, category, and optional yield tokens.",
        },
        {
            "representation": "content_svd_selected",
            "input_shape": str(tuple(X_content.shape)),
            "output_shape": str(tuple(X_content_svd.shape)),
            "original_dimensions": X_content.shape[1],
            "reduced_dimensions": X_content_svd.shape[1],
            "method": "TruncatedSVD",
            "sparse_or_dense": "dense",
            "preprocessing_source": "Step 4 content TF-IDF matrix",
            "selected_components": svd_result["selected_k"],
            "variance_or_energy_retained": svd_result["selected_retained_energy"],
            "reconstruction_mse": float(svd_selected_recon["reconstruction_mse_sample"]),
            "reconstruction_rmse": float(svd_selected_recon["reconstruction_rmse_sample"]),
            "intended_downstream_use": "Dense content embedding.",
            "notes": "TruncatedSVD dense content embedding; retained energy is expected to be lower than numeric PCA because text variance is diffuse.",
        },
        {
            "representation": "combined_recipe_reduced",
            "input_shape": f"{tuple(X_content_svd.shape)} + {tuple(X_numeric_pca.shape)}",
            "output_shape": str(tuple(X_recipe_reduced.shape)),
            "original_dimensions": X_content.shape[1] + X_numeric.shape[1],
            "reduced_dimensions": X_recipe_reduced.shape[1],
            "method": "Concatenation",
            "sparse_or_dense": "dense",
            "preprocessing_source": "Step 5 PCA/SVD outputs",
            "selected_components": X_recipe_reduced.shape[1],
            "variance_or_energy_retained": "",
            "reconstruction_mse": "",
            "reconstruction_rmse": "",
            "intended_downstream_use": "Future clustering and content-based recommendation.",
            "notes": "Concatenated content SVD and numeric PCA representation for clustering and content-based recommendation.",
        },
    ]
    return pd.DataFrame(rows)


def save_pipeline_config(
    inputs: dict[str, Any],
    output_dir: Path,
    figures_dir: Path,
    pca_result: dict[str, Any],
    svd_result: dict[str, Any],
    combined: dict[str, Any],
    comparison_path: Path,
    recipe_ids_path: Path,
) -> Path:
    X_numeric = inputs["X_numeric"]
    X_content = inputs["X_content"]
    X_recipe_reduced = combined["X_recipe_reduced"]

    pca_artifacts = [
        value
        for key, value in pca_result["paths"].items()
        if not key.startswith("plot_")
    ]
    svd_artifacts = [
        value
        for key, value in svd_result["paths"].items()
        if not key.startswith("plot_")
    ]
    svd_model_artifacts = sorted(output_dir.glob("svd_model_k*.joblib"))
    artifacts_created = [
        str(path)
        for path in [
            recipe_ids_path,
            *pca_artifacts,
            *svd_artifacts,
            *svd_model_artifacts,
            combined["matrix_path"],
            combined["features_path"],
            comparison_path,
        ]
    ]
    figures_created = [
        str(pca_result["paths"]["plot_cumulative"]),
        str(pca_result["paths"]["plot_scree"]),
        str(pca_result["paths"]["plot_reconstruction"]),
        str(svd_result["paths"]["plot_retained_energy"]),
        str(svd_result["paths"]["plot_selected_cumulative"]),
    ]
    if Path(svd_result["paths"]["plot_reconstruction"]).exists():
        figures_created.append(str(svd_result["paths"]["plot_reconstruction"]))

    config = {
        "input_paths": {
            "numeric_matrix": str(inputs["numeric_matrix_path"]),
            "numeric_feature_names": str(inputs["numeric_feature_names_path"]),
            "numeric_recipe_ids": str(inputs["numeric_recipe_ids_path"]),
            "content_matrix": str(inputs["content_matrix_path"]),
            "content_feature_names": str(inputs["content_feature_names_path"]),
            "content_recipe_ids": str(inputs["content_recipe_ids_path"]),
        },
        "output_directory": str(output_dir),
        "figures_directory": str(figures_dir),
        "n_recipes": int(X_numeric.shape[0]),
        "numeric_input_shape": list(X_numeric.shape),
        "content_input_shape": list(X_content.shape),
        "numeric_pca_selected_components": int(pca_result["selected_k"]),
        "pca_achieved_variance": float(pca_result["achieved_variance"]),
        "svd_candidate_components": SVD_CANDIDATES,
        "svd_selected_components": int(svd_result["selected_k"]),
        "svd_retained_energy": float(svd_result["selected_retained_energy"]),
        "combined_reduced_shape": list(X_recipe_reduced.shape),
        "random_state": RANDOM_STATE,
        "matrix_memory_estimates_mb": {
            "X_numeric_pca": estimate_mb(pca_result["X_numeric_pca"]),
            "X_content_svd": estimate_mb(svd_result["X_content_svd"]),
            "X_recipe_reduced": estimate_mb(X_recipe_reduced),
        },
        "artifacts_created": artifacts_created,
        "figures_created": figures_created,
        "methodological_notes": [
            "PCA used for dense scaled numeric matrix.",
            "TruncatedSVD used for sparse TF-IDF matrix.",
            "t-SNE intentionally not run here because it is for visualization only, not a reusable representation.",
            "Ratings and popularity variables were excluded to avoid leakage.",
            "X_recipe_reduced can be reused in Week 7 clustering and Week 10 content-based recommendation.",
        ],
    }
    path = output_dir / "reduction_pipeline_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply PCA and TruncatedSVD to existing Week 5 matrices."
    )
    parser.add_argument("--numeric-matrix", required=True, type=Path)
    parser.add_argument("--numeric-feature-names", required=True, type=Path)
    parser.add_argument("--numeric-recipe-ids", required=True, type=Path)
    parser.add_argument("--content-matrix", required=True, type=Path)
    parser.add_argument("--content-feature-names", required=True, type=Path)
    parser.add_argument("--content-recipe-ids", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.out
    figures_dir = args.figures
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(args)
    validate_inputs(inputs)

    recipe_ids_path = save_recipe_ids(inputs["numeric_ids"], output_dir)

    pca_result = run_pca(
        inputs["X_numeric"],
        numeric_feature_labels(inputs["numeric_features"]),
        output_dir,
        figures_dir,
    )
    svd_result = run_svd(
        inputs["X_content"],
        inputs["content_features"],
        output_dir,
        figures_dir,
    )
    combined = build_combined_representation(
        svd_result["X_content_svd"],
        pca_result["X_numeric_pca"],
        output_dir,
    )

    validate_reduced_outputs(
        n_recipes=inputs["X_numeric"].shape[0],
        X_numeric_pca=pca_result["X_numeric_pca"],
        X_content_svd=svd_result["X_content_svd"],
        X_recipe_reduced=combined["X_recipe_reduced"],
        reduced_feature_names=combined["feature_names"],
        selected_pca_components=pca_result["selected_k"],
        selected_svd_components=svd_result["selected_k"],
        pca_achieved_variance=pca_result["achieved_variance"],
    )

    if svd_result["selected_retained_energy"] < 0.20:
        print(
            "WARNING: selected SVD retained energy is low "
            f"({svd_result['selected_retained_energy']:.4f}); this is common for sparse text."
        )
    if estimate_mb(combined["X_recipe_reduced"]) > 1024:
        print(
            "WARNING: X_recipe_reduced is larger than 1 GB in memory: "
            f"{estimate_mb(combined['X_recipe_reduced']):.2f} MB."
        )

    comparison = build_dimensionality_comparison(
        inputs["X_numeric"],
        inputs["X_content"],
        pca_result["X_numeric_pca"],
        svd_result["X_content_svd"],
        combined["X_recipe_reduced"],
        pca_result,
        svd_result,
    )
    comparison_path = output_dir / "dimensionality_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    save_pipeline_config(
        inputs,
        output_dir,
        figures_dir,
        pca_result,
        svd_result,
        combined,
        comparison_path,
        recipe_ids_path,
    )

    print("Dimensionality reduction complete.")
    print(f"Numeric input shape: {inputs['X_numeric'].shape}")
    print(f"Content input shape: {inputs['X_content'].shape}")
    print(f"Selected PCA components: {pca_result['selected_k']}")
    print(f"PCA achieved cumulative variance: {pca_result['achieved_variance']:.6f}")
    print(f"Selected SVD components: {svd_result['selected_k']}")
    print(f"SVD retained energy: {svd_result['selected_retained_energy']:.6f}")
    print(f"Combined reduced matrix shape: {combined['X_recipe_reduced'].shape}")
    print(f"Artifact output directory: {output_dir}")
    print(f"Figure output directory: {figures_dir}")
    print(f"X_numeric_pca: {pca_result['paths']['matrix']}")
    print(f"X_content_svd: {svd_result['paths']['matrix']}")
    print(f"X_recipe_reduced: {combined['matrix_path']}")
    print(f"Dimensionality comparison: {comparison_path}")
    print("Generated plots:")
    print(f"- {pca_result['paths']['plot_cumulative']}")
    print(f"- {pca_result['paths']['plot_scree']}")
    print(f"- {pca_result['paths']['plot_reconstruction']}")
    print(f"- {svd_result['paths']['plot_retained_energy']}")
    print(f"- {svd_result['paths']['plot_selected_cumulative']}")
    if Path(svd_result["paths"]["plot_reconstruction"]).exists():
        print(f"- {svd_result['paths']['plot_reconstruction']}")


if __name__ == "__main__":
    main()
