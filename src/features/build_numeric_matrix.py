"""
Build the Week 5 PCA-ready numeric matrix for Food.com recipes.

This script performs numeric preprocessing only. It selects intrinsic recipe
features, handles residual invalid/missing values, clips selected upper-tail
outliers, applies log1p, and fits a StandardScaler. It does not run PCA, SVD,
TF-IDF, clustering, or recommendation/ranking logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


NUTRITION_COLS = [
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
]

TIME_COLS = [
    "CookTime_Minutes_imputed",
    "PrepTime_Minutes_imputed",
    "TotalTime_Minutes_imputed",
]

COUNT_COLS = [
    "NumIngredients",
    "NumQuantities",
]

SERVING_COLS = [
    "ResolvedServings_imputed",
]

NUMERIC_FEATURE_COLS = [
    *NUTRITION_COLS,
    *TIME_COLS,
    *COUNT_COLS,
    *SERVING_COLS,
]

EXCLUDED_LEAKAGE_COLUMNS = [
    "AggregatedRating",
    "ReviewCount",
    "Rating",
    "RecipeId",
    "AuthorId",
    "ReviewId",
]

EXCLUDED_INDICATOR_COLUMNS = [
    "servings_missing_original",
    "servings_from_yield",
    "servings_missing_after_yield_parse",
    "cooktime_missing_original",
    "cooktime_derived_from_total_prep",
    "preptime_missing_original",
    "preptime_derived_from_total_cook",
    "totaltime_missing_original",
    "totaltime_derived_from_prep_cook",
    "category_missing_original",
    "time_arithmetic_consistent",
    "time_arithmetic_mismatch",
]

CLIPPING_QUANTILE_BY_GROUP = {
    "nutrition": 0.995,
    "time": 0.995,
    "count": None,
    "serving": 0.995,
}

OUTPUT_SUBDIR = "numeric_matrix_outputs"


def load_recipes(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Reading Parquet requires pyarrow. Install it with "
            "`pip install pyarrow` and rerun this script."
        ) from exc
    except ValueError as exc:
        if "parquet" in str(exc).lower() or "pyarrow" in str(exc).lower():
            raise ValueError(
                "Unable to read the Parquet input. Install pyarrow with "
                "`pip install pyarrow` and verify the file is a valid Parquet file."
            ) from exc
        raise


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def feature_group(feature: str) -> str:
    if feature in NUTRITION_COLS:
        return "nutrition"
    if feature in TIME_COLS:
        return "time"
    if feature in COUNT_COLS:
        return "count"
    if feature in SERVING_COLS:
        return "serving"
    raise ValueError(f"Unknown numeric feature group for {feature}")


def treatment_for_group(group: str) -> str:
    if group in {"nutrition", "time", "serving"}:
        return "numeric_coerce_nonnegative_median_impute_clip_p995_log1p_standard_scale"
    return "numeric_coerce_nonnegative_median_impute_log1p_standard_scale"


def summarize_series(values: pd.Series, prefix: str) -> dict[str, float]:
    quantiles = values.quantile([0.95, 0.99, 0.995])
    return {
        f"min_{prefix}": float(values.min()),
        f"median_{prefix}": float(values.median()),
        f"p95_{prefix}": float(quantiles.loc[0.95]),
        f"p99_{prefix}": float(quantiles.loc[0.99]),
        f"p995_{prefix}": float(quantiles.loc[0.995]),
        f"max_{prefix}": float(values.max()),
    }


def prepare_numeric_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the log-transformed numeric matrix before StandardScaler.

    Nutrition, time, and serving variables are clipped at p99.5 because EDA
    showed extreme upper-tail outliers. NumIngredients and NumQuantities are
    log-transformed but not clipped because their upper tails are plausible.
    Binary missingness/derivation indicators are excluded from PCA to avoid PCA
    capturing data-quality artifacts instead of recipe structure. Popularity or
    outcome variables such as AggregatedRating and ReviewCount are excluded to
    avoid leakage.
    """
    transformed = pd.DataFrame(index=df.index)
    summary_rows = []

    for col in NUMERIC_FEATURE_COLS:
        group = feature_group(col)
        clipping_quantile = CLIPPING_QUANTILE_BY_GROUP[group]

        numeric = pd.to_numeric(df[col], errors="coerce")
        original_missing_count = int(numeric.isna().sum())
        invalid_negative_count = int((numeric < 0).sum())
        invalid_nonpositive_count = int((numeric <= 0).sum()) if group == "serving" else 0

        cleaned = numeric.copy()
        cleaned.loc[cleaned < 0] = np.nan
        if group == "serving":
            cleaned.loc[cleaned <= 0] = np.nan

        median_used = cleaned.median(skipna=True)
        if pd.isna(median_used):
            raise ValueError(f"Cannot impute {col}: median is NaN after cleaning.")

        imputed = cleaned.fillna(median_used)
        missing_after_imputation_count = int(imputed.isna().sum())

        before_stats = summarize_series(imputed, "before")
        skew_before = float(imputed.skew())

        clipping_applied = clipping_quantile is not None
        clipping_cap_value = np.nan
        clipped_value_count = 0
        clipped = imputed.copy()

        if clipping_applied:
            clipping_cap_value = float(imputed.quantile(clipping_quantile))
            clipped_value_count = int((imputed > clipping_cap_value).sum())
            clipped = imputed.clip(upper=clipping_cap_value)

        logged = np.log1p(clipped)
        transformed[col] = logged

        after_stats = summarize_series(logged, "after_log1p")
        skew_after_log1p = float(logged.skew())

        summary_rows.append(
            {
                "feature": col,
                "feature_group": group,
                "original_missing_count": original_missing_count,
                "invalid_negative_count": invalid_negative_count,
                "invalid_nonpositive_count": invalid_nonpositive_count,
                "median_used_for_imputation": float(median_used),
                "missing_after_imputation_count": missing_after_imputation_count,
                "clipping_applied": clipping_applied,
                "clipping_quantile": clipping_quantile if clipping_applied else "",
                "clipping_cap_value": clipping_cap_value if clipping_applied else "",
                "clipped_value_count": clipped_value_count,
                "skew_before": skew_before,
                "skew_after_log1p": skew_after_log1p,
                **before_stats,
                **after_stats,
            }
        )

    summary = pd.DataFrame(summary_rows)
    return transformed[NUMERIC_FEATURE_COLS], summary


def scale_numeric_matrix(
    X_transformed: pd.DataFrame,
) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_transformed)
    return X_scaled.astype(np.float32), scaler


def build_feature_names() -> pd.DataFrame:
    rows = []
    for feature_index, col in enumerate(NUMERIC_FEATURE_COLS):
        group = feature_group(col)
        clipped_part = "_clipped" if CLIPPING_QUANTILE_BY_GROUP[group] is not None else ""
        rows.append(
            {
                "feature_index": feature_index,
                "original_feature": col,
                "transformed_feature": f"{col}_log1p{clipped_part}_scaled",
                "feature_group": group,
                "treatment": treatment_for_group(group),
            }
        )
    return pd.DataFrame(rows)


def build_scaled_summary(
    X_scaled: np.ndarray, feature_names: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for idx, feature in enumerate(feature_names["original_feature"]):
        col = X_scaled[:, idx]
        rows.append(
            {
                "feature": feature,
                "scaled_mean": float(col.mean()),
                "scaled_std": float(col.std(ddof=0)),
                "scaled_min": float(col.min()),
                "scaled_max": float(col.max()),
            }
        )
    return pd.DataFrame(rows)


def build_config(input_path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "numeric_feature_cols": NUMERIC_FEATURE_COLS,
        "nutrition_cols": NUTRITION_COLS,
        "time_cols": TIME_COLS,
        "count_cols": COUNT_COLS,
        "serving_cols": SERVING_COLS,
        "clipping_quantile_by_group": CLIPPING_QUANTILE_BY_GROUP,
        "transformation": "log1p",
        "scaler": "StandardScaler",
        "excluded_leakage_columns": EXCLUDED_LEAKAGE_COLUMNS,
        "excluded_indicator_columns": EXCLUDED_INDICATOR_COLUMNS,
        "random_state": None,
    }


def resolve_output_dir(out_path: Path) -> Path:
    """Treat --out as the Week 5 parent unless the numeric subdir is explicit."""
    if out_path.name == OUTPUT_SUBDIR:
        return out_path
    return out_path / OUTPUT_SUBDIR


def validate_outputs(
    df: pd.DataFrame,
    X_transformed: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_names: pd.DataFrame,
) -> None:
    n_recipes = len(df)
    expected_shape = (n_recipes, len(NUMERIC_FEATURE_COLS))

    if X_transformed.shape != expected_shape:
        raise ValueError(
            f"Unexpected transformed matrix shape: {X_transformed.shape}; "
            f"expected {expected_shape}."
        )
    if X_scaled.shape != expected_shape:
        raise ValueError(
            f"Unexpected scaled matrix shape: {X_scaled.shape}; expected {expected_shape}."
        )
    if df["RecipeId"].isna().any():
        raise ValueError("RecipeId contains null values.")
    if df["RecipeId"].duplicated().any():
        duplicate_count = int(df["RecipeId"].duplicated().sum())
        raise ValueError(f"RecipeId contains duplicates: {duplicate_count} rows.")

    transformed_values = X_transformed.to_numpy()
    if not np.isfinite(transformed_values).all():
        raise ValueError("X_numeric_log_transformed contains NaN or infinite values.")
    if not np.isfinite(X_scaled).all():
        raise ValueError("X_numeric_scaled contains NaN or infinite values.")

    scaled_means = X_scaled.mean(axis=0)
    scaled_stds = X_scaled.std(axis=0, ddof=0)
    if not np.allclose(scaled_means, 0.0, atol=1e-5):
        raise ValueError("Scaled matrix column means are not approximately zero.")
    if not np.allclose(scaled_stds, 1.0, atol=1e-5):
        raise ValueError("Scaled matrix column standard deviations are not approximately one.")

    expected_features = list(NUMERIC_FEATURE_COLS)
    actual_features = list(feature_names["original_feature"])
    if actual_features != expected_features:
        raise ValueError("Feature name order does not match matrix column order.")


def save_outputs(
    df: pd.DataFrame,
    X_transformed: pd.DataFrame,
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    feature_names: pd.DataFrame,
    preprocessing_summary: pd.DataFrame,
    scaled_summary: pd.DataFrame,
    config: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "recipe_ids": output_dir / "recipe_ids.csv",
        "log_transformed": output_dir / "X_numeric_log_transformed.npy",
        "scaled": output_dir / "X_numeric_scaled.npy",
        "feature_names": output_dir / "numeric_feature_names.csv",
        "preprocessing_summary": output_dir / "numeric_preprocessing_summary.csv",
        "scaler": output_dir / "numeric_scaler.joblib",
        "config": output_dir / "numeric_preprocessing_config.json",
        "scaled_summary": output_dir / "numeric_scaled_summary.csv",
    }

    recipe_ids = pd.DataFrame(
        {
            "row_index": np.arange(len(df), dtype=np.int64),
            "RecipeId": df["RecipeId"].to_numpy(),
        }
    )
    recipe_ids.to_csv(paths["recipe_ids"], index=False)
    np.save(paths["log_transformed"], X_transformed.to_numpy(dtype=np.float32))
    np.save(paths["scaled"], X_scaled.astype(np.float32))
    feature_names.to_csv(paths["feature_names"], index=False)
    preprocessing_summary.to_csv(paths["preprocessing_summary"], index=False)
    joblib.dump(scaler, paths["scaler"])
    scaled_summary.to_csv(paths["scaled_summary"], index=False)
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")

    return paths


def print_warnings(preprocessing_summary: pd.DataFrame, n_rows: int) -> list[str]:
    high_skew_features = preprocessing_summary.loc[
        preprocessing_summary["skew_after_log1p"].abs() > 5, "feature"
    ].tolist()

    clipped_too_much = preprocessing_summary.loc[
        preprocessing_summary["clipping_applied"]
        & (preprocessing_summary["clipped_value_count"] > n_rows * 0.01),
        ["feature", "clipped_value_count"],
    ]

    for _, row in clipped_too_much.iterrows():
        pct = row["clipped_value_count"] / n_rows * 100
        print(
            "WARNING: clipping affected more than 1% of rows for "
            f"{row['feature']}: {int(row['clipped_value_count'])} ({pct:.4f}%)."
        )

    for feature in high_skew_features:
        skew = preprocessing_summary.loc[
            preprocessing_summary["feature"].eq(feature), "skew_after_log1p"
        ].iloc[0]
        print(
            "WARNING: high post-log skewness detected for "
            f"{feature}: {skew:.4f}."
        )

    return high_skew_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PCA-ready numeric matrix from resolved recipe features."
    )
    parser.add_argument(
        "--recipes",
        required=True,
        type=Path,
        help="Path to data/interim/recipes_resolved_features.parquet.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help=(
            "Week 5 output parent directory. Numeric matrix artifacts are written "
            "under numeric_matrix_outputs/ inside this directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes_path = args.recipes
    output_dir = resolve_output_dir(args.out)

    df = load_recipes(recipes_path)
    input_shape = df.shape
    validate_required_columns(df, ["RecipeId", *NUMERIC_FEATURE_COLS])

    X_transformed, preprocessing_summary = prepare_numeric_matrix(df)
    X_scaled, scaler = scale_numeric_matrix(X_transformed)
    feature_names = build_feature_names()
    scaled_summary = build_scaled_summary(X_scaled, feature_names)

    validate_outputs(df, X_transformed, X_scaled, feature_names)
    high_skew_features = print_warnings(preprocessing_summary, len(df))

    config = build_config(recipes_path, output_dir)
    paths = save_outputs(
        df=df,
        X_transformed=X_transformed,
        X_scaled=X_scaled,
        scaler=scaler,
        feature_names=feature_names,
        preprocessing_summary=preprocessing_summary,
        scaled_summary=scaled_summary,
        config=config,
        output_dir=output_dir,
    )

    clipped_feature_count = int(
        (
            preprocessing_summary["clipping_applied"]
            & (preprocessing_summary["clipped_value_count"] > 0)
        ).sum()
    )

    print("Numeric matrix build complete.")
    print(f"Input shape: {input_shape}")
    print(f"Selected numeric matrix shape: {df[NUMERIC_FEATURE_COLS].shape}")
    print(f"Transformed matrix shape: {X_transformed.shape}")
    print(f"Scaled matrix shape: {X_scaled.shape}")
    print(f"Output directory: {output_dir}")
    print(f"Recipe ID mapping: {paths['recipe_ids']}")
    print(f"Scaled numeric matrix: {paths['scaled']}")
    print(f"Preprocessing summary: {paths['preprocessing_summary']}")
    print(f"Number of features clipped: {clipped_feature_count}")
    print(
        "Features with high post-log skewness: "
        f"{', '.join(high_skew_features) if high_skew_features else 'none'}"
    )


if __name__ == "__main__":
    main()
