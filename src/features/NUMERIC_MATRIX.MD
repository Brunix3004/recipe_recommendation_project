# Numeric Matrix Builder

`build_numeric_matrix.py` creates the Week 5 Step 3 PCA-ready numeric feature matrix from:

```text
data/interim/recipes_resolved_features.parquet
```

This step performs numeric preprocessing only. It does not run PCA, SVD, TF-IDF, clustering, or recommendation ranking.

## Usage

Run from the project root:

```bash
.venv/bin/python src/features/build_numeric_matrix.py \
  --recipes data/interim/recipes_resolved_features.parquet \
  --out artifacts/week5
```

All outputs are written to:

```text
artifacts/week5/numeric_matrix_outputs/
```

## Selected Features

The matrix contains 15 intrinsic numeric recipe features:

- nutrition: `Calories`, `FatContent`, `SaturatedFatContent`, `CholesterolContent`, `SodiumContent`, `CarbohydrateContent`, `FiberContent`, `SugarContent`, `ProteinContent`
- resolved time: `CookTime_Minutes_imputed`, `PrepTime_Minutes_imputed`, `TotalTime_Minutes_imputed`
- complexity counts: `NumIngredients`, `NumQuantities`
- servings: `ResolvedServings_imputed`

Popularity/outcome fields such as `AggregatedRating` and `ReviewCount` are excluded to avoid leakage. Binary missingness and derivation indicators are also excluded so PCA captures recipe structure rather than data-quality artifacts.

## Preprocessing

For nutrition, time, and servings:

- coerce to numeric
- set invalid negative values to missing
- impute residual missing values with the column median
- clip upper outliers at p99.5
- apply `log1p`
- fit and apply `StandardScaler`

For `NumIngredients` and `NumQuantities`:

- coerce to numeric
- set invalid negative values to missing
- impute residual missing values with the column median
- apply `log1p`
- fit and apply `StandardScaler`

Counts are not clipped because their upper tails are unusual but plausible.

## Outputs

The output directory contains:

- `recipe_ids.csv`: row index to `RecipeId` mapping
- `X_numeric_log_transformed.npy`: matrix after clipping and `log1p`, before scaling
- `X_numeric_scaled.npy`: PCA-ready scaled matrix
- `numeric_feature_names.csv`: feature order and transformation labels
- `numeric_preprocessing_summary.csv`: missingness, clipping, skewness, and distribution summary
- `numeric_scaled_summary.csv`: scaled mean, std, min, and max by feature
- `numeric_scaler.joblib`: fitted `StandardScaler`
- `numeric_preprocessing_config.json`: reproducible preprocessing configuration

## Validation

Before saving, the script validates:

- row count is unchanged
- `RecipeId` has no nulls or duplicates
- both matrices have shape `n_recipes x 15`
- both matrices contain no NaN or infinite values
- scaled columns are approximately mean 0 and standard deviation 1
- feature-name order matches matrix column order
