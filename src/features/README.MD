# Resolved Recipe Features

`build_resolved_features.py` builds the Week 5 Step 2 intermediate recipe table for the Food.com recommendation project.

It preserves the original processed recipe columns and adds resolved/imputed columns for:

- `RecipeCategory` cleaning and missingness tracking
- `RecipeServings` resolution from explicit servings plus serving-like `RecipeYield`
- prep, cook, and total time consistency resolution
- broad non-serving `yield_unit_token` extraction

This script does not run TF-IDF, SVD, PCA, scaling, `log1p`, or outlier clipping.

## Usage

Run from the project root:

```bash
.venv/bin/python src/features/build_resolved_features.py \
  --recipes data/processed/recipes_processed.csv \
  --out data/interim/recipes_resolved_features.parquet \
  --summary-out artifacts/week5
```

The `--summary-out` argument is treated as the Week 5 parent directory. Summary CSVs are written to:

```text
artifacts/week5/resolved_features_summaries/
```

The resolved dataset is saved only as Parquet. If `pyarrow` is unavailable, the script raises a clear install error and does not write a CSV fallback.

## Outputs

The Parquet output includes all original columns plus the resolved feature columns. The summary directory contains:

- `category_missing_summary.csv`
- `servings_yield_resolution_summary.csv`
- `yield_unit_token_counts.csv`
- `time_presence_summary.csv`
- `time_consistency_summary.csv`
- `time_derivation_summary.csv`
- `time_imputation_summary.csv`

## Notes

`RecipeCategory` is cleaned but not imputed. Missing categories remain missing and are tracked with `category_missing_original`.

Remaining missing numeric servings and time values use category medians when `RecipeCategory_clean` is known and has an available median. Otherwise they use the global median.
