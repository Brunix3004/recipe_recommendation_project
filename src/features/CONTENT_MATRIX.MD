# Content TF-IDF Matrix Builder

`build_content_matrix.py` creates the Week 5 Step 4 sparse recipe-content matrix from the resolved recipe dataset:

```text
data/interim/recipes_resolved_features.parquet
```

This step builds a sparse TF-IDF/content representation only. It does not run SVD, PCA, clustering, or recommendations.

## Usage

Run from the project root:

```bash
.venv/bin/python src/features/build_content_matrix.py \
  --recipes data/interim/recipes_resolved_features.parquet \
  --out artifacts/week5/content_tf_idf_matrix \
  --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv
```

All outputs are written to:

```text
artifacts/week5/content_tf_idf_matrix/
```

## Content Sources

The matrix combines separate sparse feature groups:

- ingredients from `RecipeIngredientParts`
- keywords from `RecipeKeywords` if available, otherwise `Keywords`
- category metadata from `RecipeCategory_clean`
- interpretable yield-unit tokens from `yield_unit_token`

Raw `RecipeYield` is excluded because it mixes servings, product units, volume units, and informal text. Missing categories are not converted into `category__unknown`, and missing yield units are not converted into `yield__missing`.

## Preprocessing

List-like columns are parsed with a safe parser that handles actual lists, stringified lists, malformed strings, empty strings, and missing values.

Tokens are normalized by lowercasing, stripping whitespace, replacing `&` with `and`, removing apostrophes, cleaning punctuation, and replacing whitespace or hyphens with underscores.

Keyword documents remove time-like tags such as:

- `15_mins`
- `30_mins`
- `60_mins`
- `4_hours`
- `time_to_make`
- `weeknight`

Those concepts are already represented in the numeric matrix, so removing them reduces redundancy.

## Vectorizers And Weights

Each content group is vectorized separately and weighted before horizontal stacking:

- ingredients: TF-IDF, `min_df=10`, `max_df=0.80`, weight `1.0`
- keywords: TF-IDF, `min_df=10`, `max_df=0.60`, weight `0.6`
- category: sparse binary token matrix, weight `0.4`
- yield unit: sparse binary token matrix, `min_df=50`, weight `0.2`

Separate vectorizers are used because ingredients, keywords, categories, and yield units represent different semantic sources. Weighting keeps ingredients as the main content signal while letting metadata contribute without dominating.

## Outputs

The output directory contains:

- `X_content_tfidf.npz`: sparse CSR content matrix
- `content_recipe_ids.csv`: row index to `RecipeId` mapping
- `content_feature_names.csv`: matrix column order and feature metadata
- `content_matrix_summary.csv`: per-group matrix statistics
- `top_ingredient_tokens.csv`: ingredient document frequency and total weight
- `top_keyword_tokens.csv`: keyword document frequency and total weight
- `category_token_counts.csv`: category token counts
- `yield_token_counts.csv`: yield token counts
- `removed_time_keyword_counts.csv`: removed time-like keyword counts
- `content_vectorizers.joblib`: fitted vectorizers
- `content_preprocessing_config.json`: reproducible preprocessing configuration

## Validation

Before saving, the script validates:

- row count is unchanged
- `RecipeId` has no nulls or duplicates
- matrix rows equal the number of recipes
- matrix columns equal the rows in `content_feature_names.csv`
- sparse matrix data contains no NaN or infinite values
- `feature_index` is contiguous from `0` to `n_features - 1`
- local content recipe IDs match the dataframe row order
- numeric matrix recipe IDs match content row order when `--numeric-recipe-ids` is provided

The row-order validation is important because later SVD and recommendation steps depend on alignment between the numeric and content representations.
