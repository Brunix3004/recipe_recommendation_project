# PCA And SVD Reducer

`reduce_dimensions.py` creates the Week 5 Step 5 reduced recipe representations from the existing Step 3 numeric matrix and Step 4 sparse content matrix.

This step only performs dimensionality reduction. It does not rebuild features, run t-SNE, cluster recipes, or generate recommendations.

## Inputs

Numeric inputs from Step 3:

```text
artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy
artifacts/week5/numeric_matrix_outputs/recipe_ids.csv
artifacts/week5/numeric_matrix_outputs/numeric_feature_names.csv
```

Content inputs from Step 4:

```text
artifacts/week5/content_tf_idf_matrix/X_content_tfidf.npz
artifacts/week5/content_tf_idf_matrix/content_recipe_ids.csv
artifacts/week5/content_tf_idf_matrix/content_feature_names.csv
```

## Usage

Run from the project root:

```bash
.venv/bin/python src/features/reduce_dimensions.py \
  --numeric-matrix artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy \
  --numeric-feature-names artifacts/week5/numeric_matrix_outputs/numeric_feature_names.csv \
  --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv \
  --content-matrix artifacts/week5/content_tf_idf_matrix/X_content_tfidf.npz \
  --content-feature-names artifacts/week5/content_tf_idf_matrix/content_feature_names.csv \
  --content-recipe-ids artifacts/week5/content_tf_idf_matrix/content_recipe_ids.csv \
  --out artifacts/week5/pca_svd \
  --figures reports/figures
```

## Methods

PCA is applied to the dense scaled numeric matrix because nutrition, time, complexity, and serving features have meaningful covariance structure after preprocessing.

TruncatedSVD is applied to the sparse TF-IDF content matrix because content features are high-dimensional and sparse. The script does not densify the full TF-IDF matrix.

The selected reduced representation concatenates:

```text
X_recipe_reduced = [X_content_svd, X_numeric_pca]
```

Content SVD features come first, followed by numeric PCA features.

## Outputs

Artifacts are saved under:

```text
artifacts/week5/pca_svd/
```

Key outputs:

- `X_numeric_pca.npy`
- `X_numeric_pca_2d.npy`
- `X_content_svd.npy`
- `X_recipe_reduced.npy`
- `reduced_recipe_ids.csv`
- `reduced_feature_names.csv`
- `dimensionality_comparison.csv`
- `pca_model_full.joblib`
- `svd_model_selected.joblib`
- `pca_explained_variance.csv`
- `pca_component_loadings.csv`
- `pca_top_loadings.csv`
- `pca_reconstruction_metrics.csv`
- `svd_component_sweep.csv`
- `svd_explained_variance_selected.csv`
- `svd_top_terms.csv`
- `reduction_pipeline_config.json`

Figures are saved under:

```text
reports/figures/
```

Generated plots:

- `pca_cumulative_variance.png`
- `pca_scree_plot.png`
- `pca_reconstruction_error.png`
- `svd_retained_energy_sweep.png`
- `svd_selected_cumulative_energy.png`
- `svd_reconstruction_error_sweep.png`

## Selection Rules

Numeric PCA:

- fits PCA with all numeric components
- selects the smallest number of components reaching at least 90% cumulative explained variance

Content SVD:

- evaluates candidate component counts `50`, `100`, `200`, and `300`
- selects `200` components when valid
- records retained energy and sampled reconstruction error

Sparse text variance is diffuse, so SVD retained energy is expected to be lower than PCA explained variance.

## Validation

Before final outputs are accepted, the script validates:

- numeric and content row counts match
- numeric and content `RecipeId` order is identical
- feature-name counts match matrix dimensions
- numeric matrix contains no NaN or infinite values
- sparse content matrix data contains no NaN or infinite values
- reduced matrices contain no NaN or infinite values
- PCA selected variance reaches the target threshold
- `reduced_feature_names.csv` matches `X_recipe_reduced.npy` column order

The combined `X_recipe_reduced.npy` matrix is intended for future Week 7 clustering and Week 10 content-based recommendation experiments.
