# Week 5 Feature Engineering and Representation Pipeline

## 1. Purpose of the Week 5 Pipeline

The Week 5 pipeline moves the project from processed raw-like Food.com tables into reusable feature representations. The processed recipe and review files contain useful information, but they are not yet in the right form for clustering, recommendation, or dimensionality reduction. This week therefore focuses on constructing stable recipe-level matrices rather than training a final model.

The pipeline creates several complementary representations:

| Representation | Purpose |
|---|---|
| resolved recipe table | Recipe-level intermediate dataset with cleaned/imputed servings and time features |
| dense numeric matrix | PCA-ready nutrition, time, complexity, and servings features |
| sparse content matrix | TF-IDF representation from ingredients, keywords, category, and yield-unit tokens |
| numeric PCA matrix | compact dense representation of numeric recipe structure |
| content SVD matrix | compact dense representation of sparse recipe content semantics |
| combined recipe embedding | reusable recipe representation for later clustering and recommendation |

This matters for the semester project because later weeks depend on row-aligned, reusable features:

- **Week 7 clustering:** cluster recipes using numeric, content, or combined embeddings.
- **Week 10 recommendation/ranking:** build content-based user profiles from recipe embeddings and compare against collaborative/popularity baselines.
- **Week 12 graph comparison:** compare embedding-based similarity with ingredient/recipe graph similarity.

This is not a recommendation model yet. It is the representation layer that makes later modeling possible.

## 2. Layer Separation

The project separates the data into layers so that features, interactions, and future graph work do not contaminate each other.

| Layer | Role |
|---|---|
| catalog layer | recipe-level metadata and intrinsic recipe attributes |
| feature layer | numeric/content matrices derived from recipe attributes |
| interaction layer | user reviews and ratings, reserved for evaluation and recommender training |
| graph layer | future ingredient/recipe graph representation |
| pipeline layer | reproducible scripts and saved artifacts |

The most important design rule is that recipe-level feature matrices must not be built by directly joining reviews. A review join duplicates recipe rows by review count, which would make popular recipes appear more important simply because they have more reviews. That would corrupt PCA/SVD geometry and later clustering.

For the same reason, `AggregatedRating`, `ReviewCount`, and review-level `Rating` are excluded from intrinsic recipe features. They are outcomes or popularity signals, not recipe content. `RecipeId` is kept only for row alignment and traceability; it is never used as a numeric or text feature.

## 3. Step 1 — EDA and Feature Audit

Step 1 audited the processed Food.com tables and produced EDA summaries under `artifacts/week5/eda_summaries` plus diagnostic figures under `reports/figures`. 

The processed data contained:

| Table | Rows | Columns |
|---|---:|---:|
| recipes | 522,517 | 33 |
| reviews | 1,401,982 | 8 |

The review table covered 271,678 reviewed recipes, while the recipe catalog contained 522,517 recipes. This means 250,843 catalog recipes had zero reviews. The user-item interaction matrix was extremely sparse, with density `0.000009867829`. This supports keeping the review interaction layer separate from recipe content features.

EDA served five purposes:

- identify missingness patterns,
- identify skewness and outliers,
- inspect numeric correlations and consistency,
- audit leakage variables,
- decide which variables belong in PCA versus sparse TF-IDF/SVD.

Key data insights:

- `RecipeYield` was missing for 348,071 recipes, or 66.61%.
- `RecipeServings` was missing for 182,911 recipes, or 35.01%.
- `CookTime_Minutes` was missing for 82,545 recipes, or 15.80%.
- `RecipeCategory` was missing for only 751 recipes, or 0.14%.
- Nutrition and time variables were extremely right-skewed. For example, `CookTime_Minutes` had skewness 610.51, `TotalTime_Minutes` 662.02, and `SugarContent` 493.47.
- Some maximum values were not plausible as ordinary recipe attributes, such as `CookTime_Minutes` at 43,545,600 minutes and `SodiumContent` at 1,246,921.1.
- Ingredients and keywords were list-like sparse signals: EDA found 7,287 unique ingredient tokens and 314 unique keyword tokens.
- Keywords contained useful semantic tags but also time-like tags such as `60_mins`, `30_mins`, `4_hours`, and `15_mins`, which overlap with numeric time features.

Leakage audit:

| Column | Reason excluded from intrinsic features |
|---|---|
| `AggregatedRating` | derived from user behavior |
| `ReviewCount` | popularity metric |
| `Rating` | review outcome signal |
| `RecipeId`, `ReviewId`, `AuthorId` | identifiers, not recipe properties |

Methodological decision from EDA:

- PCA should be reserved for dense numeric features after scaling.
- TruncatedSVD should be reserved for sparse TF-IDF content features.
- t-SNE, if used later, should be visualization-only rather than a reusable modeling representation.

## 4. Step 2 — Resolved Feature Construction

Script: `src/features/build_resolved_features.py`

Step 2 creates the intermediate resolved recipe dataset:

```text
data/interim/recipes_resolved_features.parquet
```

The output has 522,517 rows and 69 columns. It preserves the original processed recipe columns and adds cleaned/resolved features. No log transformation, scaling, PCA, TF-IDF, or SVD happens in this step.

### 4.1 RecipeCategory Treatment

`RecipeCategory` had very low missingness, so complex imputation was intentionally rejected. KNN imputation would add modeling complexity and possible circularity for only 0.1437% of rows. Instead:

- missing categories are tracked with `category_missing_original`,- known categories are normalized into `RecipeCategory_clean`,
- missing categories remain missing,
- no `category__unknown` token is created,
- numeric median fallback uses the global median when category is missing.

Category missingness:

| Metric | Value |
|---|---:|
| total recipes | 522,517 |
| category missing count | 751 |
| category missing percent | 0.1437% |
| category present count | 521,766 |
| category present percent | 99.8563% |

This preserves missingness as metadata absence rather than pretending it is culinary content.

### 4.2 RecipeServings and RecipeYield Treatment

`RecipeServings` is numeric but incomplete. `RecipeYield` is text and cannot be blindly converted into servings. The key problem is that yield strings mix serving counts with product units and informal descriptions.

Serving-like examples:

- `4 servings`
- `serves 4`
- `serves 4-6`
- `6 portions`

Non-serving examples:

- `1 loaf`
- `12 muffins`
- `24 cookies`
- `3 cups`
- `1 batch`

The pipeline creates:

- `RecipeServings_clean`
- `servings_parsed_from_yield`
- `ResolvedServings`
- `ResolvedServings_imputed`
- `servings_from_yield`
- `servings_missing_after_yield_parse`
- `yield_unit_token`

The imputation hierarchy is:

1. use valid `RecipeServings_clean`,
2. else parse serving-like `RecipeYield`,
3. else use the median `ResolvedServings` by `RecipeCategory_clean`,
4. else use the global median.

This is better than simple median imputation because it uses deterministic serving information when available but avoids treating product units as serving counts.

Servings/yield resolution summary:

| Metric | Count | Percent |
|---|---:|---:|
| servings only | 260,852 | 49.9222% |
| yield only | 95,692 | 18.3137% |
| both present | 78,754 | 15.0720% |
| both missing | 87,219 | 16.6921% |
| parsed from yield | 279 | 0.0534% |
| still missing after yield parse | 182,632 | 34.9524% |
| imputed with category median | 182,353 | 34.8990% |
| imputed with global median | 279 | 0.0534% |

Median `ResolvedServings` was 6.0 before imputation and remained 6.0 after imputation.

The most common non-serving yield-unit tokens were:

| Yield token | Recipe count |
|---|---:|
| `yield__cup` | 32,674 |
| `yield__cookie` | 11,429 |
| `yield__loaf` | 8,862 |
| `yield__cake` | 7,639 |
| `yield__pie` | 6,616 |
| `yield__muffin` | 5,742 |
| `yield__quart` | 3,975 |

These tokens are not used as servings. They are preserved as optional content signals for Step 4.

### 4.3 Time Feature Treatment

Raw time columns:

- `CookTime_Minutes`
- `PrepTime_Minutes`
- `TotalTime_Minutes`

Step 2 creates clean, resolved, and imputed versions:

- `CookTime_Minutes_clean`
- `PrepTime_Minutes_clean`
- `TotalTime_Minutes_clean`
- `CookTime_Minutes_resolved`
- `PrepTime_Minutes_resolved`
- `TotalTime_Minutes_resolved`
- `CookTime_Minutes_imputed`
- `PrepTime_Minutes_imputed`
- `TotalTime_Minutes_imputed`
- derivation and consistency indicators

Deterministic rules:

- if cook time is missing and `TotalTime >= PrepTime`, then `CookTime = TotalTime - PrepTime`;
- if prep time is missing and `TotalTime >= CookTime`, then `PrepTime = TotalTime - CookTime`;
- if total time is missing and prep/cook exist, then `TotalTime = PrepTime + CookTime`.

This is superior to direct median imputation because time fields have arithmetic relationships. Median imputation would have inserted artificial cook or prep time and could break consistency.

Time resolution results:

| Feature | Original missing | Resolved missing | Category-median imputed | Global-median imputed | Final missing |
|---|---:|---:|---:|---:|---:|
| CookTime_Minutes | 82,545 | 2 | 2 | 0 | 0 |
| PrepTime_Minutes | 2 | 1 | 1 | 0 | 0 |
| TotalTime_Minutes | 1 | 1 | 1 | 0 | 0 |

Derivation results:

| Rule | Count | Percent |
|---|---:|---:|
| derive CookTime from TotalTime − PrepTime | 82,543 | 15.7972% |
| derive PrepTime from TotalTime − CookTime | 1 | 0.0002% |
| derive TotalTime from PrepTime + CookTime | 0 | 0.0000% |

The key insight is that `CookTime_Minutes` had substantial original missingness, but nearly all missing values were recovered through `TotalTime_Minutes − PrepTime_Minutes`. This indicates that the missingness was largely structural rather than random.

More specifically, among rows where cook time was missing but prep and total time were present, `PrepTime` equaled `TotalTime` in 81,962 cases, or 99.2949% of that relevant case. That means most recovered cook times were zero, not median-like positive cooking durations.

There were 24 rows where all three original clean time fields existed but `TotalTime` did not equal `PrepTime + CookTime`. These were flagged rather than overwritten.

## 5. Step 3 — Numeric PCA-Ready Matrix

Script: `src/features/build_numeric_matrix.py`

Step 3 builds a dense numeric matrix suitable for PCA. It uses only intrinsic recipe features and excludes ratings, review counts, identifiers, and audit indicators.

Outputs:

- `artifacts/week5/numeric_matrix_outputs/recipe_ids.csv`
- `artifacts/week5/numeric_matrix_outputs/X_numeric_log_transformed.npy`
- `artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy`
- `artifacts/week5/numeric_matrix_outputs/numeric_feature_names.csv`
- `artifacts/week5/numeric_matrix_outputs/numeric_preprocessing_summary.csv`
- `artifacts/week5/numeric_matrix_outputs/numeric_scaled_summary.csv`
- `artifacts/week5/numeric_matrix_outputs/numeric_preprocessing_config.json`
- `artifacts/week5/numeric_matrix_outputs/numeric_scaler.joblib`

Selected numeric features:

- nutrition: `Calories`, `FatContent`, `SaturatedFatContent`, `CholesterolContent`, `SodiumContent`, `CarbohydrateContent`, `FiberContent`, `SugarContent`, `ProteinContent`
- time: `CookTime_Minutes_imputed`, `PrepTime_Minutes_imputed`, `TotalTime_Minutes_imputed`
- complexity: `NumIngredients`, `NumQuantities`
- servings: `ResolvedServings_imputed`

Excluded features:

- `AggregatedRating`
- `ReviewCount`
- `Rating`
- `RecipeId`
- `AuthorId`
- binary missingness indicators
- derivation indicators

These exclusions prevent PCA from capturing popularity, identifiers, or data-quality artifacts instead of recipe structure.

### 5.1 Transformation Logic

The preprocessing sequence was:

1. validate numeric values,
2. treat invalid negative values as missing,
3. for servings, also treat nonpositive values as missing,
4. median-impute residual missing values,
5. clip selected upper-tail outliers at p99.5,
6. apply `log1p`,
7. apply `StandardScaler`.

Normality is not the goal. PCA does not require normally distributed inputs. The goal is to reduce extreme leverage from outliers, put features on comparable scales, and make covariance structure more meaningful.

Group-specific treatment:

| Group | Treatment | Reason |
|---|---|---|
| nutrition | p99.5 clipping + `log1p` + scaling | extreme upper-tail values |
| time | p99.5 clipping + `log1p` + scaling | extreme maximums despite reasonable medians |
| counts | `log1p` + scaling, no clipping | upper tails are unusual but plausible |
| servings | p99.5 clipping + `log1p` + scaling | serving counts contain unrealistic extremes |

Clipping and skewness examples:

| Feature | p99.5 cap | Clipped count | Skew before | Skew after log1p |
|---|---:|---:|---:|---:|
| CookTime_Minutes_imputed | 1,200.000 | 2,589 | 665.321 | -0.403 |
| TotalTime_Minutes_imputed | 1,540.000 | 2,607 | 662.023 | 0.276 |
| SugarContent | 430.142 | 2,613 | 493.471 | 0.540 |
| CarbohydrateContent | 648.842 | 2,613 | 413.827 | -0.284 |
| FatContent | 284.684 | 2,613 | 410.573 | -0.264 |
| SaturatedFatContent | 123.500 | 2,605 | 409.774 | 0.243 |
| ResolvedServings_imputed | 50.000 | 2,232 | 344.382 | 0.456 |
| PrepTime_Minutes_imputed | 1,440.000 | 1,052 | 341.383 | 0.510 |

The count features were not clipped:

| Feature | Skew before | Skew after log1p |
|---|---:|---:|
| NumIngredients | 0.828 | -0.579 |
| NumQuantities | 0.970 | -0.291 |

After scaling, every selected feature had mean approximately 0 and standard deviation 1. The resulting `X_numeric_scaled.npy` is dense with shape `(522517, 15)`. This is the PCA input for Step 5.

## 6. Step 4 — Sparse Content TF-IDF Matrix

Script: `src/features/build_content_matrix.py`

Step 4 builds a sparse content matrix from recipe list/text/category signals. The output directory is:

```text
artifacts/week5/content_tf_idf_matrix
```

Content sources:

1. `RecipeIngredientParts`
2. `RecipeKeywords` if available, otherwise `Keywords`
3. `RecipeCategory_clean`
4. `yield_unit_token`

These are content features because they describe what the recipe is made of, how it is tagged, what broad class it belongs to, and occasionally its physical format.

### 6.1 Token Parsing and Normalization

Food.com list-like columns may be stored as stringified Python lists. Safe parsing is therefore required. The parser handles:

- actual Python lists,
- stringified lists,
- empty strings,
- missing values,
- malformed strings.

Malformed rows are not allowed to crash the pipeline. Tokens are normalized by lowercasing, stripping whitespace, replacing spaces and hyphens with underscores, removing problematic punctuation, and discarding invalid tokens such as empty strings or null-like values.

### 6.2 Separate Vectorizers and Weighting

Separate vectorizers were used for ingredients, keywords, category, and yield units. The resulting sparse matrices were horizontally stacked. This is preferable to one uncontrolled text field because each source has different semantic reliability.

| Group | Weight | Rationale |
|---|---:|---|
| ingredients | 1.0 | strongest recipe-content signal |
| keywords | 0.6 | useful semantic tags, but noisier |
| category | 0.4 | broad metadata |
| yield unit | 0.2 | weak auxiliary signal |

### 6.3 Keyword Time-Token Removal

Time-like keywords such as `30_mins`, `60_mins`, `4_hours`, and `weeknight` duplicate numeric time information. They were removed from keyword TF-IDF so that content SVD would focus more on semantic content and less on time metadata already represented numerically.

Removed time-like keyword counts:

| Keyword token | Removed count |
|---|---:|
| `60_mins` | 149,589 |
| `30_mins` | 112,229 |
| `4_hours` | 111,071 |
| `15_mins` | 89,340 |
| `weeknight` | 44,914 |

### 6.4 Category and Yield Treatment

Missing category does not become `category__unknown`. Raw `RecipeYield` is not used. `yield__missing` and `yield__other` are excluded. Only interpretable tokens such as `yield__loaf`, `yield__muffin`, and `yield__cookie` are used.

Content matrix summary:

| Feature group | Non-empty docs | Non-empty percent | Vocabulary size | Nonzeros | Density |
|---|---:|---:|---:|---:|---:|
| ingredients | 522,517 | 100.0000% | 4,053 | 4,005,875 | 0.001892 |
| keywords | 478,263 | 91.5306% | 269 | 2,021,898 | 0.014385 |
| category | 521,766 | 99.8563% | 311 | 521,766 | 0.003211 |
| yield | 89,537 | 17.1357% | 14 | 89,537 | 0.012240 |
| combined content matrix | not applicable | not applicable | 4,647 | 6,639,076 | 0.002734 |

Top ingredient tokens:

| Token | Document frequency |
|---|---:|
| `salt` | 190,464 |
| `butter` | 123,598 |
| `sugar` | 102,808 |
| `onion` | 86,321 |
| `eggs` | 80,436 |
| `water` | 79,884 |
| `olive_oil` | 72,763 |

Top keyword tokens after time-token removal:

| Token | Document frequency |
|---|---:|
| `easy` | 276,838 |
| `meat` | 103,191 |
| `healthy` | 83,305 |
| `vegetable` | 81,264 |
| `low_cholesterol` | 74,121 |
| `beginner_cook` | 67,092 |
| `inexpensive` | 66,052 |

Top category tokens:

| Category token | Recipe count |
|---|---:|
| `category__dessert` | 62,072 |
| `category__lunch_snacks` | 32,586 |
| `category__one_dish_meal` | 31,345 |
| `category__vegetable` | 27,231 |
| `category__breakfast` | 21,101 |

Top yield tokens:

| Yield token | Recipe count |
|---|---:|
| `yield__cup` | 32,674 |
| `yield__cookie` | 11,429 |
| `yield__loaf` | 8,862 |
| `yield__cake` | 7,639 |
| `yield__pie` | 6,616 |

The final `X_content_tfidf.npz` matrix is sparse CSR with shape `(522517, 4647)`, 6,639,076 nonzero values, and density 0.002734. It is not appropriate for dense PCA, but it is prepared for TruncatedSVD.

## 7. Step 5 — PCA and TruncatedSVD

Script: `src/features/reduce_dimensions.py`

Step 5 applies dimensionality reduction to the matrices produced in Steps 3 and 4. It does not rebuild features, cluster recipes, or recommend recipes.

### 7.1 PCA on Numeric Matrix

PCA was applied to `X_numeric_scaled.npy`. PCA is appropriate here because the numeric matrix is dense, scaled, and contains correlated continuous variables describing nutrition, time, complexity, and servings.

The PCA model was fit with all 15 possible components. The selected number of components is the smallest `k` reaching at least 90% cumulative explained variance.

PCA results:

| Metric | Value |
|---|---:|
| numeric input shape | `(522517, 15)` |
| selected PCA components | 8 |
| achieved cumulative variance | 0.931616 |
| reconstruction MSE at selected k | 0.068371 |
| reconstruction RMSE at selected k | 0.261479 |
| numeric PCA matrix shape | `(522517, 8)` |

First components are interpretable from their top loadings:

| Component | Main loading pattern | Interpretation |
|---|---|---|
| PC1 | Calories, fat, protein, saturated fat, sodium, cholesterol | overall nutritional magnitude/richness |
| PC2 | total time, prep time, cook time, ingredient count, quantities, servings | time/complexity/scale dimension |
| PC3 | sugar, carbohydrates, fiber versus cholesterol/fat/protein | carbohydrate/sweetness versus animal-fat/protein contrast |
| PC4 | ingredient count and quantities versus servings/time/sugar | recipe complexity and ingredient breadth |

Plots saved under `reports/figures`:

- `pca_cumulative_variance.png`
- `pca_scree_plot.png`
- `pca_reconstruction_error.png`

### 7.2 TruncatedSVD on TF-IDF Matrix

TruncatedSVD was applied to `X_content_tfidf.npz`. SVD is appropriate because the content matrix is sparse and high-dimensional. Dense PCA would require densifying the full TF-IDF matrix, which is both memory inefficient and methodologically inappropriate for sparse text-style features.

Candidate component counts were evaluated:

| Components | Retained energy | Sample reconstruction RMSE |
|---:|---:|---:|
| 50 | 0.283000 | 0.014919 |
| 100 | 0.394409 | 0.013722 |
| 200 | 0.527379 | 0.012137 |
| 300 | 0.612506 | 0.011006 |

The selected model uses 200 components. Retained energy is 0.527379. This is much lower than PCA’s 0.931616 variance threshold, but that is expected: sparse text variance is diffuse across many terms. Low retained energy in TF-IDF SVD is not automatically a failure; the goal is a compact semantic embedding, not perfect reconstruction.

The sampled reconstruction error used 5,379 rows, bounded by a dense sample limit of 25,000,000 entries. The full sparse matrix was not densified.

Examples of SVD component themes:

| Component | Top positive terms | Interpretation |
|---|---|---|
| SVD1 | `keyword__easy`, `category__dessert`, `ingredient__salt`, `ingredient__butter`, `ingredient__sugar`, `ingredient__eggs` | broad high-frequency easy/dessert/basic ingredient axis |
| SVD2 | `keyword__easy`, `keyword__meat`, `ingredient__onion`, `ingredient__olive_oil`, `keyword__vegetable`, `keyword__poultry` | savory/easy/meat-vegetable preparation theme |
| SVD3 | `keyword__easy`, `category__dessert`, `category__beverages`, `category__15_mins`, `category__30_mins` | quick/easy category-heavy theme |
| SVD4 | `keyword__low_cholesterol`, `keyword__healthy`, `keyword__low_protein`, `category__vegetable`, `keyword__vegan` | health-oriented/vegetable theme |

The presence of broad terms like `keyword__easy` in early components is expected because high-frequency tags often define large axes of variation. Later components are typically more specific.

Plots saved under `reports/figures`:

- `svd_retained_energy_sweep.png`
- `svd_selected_cumulative_energy.png`
- `svd_reconstruction_error_sweep.png`

### 7.3 Combined Reduced Representation

The final dense recipe embedding concatenates:

1. content SVD features first,
2. numeric PCA features second.

This produces:

```text
X_recipe_reduced = [X_content_svd, X_numeric_pca]
```

Shapes:

| Matrix | Shape |
|---|---:|
| `X_numeric_pca.npy` | `(522517, 8)` |
| `X_content_svd.npy` | `(522517, 200)` |
| `X_recipe_reduced.npy` | `(522517, 208)` |

This combined matrix is not a final recommendation model. It is a reusable recipe embedding for Week 7 clustering, Week 10 content-based recommendation, and Week 12 comparison against graph-based similarity.

## 8. Summary of Final Feature Artifacts

| Artifact | Type | Shape if known | Sparse/dense | Purpose | Future use |
|---|---|---:|---|---|---|
| `data/interim/recipes_resolved_features.parquet` | Parquet table | `(522517, 69)` | table | resolved category, servings, time, yield-unit fields | source for Step 3 and Step 4 |
| `artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy` | NumPy matrix | `(522517, 15)` | dense | PCA-ready numeric matrix | numeric PCA |
| `artifacts/week5/content_tf_idf_matrix/X_content_tfidf.npz` | SciPy sparse matrix | `(522517, 4647)` | sparse CSR | recipe content TF-IDF | content SVD |
| `artifacts/week5/pca_svd/X_numeric_pca.npy` | NumPy matrix | `(522517, 8)` | dense | reduced numeric representation | clustering/recommendation features |
| `artifacts/week5/pca_svd/X_content_svd.npy` | NumPy matrix | `(522517, 200)` | dense | reduced content representation | content similarity/recommendation |
| `artifacts/week5/pca_svd/X_recipe_reduced.npy` | NumPy matrix | `(522517, 208)` | dense | combined recipe embedding | Week 7, Week 10, Week 12 |
| `artifacts/week5/pca_svd/dimensionality_comparison.csv` | CSV summary | 5 rows | table | compares original and reduced representations | report and methodology |

## 9. Key Methodological Decisions

| Decision | Reason | Risk avoided | Downstream impact |
|---|---|---|---|
| deterministic time resolution before median imputation | time fields have arithmetic relationships | artificial cook/prep times | consistent time features for PCA |
| controlled `RecipeYield` parsing | yield text mixes servings and product units | converting `12 muffins` into 12 servings | cleaner servings and interpretable yield tokens |
| no KNN imputation for `RecipeCategory` | missingness is only 0.1437% | complexity and circularity | simpler, auditable category treatment |
| no raw `RecipeYield` in TF-IDF | raw yield is heterogeneous | noisy content matrix | only normalized yield-unit signals enter content features |
| no `category__unknown` or `yield__missing` token | missingness is not culinary content | SVD learning missingness artifacts | content SVD focuses on recipe semantics |
| clipping + `log1p` for skewed numeric features | extreme upper tails dominate PCA | outlier-driven principal components | more stable numeric geometry |
| no clipping for `NumIngredients`/`NumQuantities` | upper tails are plausible | flattening complex but valid recipes | complexity remains represented |
| exclude ratings/popularity variables | avoid outcome leakage | popularity-driven clusters/recommendations | fairer content/numeric embeddings |
| PCA for dense numeric matrix | scaled dense features have covariance structure | inappropriate sparse methods | compact numeric components |
| TruncatedSVD for sparse content matrix | TF-IDF is high-dimensional and sparse | densifying sparse matrix | compact content embeddings |
| preserve `RecipeId` row order | later matrices must align | mismatched embeddings/interactions | reliable clustering and recommendation joins |

## 10. Known Limitations and Risks

- `RecipeYield` parsing is heuristic and may miss edge cases or ambiguous serving descriptions.
- Category labels are metadata and may be noisy or inconsistent.
- Keywords may reflect platform/user tagging bias rather than objective recipe semantics.
- Some category tokens are time-like, such as `category__15_mins`; time-like keyword tags were removed, but category labels were preserved as metadata.
- TruncatedSVD retained energy is lower than PCA variance because sparse text variance is diffuse.
- PCA components are linear and may not capture nonlinear relationships among nutrition, time, and complexity.
- `X_recipe_reduced` concatenates content and numeric representations but does not yet optimize feature weights for recommendation quality.
- This pipeline does not evaluate recommendation performance.
- This pipeline does not yet include graph features.

## 11. How These Outputs Will Be Used Later

Week 7:

- K-means or DBSCAN can use `X_recipe_reduced`, `X_content_svd`, or `X_numeric_pca`.
- Cluster profiles can be interpreted using original metadata, top ingredients, categories, and PCA/SVD loadings.
- Numeric-only and content-only clusters can be compared to understand what each representation captures.

Week 10:

- A content-based recommender can use cosine similarity over `X_content_svd` or `X_recipe_reduced`.
- User profiles can be built by averaging embeddings of positively rated recipes.
- Popularity baselines and collaborative filtering should be compared separately, using ratings only in the interaction layer.

Week 12:

- An ingredient graph can be built from normalized ingredient lists.
- Graph centrality or PageRank can be compared with content SVD similarity and popularity.
- Graph-based similarity can be evaluated against embedding-based similarity.

## 12. Reproducibility

The pipeline is reproducible from processed/interim artifacts. The exact run order is:

```bash
python src/features/build_resolved_features.py \
  --recipes data/processed/recipes_processed.csv \
  --out data/interim/recipes_resolved_features.parquet \
  --summary-out artifacts/week5
```

```bash
python src/features/build_numeric_matrix.py \
  --recipes data/interim/recipes_resolved_features.parquet \
  --out artifacts/week5
```

```bash
python src/features/build_content_matrix.py \
  --recipes data/interim/recipes_resolved_features.parquet \
  --out artifacts/week5/content_tf_idf_matrix \
  --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv
```

```bash
python src/features/reduce_dimensions.py \
  --numeric-matrix artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy \
  --numeric-feature-names artifacts/week5/numeric_matrix_outputs/numeric_feature_names.csv \
  --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv \
  --content-matrix artifacts/week5/content_tf_idf_matrix/X_content_tfidf.npz \
  --content-feature-names artifacts/week5/content_tf_idf_matrix/content_feature_names.csv \
  --content-recipe-ids artifacts/week5/content_tf_idf_matrix/content_recipe_ids.csv \
  --out artifacts/week5/pca_svd \
  --figures reports/figures
```

windows users 🥀, icl ts pmo sm rn   :

PowerShell (Windows) equivalents (same scripts, Windows paths):

```powershell
python src\features\build_resolved_features.py `
  --recipes data\processed\recipes_processed.csv `
  --reviews data\processed\reviews_processed.csv `
  --out data\interim\recipes_resolved_features.parquet `
  --summary-out artifacts\week5
```

```powershell
python src\features\build_numeric_matrix.py `
  --recipes data\interim\recipes_resolved_features.parquet `
  --out artifacts\week5
```

```powershell
python src\features\build_content_matrix.py `
  --recipes data\interim\recipes_resolved_features.parquet `
  --out artifacts\week5\content_tf_idf_matrix `
  --numeric-recipe-ids artifacts\week5\numeric_matrix_outputs\recipe_ids.csv
```

```powershell
python src\features\reduce_dimensions.py `
  --numeric-matrix artifacts\week5\numeric_matrix_outputs\X_numeric_scaled.npy `
  --numeric-feature-names artifacts\week5\numeric_matrix_outputs\numeric_feature_names.csv `
  --numeric-recipe-ids artifacts\week5\numeric_matrix_outputs\recipe_ids.csv `
  --content-matrix artifacts\week5\content_tf_idf_matrix\X_content_tfidf.npz `
  --content-feature-names artifacts\week5\content_tf_idf_matrix\content_feature_names.csv `
  --content-recipe-ids artifacts\week5\content_tf_idf_matrix\content_recipe_ids.csv `
  --out artifacts\week5\pca_svd `
  --figures reports\figures
```

All artifacts are regenerated from processed/interim data. Matrix row order is preserved through `RecipeId` mapping files. This avoids hidden notebook state and makes downstream clustering/recommendation scripts safer. Parquet is used for the intermediate resolved dataset because it preserves schema and improves I/O efficiency relative to repeatedly reading a large CSV.

## 13. Report-Ready Conclusion

The Week 5 pipeline transformed processed Food.com recipe data into two complementary recipe representations: a dense numeric PCA representation and a sparse-content SVD representation. The strongest data-quality insight was that `CookTime_Minutes` missingness was mostly structural: most missing cook times could be recovered arithmetically from `TotalTime_Minutes - PrepTime_Minutes`, and most of those cases reflected recipes where total time equaled prep time.

The numeric side required careful preprocessing before PCA. Nutrition, time, and servings features had extreme right skew and outliers, so the pipeline used median imputation, p99.5 clipping, `log1p`, and standardization. The resulting 15-dimensional numeric matrix was reduced to 8 PCA components while preserving 93.1616% cumulative explained variance.

The content side required a sparse representation. Ingredients, keywords, cleaned category tokens, and interpretable yield-unit tokens were vectorized separately, weighted, and horizontally stacked into a sparse TF-IDF matrix with 4,647 features. TruncatedSVD reduced this sparse content matrix to 200 dense semantic dimensions with retained energy 0.527379.

The final `X_recipe_reduced.npy` matrix combines 200 content SVD dimensions with 8 numeric PCA dimensions, producing a 208-dimensional recipe embedding for future clustering and content-based recommendation experiments. This embedding is not a final model; it is the reusable representation layer that later weeks will evaluate and compare against interaction-based and graph-based approaches.
