# Week 7 Clustering and Recipe Structure Discovery

## 1. Purpose of Week 7

Week 7 introduces unsupervised structure discovery over recipe embeddings to answer a different question than supervised recommendation. A supervised recommender asks, “Given user interaction history, what should be ranked next?” Week 7 asks, “What latent structure exists in the recipe catalog itself, independent of user labels?” This distinction matters because the catalog contains many recipes with sparse or no interactions; unsupervised grouping can still organize these items before any user-specific model is applied.

The Week 7 objective is therefore methodological and infrastructural: construct reproducible cluster artifacts that expose latent recipe neighborhoods and make the embedding space interpretable. This contributes to four downstream needs. First, exploratory analysis: clusters provide a compact audit of how semantic and nutritional structure is distributed across ~500k recipes. Second, semantic organization: cluster assignments create catalog partitions that are easier to inspect and debug than raw 208-dimensional vectors. Third, cold-start support: new or low-interaction recipes can still be mapped to latent groups without rating history. Fourth, recommendation interpretability: cluster-level summaries provide explanations for why candidate sets are similar (e.g., ingredient style, category tendencies, preparation profile), which is not visible from raw latent coordinates.

In this project, clustering is not positioned as the final recommendation algorithm. It is a representation-analysis layer that informs, constrains, and explains later recommendation logic.

---

## 2. Inputs From Week 5 

Week 7 consumes Week 5 dimensionality-reduced artifacts rather than rebuilding raw features. The three core matrices are already available and row-aligned:

| Representation | Shape | Purpose |
| -------------- | ----- | ------- |
| `X_numeric_pca.npy` | `(522517, 8)` | Compact dense representation of numeric nutrition/time/complexity structure |
| `X_content_svd.npy` | `(522517, 200)` | Dense semantic representation derived from sparse TF-IDF recipe content |
| `X_recipe_reduced.npy` | `(522517, 208)` | Combined latent embedding used for clustering and later recommendation workflows |

Supporting Week 5 dimensionality metadata confirms:
- Numeric PCA selected **8** components with achieved cumulative variance **0.931614875793457**.
- Content TruncatedSVD selected **200** components with retained energy **0.5273789167404175**.
- Combined reduced shape is **(522517, 208)**.

### Why PCA for numeric features

PCA is suitable for dense, standardized numeric variables because these features have meaningful covariance structure and moderate dimensionality after preprocessing (15 numeric inputs reduced to 8 latent components). PCA removes redundancy and stabilizes Euclidean geometry for centroid-based methods by concentrating variance into orthogonal axes.

### Why SVD for sparse TF-IDF content

TruncatedSVD is appropriate for high-dimensional sparse TF-IDF matrices (content input width 4647) because direct dense PCA would be computationally inefficient and memory-heavy. SVD yields dense latent semantic directions while preserving major variance structure needed for similarity-based grouping.

### Why not cluster directly on raw TF-IDF

Direct clustering on raw TF-IDF is problematic due to sparsity, very high dimensionality, and noisy distance concentration effects. In that regime, Euclidean distances become less discriminative, centroid updates become unstable, and runtime/memory costs increase. SVD compression is a pragmatic conditioning step before clustering.

---

## 3. Clustering Objectives

The Week 7 clustering objectives are exploratory and structural, not predictive:

1. Identify latent recipe groups that emerge from combined semantic and numeric embeddings.
2. Discover recurring cuisine/ingredient/preparation regimes in the catalog.
3. Quantify how cluster quality changes with `k` rather than assuming a single predefined cluster count.
4. Build interpretable artifacts (profiles, representative recipes, plots) that can support later recommendation analysis and debugging.
5. Establish reproducible unsupervised baselines for future comparison (e.g., graph-based structure in later milestones).

The clustering outputs are not treated as labels of “ground truth classes.” They are operational partitions of latent space used for analysis, candidate generation, and interpretation.

---

## 4. Clustering Pipeline Architecture

The Week 7 pipeline is script-driven and artifact-first. Existing scripts and outputs indicate the following sequence:

| Stage | Script | Main Inputs | Main Outputs |
| ----- | ------ | ----------- | ------------ |
| Build clustering matrix | `src/features/build_clustering_matrix.py` | `X_content_svd.npy`, `X_numeric_pca.npy`, `reduced_recipe_ids.csv`, `reduced_feature_names.csv` | `artifacts/week7/clustering_matrix/X_recipe_clustering.npy`, feature metadata/config |
| Run clustering sweep | `src/features/run_clustering_experiments.py` | `X_recipe_clustering.npy`, feature metadata, recipe IDs | `cluster_assignments_k*.csv`, `cluster_centroids_k*.csv`, `minibatch_kmeans_k*.joblib`, evaluation summary |
| Profile selected clustering | `src/features/profile_clusters.py` | selected assignment file + processed recipe metadata + clustering matrix | `cluster_profile_summary.csv`, `cluster_numeric_statistics.csv`, `representative_recipes_k*.csv`, cluster boxplots |
| Visual diagnostics | `src/features/visualize_clusters.py` | `X_recipe_reduced.npy`, selected assignments, reduced feature names | `pca_2d_coordinates.csv`, `tsne_2d_coordinates.csv`, visual centroids, diagnostics JSON, scatterplots |
| Week 7 rubric reporting | `src/features/generate_week7_reports.py` | evaluation summary + profile artifacts | validation metrics table, sweep figures, failure analysis markdown, interpretation markdown, week summary markdown |

### Dimensionality/scaling decisions in clustering matrix construction

The Week 7 clustering matrix configuration documents:
- Shape: `(522517, 208)`
- Content branch (`200 dims`) left unscaled
- Numeric PCA branch (`8 dims`) standardized and then weighted
- Weights: `CONTENT_WEIGHT = 1.0`, `NUMERIC_WEIGHT = 0.35`

This design intentionally preserves semantic geometry from SVD while keeping numeric structure as a controlled refinement signal rather than the dominant geometry.

### Artifact organization

- Model/metrics/profile artifacts: `artifacts/week7/`
- Figures: `reports/figures/`

This keeps computational artifacts and report figures separated, supporting reproducibility and grading traceability.

---

## 5. K-Means Clustering

### Method selection and assumptions

The implemented method is `MiniBatchKMeans` (not full-batch Lloyd K-Means), configured with:
- `random_state = 42`
- `batch_size = 4096`
- `n_init = 10`

The choice is operational: the matrix has 522,517 rows and 208 dimensions, so mini-batch updates substantially reduce runtime and memory pressure while preserving centroid-based cluster structure for large-scale unsupervised analysis.

K-Means assumes:
- Euclidean geometry is meaningful in the embedding space,
- clusters are roughly convex/spherical around centroids,
- a fixed `k` is specified,
- initialization can affect local minima.

These assumptions are imperfect in recipe data, but latent reduction and controlled weighting make centroid methods tractable for baseline structure discovery.

### K sweep and internal metrics

Tested `k` values: `[5, 8, 10, 12, 15, 20, 25]`.

| K | Inertia | Silhouette | Davies–Bouldin | Cluster Count | Runtime (s) |
| - | -------:| ----------:| --------------:| ------------:| ----------:|
| 5  | 764829.0000 | 0.0502133 | 3.0823326 | 5  | 3.8624879 |
| 8  | 716293.4375 | 0.0500851 | 2.7520364 | 8  | 2.6828041 |
| 10 | 694723.7500 | 0.0472009 | 2.8612230 | 10 | 2.5966286 |
| 12 | 672357.7500 | 0.0441660 | 2.8961706 | 12 | 2.5089448 |
| 15 | 649028.6875 | 0.0453426 | 2.8386483 | 15 | 4.2803269 |
| 20 | 629902.3125 | 0.0432325 | 2.9514769 | 20 | 4.9958257 |
| 25 | 607975.2500 | 0.0374835 | 2.9921967 | 25 | 5.0477112 |

Interpretation:
- Inertia decreases monotonically with larger `k` (expected).
- Best silhouette occurs at **k=5**.
- Best Davies–Bouldin occurs at **k=8**.
- Metrics disagree; selection is therefore a tradeoff, not a single-metric optimum.

### Final K selection rationale

Current selected operational assignment file is `cluster_assignments_k5.csv` (selected `k=5`). Justification is documented as a balance between silhouette quality and interpretability (smaller number of clusters, easier semantic profiling), while acknowledging that Davies–Bouldin prefers `k=8`. This is a valid engineering tradeoff when unsupervised metrics conflict.

---

**DBSCAN section intentionally omitted:** no DBSCAN clustering script/artifact is present in the current Week 7 code/artifact set.

---

## 7. Cluster Evaluation

Internal validation is available from sweep outputs and derived Week 7 report artifacts:
- `inertia`
- `silhouette_score`
- `davies_bouldin_score`
- runtime and cluster-size statistics

A condensed decision view:

| Selection Reference | K |
| ------------------- | -:|
| Selected operational clustering | 5 |
| Best silhouette | 5 |
| Best Davies–Bouldin | 8 |

Cluster size balance for selected `k=5`:

| Cluster | Recipe Count | Dataset % |
| -------:| -----------:| ---------:|
| 0 | 113,794 | 21.7780 |
| 1 | 70,044  | 13.4051 |
| 2 | 113,927 | 21.8035 |
| 3 | 140,434 | 26.8764 |
| 4 | 84,318  | 16.1369 |

Balance diagnosis:
- Largest/smallest ratio ≈ `140434 / 70044 ≈ 2.00`
- Nontrivial imbalance exists; clusters are not uniformly populated.

### Metric limitations

Internal metrics do not provide semantic correctness guarantees:
- No ground-truth labels exist for “true recipe clusters.”
- Better silhouette does not automatically imply better culinary interpretability.
- Davies–Bouldin and silhouette can disagree, as observed.
- Inertia is scale-sensitive and always improves with larger `k`.

Therefore, semantic profiling and representative-recipe inspection remain necessary validation layers.

---

## 8. Cluster Interpretation

Interpretation was performed by joining cluster assignments to human-readable metadata and by examining nearest recipes to centroids. The profiling stack uses:
- dominant categories,
- dominant keywords,
- dominant ingredient tokens,
- nutritional/time averages,
- representative recipes (closest to centroid in embedding space).

### Cluster semantic summary (selected `k=5`)

| Cluster | Dominant Ingredients | Dominant Categories | Interpretation (from generated profile) |
| ------- | -------------------- | ------------------- | --------------------------------------- |
| 0 | salt, butter, onion | lunch/snacks | High-protein, high-calorie, longer-prep signals with mixed savory semantics |
| 1 | sugar, salt, water | beverages | Low-protein, low-calorie profile with beverage/light-prep tendencies |
| 2 | sugar, butter, salt | dessert | Calorie-dense dessert-oriented cluster with long total-time tendency |
| 3 | salt, onion, olive_oil | one_dish_meal | Savory one-dish structure, moderate-to-high protein profile |
| 4 | salt, onion, water | one_dish_meal | One-dish/healthy-tag overlap, high calorie and long-time tendency |

### Numeric profile view

| Cluster | Avg Calories | Avg Time (min) | Avg Ingredients |
| ------- | -----------: | -------------: | --------------- |
| 0 | 472.0700 | 77.6607  | [INSERT AVG INGREDIENTS CLUSTER 0] |
| 1 | 250.2682 | 112.7141 | [INSERT AVG INGREDIENTS CLUSTER 1] |
| 2 | 606.4366 | 483.7562 | [INSERT AVG INGREDIENTS CLUSTER 2] |
| 3 | 437.1568 | 105.0824 | [INSERT AVG INGREDIENTS CLUSTER 3] |
| 4 | 609.5695 | 550.0540 | [INSERT AVG INGREDIENTS CLUSTER 4] |

(`Avg Ingredients` is not currently exported in the Week 7 profiling CSVs and is intentionally left as placeholder.)

### Representative recipes (nearest to centroids)

| Cluster | Example Nearest Recipes (top-ranked) |
| ------- | ------------------------------------ |
| 0 | *Pescado Borracho (Drunken Fish)*; *Salmon With Cucumber-Dill Cream Napoleons*; *Authentic Rice Pilaf* |
| 1 | *Glazed Onions*; *Foie Gras on a Bed of Pears*; *Quick Pear Tart by Jacques Pepin* |
| 2 | *Passover Nut Bars*; *Stuffed Chocolate Cloud Cupcakes*; *Mint Chocolate Pistachio Cake* |
| 3 | *Creamy Pesto Manicotti*; *Smoked Salmon Quesadillas With Avocado Salsa*; *Summer Vegetable Ragout* |
| 4 | *Artichoke Stuffed Manicotti - Weight Watchers*; *Easy Weeknight Okra & Beef Gumbo*; *Wild Rice Brussels Sprouts and Smoked Gouda Salad* |

### Numeric-only vs content-based differences

Direct side-by-side clustering experiments on `X_numeric_pca` alone and `X_content_svd` alone are **not currently exported** in Week 7 artifacts. Consequently, the following interpretation is methodological (not claimed as measured outcome in this run):
- Numeric-only clustering typically emphasizes nutrition/time complexity regimes.
- Content-only clustering typically emphasizes ingredient/category semantic neighborhoods.
- Combined embedding (`208 dims`) intentionally blends both, with content dominance and numeric refinement.

Any quantitative numeric-vs-content comparison requires dedicated runs and separate metric exports.

---

## 9. Visualization

Visualization artifacts were generated from sampled data for exploratory interpretation, not for model-space replacement.

### Methodological split

- Clustering was run in the 208-dimensional reduced space.
- Visualization used 2D projections (PCA and t-SNE) on sampled rows.
- These 2D projections are diagnostics, not the operational clustering geometry.

### Visualization diagnostics (from exported JSON)

| Diagnostic | Value |
| ---------- | ----- |
| Input shape | `(522517, 208)` |
| Sample size requested | 10,000 |
| Sample size used | 10,000 |
| Sampling random state | 42 |
| PCA 2D explained variance ratio | `[0.38301604986190796, 0.15170733630657196]` |
| PCA 2D cumulative explained variance | `0.5347234010696411` |
| t-SNE perplexity | `40.0` |
| t-SNE KL divergence | `2.1866836547851562` |

### Generated visualization artifacts

| Artifact Type | File |
| ------------- | ---- |
| PCA coordinate export | `artifacts/week7/cluster_visualizations/pca_2d_coordinates.csv` |
| t-SNE coordinate export | `artifacts/week7/cluster_visualizations/tsne_2d_coordinates.csv` |
| Visual centroids export | `artifacts/week7/cluster_visualizations/cluster_visual_centroids.csv` |
| PCA scatter figure | `reports/figures/pca_cluster_scatterplot.png` |
| t-SNE scatter figure | `reports/figures/tsne_cluster_scatterplot.png` |
| Elbow/sweep figures | `reports/figures/clustering_elbow_curve.png`, `reports/figures/silhouette_vs_k.png`, `reports/figures/davies_bouldin_vs_k.png` |
| Cluster size/boxplots | `reports/figures/cluster_size_distribution.png`, `reports/figures/calories_by_cluster.png`, `reports/figures/protein_by_cluster.png`, `reports/figures/total_time_by_cluster.png` |

**Critical interpretation constraint:** t-SNE preserves local neighborhoods and distorts global structure; distances between distant clusters in the 2D map are not a faithful metric proxy for the original 208D space.

---

## 10. Comparison Between Representations

The current Week 7 artifact set operationalizes combined-embedding clustering. Dedicated numeric-only/content-only cluster sweeps are not currently exported as separate evaluation tables.

| Representation | Status in Current Week 7 Artifacts | Expected Structural Emphasis | Quantitative Outcome |
| -------------- | ---------------------------------- | ---------------------------- | -------------------- |
| Numeric PCA only (`X_numeric_pca`) | Not exported as standalone clustering sweep | Nutrition/time/complexity regimes | `[INSERT NUMERIC-ONLY METRICS TABLE OR PATH]` |
| Content SVD only (`X_content_svd`) | Not exported as standalone clustering sweep | Ingredient/category semantic neighborhoods | `[INSERT CONTENT-ONLY METRICS TABLE OR PATH]` |
| Combined embedding (`X_recipe_clustering` / `X_recipe_reduced`) | Implemented and evaluated | Balanced semantic + numeric structure (content-dominant weighting) | See `cluster_validation_metrics.csv` |

Methodological interpretation: combined embeddings were selected as the primary Week 7 clustering space because recommendation-oriented downstream tasks need both semantic proximity and operational nutrition/time constraints. However, without standalone numeric/content sweeps, “best representation” should be stated as an engineering choice under current evidence, not as a fully controlled ablation result.

---

## 11. Known Limitations

1. **Sensitivity to `k`:** internal metrics change across sweep values and do not agree on a single optimum (`k=5` silhouette vs `k=8` Davies–Bouldin).
2. **K-means shape assumptions:** Euclidean centroid models favor convex/spherical structures; recipe semantics may be manifold-like and overlapping.
3. **High-dimensional geometry residuals:** despite reduction, 208D space still carries anisotropy from mixed semantic/numeric signals.
4. **Sparse semantic ambiguity:** ingredient tokens such as salt/onion/butter are frequent and may blur culinary boundaries.
5. **Metadata noise:** keyword/category fields can be inconsistent or coarse, affecting interpretation quality.
6. **Cluster imbalance:** selected run has a largest/smallest cluster ratio near 2.0, which can bias recommendation coverage.
7. **Outlier influence:** extreme nutrition/time values can inflate within-cluster variance; boxplots were made outlier-robust by hiding fliers for readability only (no row deletion).
8. **Category overlap:** dominant category overlap across clusters indicates non-disjoint semantic themes.
9. **Visualization limitations:** PCA/t-SNE plots are exploratory diagnostics and not direct evidence of true global separability.
10. **Representation comparison gap:** numeric-only and content-only clustering sweeps are not currently exported as direct quantitative comparators.

---

## 12. Downstream Use

Cluster artifacts can support recommendation system behavior in several concrete ways:

| Downstream Use | Mechanism | Week 7 Artifact Dependency |
| -------------- | --------- | -------------------------- |
| Diversification | Sample candidates across multiple clusters to reduce redundancy | `cluster_assignments_k*.csv` |
| Cluster-aware reranking | Re-rank within cluster using user constraints (time, nutrition, ingredient exclusions) | assignments + profile summaries |
| Cold-start recipe handling | Assign new recipes to nearest centroid in embedding space for immediate neighborhood retrieval | `minibatch_kmeans_k*.joblib`, `cluster_centroids_k*.csv` |
| Catalog browsing and explainability | Present users with cluster labels/profiles rather than opaque embedding IDs | `cluster_profile_summary.csv`, interpretation markdown |
| Week 12 graph comparison | Compare embedding-derived neighborhoods against graph communities or similarity edges | cluster assignments + centroid structure |

This makes Week 7 an integration layer: clustering outputs are not final recommendations but become reusable priors for candidate generation, diversification control, and explanation.

---

## 13. Reproducibility

The following commands reproduce Week 7 artifacts from scripts currently present in the repository.

### Bash-style commands

```bash
python src/features/build_clustering_matrix.py \
  --content-svd artifacts/week5/pca_svd/X_content_svd.npy \
  --numeric-pca artifacts/week5/pca_svd/X_numeric_pca.npy \
  --recipe-ids artifacts/week5/pca_svd/reduced_recipe_ids.csv \
  --reduced-feature-names artifacts/week5/pca_svd/reduced_feature_names.csv \
  --out artifacts/week7/clustering_matrix

python src/features/run_clustering_experiments.py \
  --clustering-matrix artifacts/week7/clustering_matrix/X_recipe_clustering.npy \
  --feature-metadata artifacts/week7/clustering_matrix/clustering_feature_names.csv \
  --recipe-ids artifacts/week7/clustering_matrix/clustering_recipe_ids.csv \
  --models-out artifacts/week7/clustering_models \
  --figures-out reports/figures

python src/features/profile_clusters.py \
  --recipes data/processed/recipes_processed.csv \
  --assignments artifacts/week7/clustering_models/cluster_assignments_k5.csv \
  --evaluation-summary artifacts/week7/clustering_models/clustering_evaluation_summary.csv \
  --clustering-matrix artifacts/week7/clustering_matrix/X_recipe_clustering.npy \
  --clustering-recipe-ids artifacts/week7/clustering_matrix/clustering_recipe_ids.csv \
  --out artifacts/week7/clustering_reports \
  --figures-out reports/figures

python src/features/visualize_clusters.py \
  --matrix artifacts/week5/pca_svd/X_recipe_reduced.npy \
  --assignments artifacts/week7/clustering_models/cluster_assignments_k5.csv \
  --feature-names artifacts/week5/pca_svd/reduced_feature_names.csv \
  --sample-size 10000 \
  --random-state 42 \
  --tsne-perplexity 40 \
  --out artifacts/week7/cluster_visualizations \
  --figures-out reports/figures

python src/features/generate_week7_reports.py \
  --evaluation-summary artifacts/week7/clustering_models/clustering_evaluation_summary.csv \
  --selected-assignments artifacts/week7/clustering_models/cluster_assignments_k5.csv \
  --cluster-profile-summary artifacts/week7/clustering_reports/cluster_profile_summary.csv \
  --cluster-numeric-stats artifacts/week7/clustering_reports/cluster_numeric_statistics.csv \
  --clustering-matrix-config artifacts/week7/clustering_matrix/clustering_matrix_config.json \
  --out artifacts/week7/clustering_reports \
  --figures-out reports/figures
```

### Windows PowerShell equivalents

```powershell
python .\src\features\build_clustering_matrix.py `
  --content-svd .\artifacts\week5\pca_svd\X_content_svd.npy `
  --numeric-pca .\artifacts\week5\pca_svd\X_numeric_pca.npy `
  --recipe-ids .\artifacts\week5\pca_svd\reduced_recipe_ids.csv `
  --reduced-feature-names .\artifacts\week5\pca_svd\reduced_feature_names.csv `
  --out .\artifacts\week7\clustering_matrix

python .\src\features\run_clustering_experiments.py `
  --clustering-matrix .\artifacts\week7\clustering_matrix\X_recipe_clustering.npy `
  --feature-metadata .\artifacts\week7\clustering_matrix\clustering_feature_names.csv `
  --recipe-ids .\artifacts\week7\clustering_matrix\clustering_recipe_ids.csv `
  --models-out .\artifacts\week7\clustering_models `
  --figures-out .\reports\figures

python .\src\features\profile_clusters.py `
  --recipes .\data\processed\recipes_processed.csv `
  --assignments .\artifacts\week7\clustering_models\cluster_assignments_k5.csv `
  --evaluation-summary .\artifacts\week7\clustering_models\clustering_evaluation_summary.csv `
  --clustering-matrix .\artifacts\week7\clustering_matrix\X_recipe_clustering.npy `
  --clustering-recipe-ids .\artifacts\week7\clustering_matrix\clustering_recipe_ids.csv `
  --out .\artifacts\week7\clustering_reports `
  --figures-out .\reports\figures

python .\src\features\visualize_clusters.py `
  --matrix .\artifacts\week5\pca_svd\X_recipe_reduced.npy `
  --assignments .\artifacts\week7\clustering_models\cluster_assignments_k5.csv `
  --feature-names .\artifacts\week5\pca_svd\reduced_feature_names.csv `
  --sample-size 10000 `
  --random-state 42 `
  --tsne-perplexity 40 `
  --out .\artifacts\week7\cluster_visualizations `
  --figures-out .\reports\figures

python .\src\features\generate_week7_reports.py `
  --evaluation-summary .\artifacts\week7\clustering_models\clustering_evaluation_summary.csv `
  --selected-assignments .\artifacts\week7\clustering_models\cluster_assignments_k5.csv `
  --cluster-profile-summary .\artifacts\week7\clustering_reports\cluster_profile_summary.csv `
  --cluster-numeric-stats .\artifacts\week7\clustering_reports\cluster_numeric_statistics.csv `
  --clustering-matrix-config .\artifacts\week7\clustering_matrix\clustering_matrix_config.json `
  --out .\artifacts\week7\clustering_reports `
  --figures-out .\reports\figures
```

---

## 14. Report-Ready Conclusion

Week 7 established a reproducible unsupervised clustering layer over Week 5 recipe embeddings and produced a complete artifact trail for evaluation, profiling, and interpretation. The core technical finding is not that one metric is universally optimal, but that embedding-conditioned K-means can produce stable, interpretable catalog partitions at scale while exposing explicit tradeoffs in `k` selection. The observed disagreement between silhouette (`k=5`) and Davies–Bouldin (`k=8`) confirms that cluster quality in this domain is multi-objective and must be decided jointly by metric behavior and interpretability constraints.

The embedding design from Week 5 was foundational: PCA compressed correlated numeric structure, SVD compressed sparse semantic structure, and combined 208D representations enabled scalable centroid-based clustering with meaningful semantic profiles. Cluster interpretation artifacts demonstrate that latent groupings can be translated into human-readable structure through ingredients, categories, nutrition/time summaries, and centroid-nearest examples—an essential step for recommendation explainability.

Methodologically, limitations remain explicit: K-means geometric assumptions, cluster imbalance, outlier sensitivity, semantic overlap, and the non-definitive nature of 2D visualization projections. These constraints do not invalidate Week 7; they define its role. Week 7 contributes a structured unsupervised map of the recipe space that supports downstream recommendation diversification, cold-start handling, and interpretable candidate generation. In the semester trajectory, this provides the bridge between representation engineering (Week 5) and recommendation/graph-comparison stages in later milestones.