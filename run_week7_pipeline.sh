#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" src/features/build_content_matrix.py \
  --recipes data/interim/recipes_resolved_features.parquet \
  --out artifacts/week5/content_tf_idf_matrix \
  --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv

"$PYTHON_BIN" src/features/reduce_dimensions.py \
  --numeric-matrix artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy \
  --numeric-feature-names artifacts/week5/numeric_matrix_outputs/numeric_feature_names.csv \
  --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv \
  --content-matrix artifacts/week5/content_tf_idf_matrix/X_content_tfidf.npz \
  --content-feature-names artifacts/week5/content_tf_idf_matrix/content_feature_names.csv \
  --content-recipe-ids artifacts/week5/content_tf_idf_matrix/content_recipe_ids.csv \
  --out artifacts/week5/pca_svd \
  --figures reports/figures

"$PYTHON_BIN" src/features/build_clustering_matrix.py \
  --content-svd artifacts/week5/pca_svd/X_content_svd.npy \
  --numeric-pca artifacts/week5/pca_svd/X_numeric_pca.npy \
  --recipe-ids artifacts/week5/pca_svd/reduced_recipe_ids.csv \
  --reduced-feature-names artifacts/week5/pca_svd/reduced_feature_names.csv \
  --out artifacts/week7/clustering_matrix

"$PYTHON_BIN" src/features/run_clustering_experiments.py \
  --clustering-matrix artifacts/week7/clustering_matrix/X_recipe_clustering.npy \
  --feature-metadata artifacts/week7/clustering_matrix/clustering_feature_names.csv \
  --recipe-ids artifacts/week7/clustering_matrix/clustering_recipe_ids.csv \
  --models-out artifacts/week7/clustering_models \
  --figures-out reports/figures

"$PYTHON_BIN" src/features/profile_clusters.py \
  --recipes data/processed/recipes_processed.csv \
  --assignments artifacts/week7/clustering_models/cluster_assignments_k5.csv \
  --evaluation-summary artifacts/week7/clustering_models/clustering_evaluation_summary.csv \
  --clustering-matrix artifacts/week7/clustering_matrix/X_recipe_clustering.npy \
  --clustering-recipe-ids artifacts/week7/clustering_matrix/clustering_recipe_ids.csv \
  --out artifacts/week7/clustering_reports \
  --figures-out reports/figures

"$PYTHON_BIN" src/features/visualize_clusters.py \
  --matrix artifacts/week7/clustering_matrix/X_recipe_clustering.npy \
  --assignments artifacts/week7/clustering_models/cluster_assignments_k5.csv \
  --feature-names artifacts/week7/clustering_matrix/clustering_feature_names.csv \
  --sample-size 10000 \
  --random-state 42 \
  --tsne-perplexity 40 \
  --out artifacts/week7/cluster_visualizations \
  --figures-out reports/figures

"$PYTHON_BIN" src/features/generate_week7_reports.py \
  --evaluation-summary artifacts/week7/clustering_models/clustering_evaluation_summary.csv \
  --selected-assignments artifacts/week7/clustering_models/cluster_assignments_k5.csv \
  --cluster-profile-summary artifacts/week7/clustering_reports/cluster_profile_summary.csv \
  --cluster-numeric-stats artifacts/week7/clustering_reports/cluster_numeric_statistics.csv \
  --clustering-matrix-config artifacts/week7/clustering_matrix/clustering_matrix_config.json \
  --out artifacts/week7/clustering_reports \
  --figures-out reports/figures
