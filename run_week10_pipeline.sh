#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" src/features/run_recommendation_experiments.py \
  --reviews data/processed/reviews_processed.csv \
  --recipes-metadata data/processed/recipes_processed.csv \
  --content-svd artifacts/week5/pca_svd/X_content_svd.npy \
  --recipe-ids artifacts/week5/pca_svd/reduced_recipe_ids.csv \
  --out-dir artifacts/week10

"$PYTHON_BIN" src/features/generate_week10_reports.py \
  --out-dir artifacts/week10 \
  --report-path reports/Week10_recommendation_explanation.md
