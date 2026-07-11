#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python3}"

echo "========================================="
echo "Starting Recipe Recommendation Pipeline"
echo "========================================="

echo
echo "-----------------------------------------"
echo "Step 1: Data Ingestion (Week 3)"
echo "-----------------------------------------"
"$PYTHON_BIN" src/ingest_foodcom_data.py

echo
echo "-----------------------------------------"
echo "Step 2: Feature Engineering (Week 5)"
echo "-----------------------------------------"
"$PYTHON_BIN" src/features/build_resolved_features.py \
  --recipes data/processed/recipes_processed.csv \
  --out data/interim/recipes_resolved_features.parquet \
  --summary-out artifacts/week5

"$PYTHON_BIN" src/features/build_numeric_matrix.py \
  --recipes data/interim/recipes_resolved_features.parquet \
  --out artifacts/week5

echo
echo "-----------------------------------------"
echo "Step 3: Clustering Pipeline (Week 7)"
echo "-----------------------------------------"
bash run_week7_pipeline.sh

echo
echo "-----------------------------------------"
echo "Step 4: Recommendation Pipeline (Week 10)"
echo "-----------------------------------------"
bash run_week10_pipeline.sh

echo
echo "-----------------------------------------"
echo "Step 5: Graph Analytics (Week 12)"
echo "-----------------------------------------"
"$PYTHON_BIN" src/graphs/build_ingredient_graph.py

echo
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
