$ErrorActionPreference = "Stop"

Write-Host "========================================="
Write-Host "Starting Recipe Recommendation Pipeline"
Write-Host "========================================="

Write-Host ""
Write-Host "-----------------------------------------"
Write-Host "Step 1: Data Ingestion (Week 3)"
Write-Host "-----------------------------------------"
python src\ingest_foodcom_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "-----------------------------------------"
Write-Host "Step 2: Feature Engineering (Week 5)"
Write-Host "-----------------------------------------"
python src\features\build_resolved_features.py --recipes data\processed\recipes_processed.csv --out data\interim\recipes_resolved_features.parquet --summary-out artifacts\week5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\features\build_numeric_matrix.py --recipes data\interim\recipes_resolved_features.parquet --out artifacts\week5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "-----------------------------------------"
Write-Host "Step 3: Clustering Pipeline (Week 7)"
Write-Host "-----------------------------------------"
.\run_week7_pipeline.ps1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "-----------------------------------------"
Write-Host "Step 4: Recommendation Pipeline (Week 10)"
Write-Host "-----------------------------------------"
.\run_week10_pipeline.ps1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "-----------------------------------------"
Write-Host "Step 5: Graph Analytics (Week 12)"
Write-Host "-----------------------------------------"
python src\graphs\build_ingredient_graph.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "========================================="
Write-Host "Pipeline completed successfully!"
Write-Host "========================================="
