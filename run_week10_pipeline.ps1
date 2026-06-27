$ErrorActionPreference = "Stop"
python src\features\run_recommendation_experiments.py --reviews data/processed/reviews_processed.csv --recipes-metadata data/processed/recipes_processed.csv --content-svd artifacts/week5/pca_svd/X_content_svd.npy --recipe-ids artifacts/week5/pca_svd/reduced_recipe_ids.csv --out-dir artifacts/week10
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python src\features\generate_week10_reports.py --out-dir artifacts/week10 --report-path reports/Week10_recommendation_explanation.md
