# Food.com Recipe Recommendation & Intelligence System
**Big Data Course - Semester Group Assignment**

This repository contains the full source code and analysis for our Domain Discovery, Recommendation, and Graph Intelligence semester project. The pipeline runs end-to-end to ingest data, generate representations, cluster recipes, build recommendation models, and analyze ingredient co-occurrence graphs.

## Repository Structure
```text
project/
  data/
    raw/                 # Unprocessed Kaggle export data
    interim/             # Intermediate processed features
    processed/           # Cleaned modeling-ready data
  notebooks/             # Exploratory notebooks
  src/
    features/            # Feature building, clustering, recommendation scripts
    graphs/              # Graph analytics scripts
    ingest_foodcom_data.py # Ingestion pipeline
    demo.py              # Final interactive demo
  reports/               # Markdown reports by week
    figures/             # Visualizations
  artifacts/             # Saved models, data arrays, and metrics
  run_all_pipeline.ps1   # Master runbook script
```

## Setup Instructions
Enforce Python 3.12+ and initialize the environment:
```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

## RUNBOOK: End-to-End Reproducibility
To build the complete pipeline from scratch, ensure you have the raw data and run:
```powershell
.\run_all_pipeline.ps1
```
This executes sequentially:
1. **Week 3 (Ingestion):** `src/ingest_foodcom_data.py`
2. **Week 5 (Feature Engineering):** Resolves features, builds content and numeric matrices, and reduces dimensions using PCA/SVD.
3. **Week 7 (Clustering):** Runs k-means clustering, cluster profiling, and visualization.
4. **Week 10 (Recommendation):** Filters 5-core subset, runs chronological 80/20 train/test split, builds Bayesian Popularity baseline and Collaborative SVD Recommender, evaluates.
5. **Week 12 (Graph Analytics):** Builds undirected ingredient-ingredient graph, calculates PageRank, Jaccard/PPMI centralities, and outputs graphs.

## Final Demo
To explore the generated outputs interactively:
```powershell
python src/demo.py --query "chicken"
```
The demo will show:
- Matching recipes.
- Found clusters and average calorie information.
- Highly central ingredients by PageRank in the graph.

## Reports
The overarching technical report is located in `reports/Week14_Final_Report.md`. Weekly intermediate reports are also available in the `reports/` folder.
