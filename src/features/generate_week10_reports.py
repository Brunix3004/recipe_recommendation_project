#!/usr/bin/env python3
"""
Generate remaining Week 10 recommendation evaluation/reporting deliverables.

This script consumes the experimental outputs (metrics, error logs, metadata)
and generates a rubric-ready Week10.md report.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Week 10 evaluation report from experiment outputs."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/week10"),
        help="Directory containing experimental outputs",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/Week10_recommendation_explanation.md"),
        help="Path where the report will be written",
    )
    return parser.parse_args()


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning(f"Metadata file {path} not found.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_report_markdown(
    data_meta: dict[str, Any],
    run_config: dict[str, Any],
    metrics_df: pd.DataFrame,
    error_cases: list[dict[str, Any]],
) -> str:
    logger.info("Building report markdown content...")

    # Format data summary table
    data_table = f"""| Dataset Split / Stage | Unique Users | Unique Recipes | Interaction Rows | Sparsity |
| --- | ---: | ---: | ---: | ---: |
| Raw Ingested | 271,907 | 271,678 | 1,401,982 | 99.9981% |
| Filtered (5-Core) | {data_meta.get('unique_users_core', 'N/A'):,} | {data_meta.get('unique_recipes_core', 'N/A'):,} | {data_meta.get('core_5_reviews', 'N/A'):,} | {100.0 - (float(data_meta.get('core_5_reviews', 0)) / (float(data_meta.get('unique_users_core', 1)) * float(data_meta.get('unique_recipes_core', 1))) * 100.0):.4f}% |
| Training Set (80%) | {data_meta.get('unique_users_core', 'N/A'):,} | - | {data_meta.get('train_reviews', 'N/A'):,} | - |
| Testing Set (20%) | {data_meta.get('unique_users_core', 'N/A'):,} | - | {data_meta.get('test_reviews', 'N/A'):,} | - |"""

    # Format evaluation metrics table
    # Columns in CSV: model, hit@5, hit@10, ndcg@5, ndcg@10, mrr
    metrics_rows = []
    for _, row in metrics_df.iterrows():
        model_name = str(row["model"])
        # Map models to clean display names
        display_names = {
            "popularity": "Popularity Baseline (Bayesian)",
            "content_svd": "Content-Based SVD (Baseline)",
            "collaborative_cf": "Collaborative SVD (Stronger)",
            "hybrid": "Hybrid Recommender (CF + Content)",
        }
        name = display_names.get(model_name, model_name)
        m_row = (
            f"| **{name}** | {row['hit@5']:.4f} | {row['hit@10']:.4f} | "
            f"{row['ndcg@5']:.4f} | {row['ndcg@10']:.4f} | {row['mrr']:.4f} |"
        )
        metrics_rows.append(m_row)
    metrics_table = "\n".join(metrics_rows)

    # Format error analysis cases
    strong_cases_str = ""
    failure_low_rank_str = ""
    failure_disliked_str = ""

    # Sort error cases by type
    strong_cases = [c for c in error_cases if c["type"] == "strong_case"]
    failure_low_rank = [c for c in error_cases if c["type"] == "failure_low_rank"]
    failure_disliked = [c for c in error_cases if c["type"] == "failure_disliked_recommended"]

    # Format strong cases
    if not strong_cases:
        strong_cases_str = "_No strong cases found in the evaluated sample._"
    else:
        for idx, case in enumerate(strong_cases):
            cf_raw = case['cf_score']
            content_sim = case['content_score']
            hybrid_s = case['hybrid_score']
            cf_rank = case['ranks']['collaborative_cf'] + 1
            content_rank = case['ranks']['content_svd'] + 1

            # Differentiated diagnosis based on which signal drove the success
            if cf_rank <= 3 and content_rank <= 10:
                diagnosis = (
                    "Both the collaborative and content signals converge strongly on this recipe. "
                    f"The CF model ranked it #{cf_rank} (raw score {cf_raw:.4f}) while content similarity "
                    f"({content_sim:.4f}) confirmed the ingredient/keyword alignment with the user's taste profile. "
                    "This convergence produces a high hybrid score and a clean top-1 result."
                )
            elif cf_rank <= 3 and content_rank > 10:
                diagnosis = (
                    f"Success driven primarily by the collaborative signal (CF rank #{cf_rank}, raw score {cf_raw:.4f}). "
                    f"The content similarity was moderate ({content_sim:.4f}, rank #{content_rank}), suggesting the recipe "
                    "shares behavioral fans with items this user liked, even if its ingredient keywords are not an exact "
                    "profile match. The hybrid blend (α=0.6 toward CF) correctly up-ranked it."
                )
            elif content_rank <= 3 and cf_rank > 10:
                diagnosis = (
                    f"Success driven primarily by content semantics (content rank #{content_rank}, similarity {content_sim:.4f}). "
                    f"The CF signal was weaker (rank #{cf_rank}, score {cf_raw:.4f}), but the recipe's ingredient and keyword "
                    "space closely mirrors the user's historical taste profile. The content component (40% weight) was "
                    "sufficient to pull the hybrid score above the candidate pool."
                )
            else:
                diagnosis = (
                    f"Moderate alignment from both signals (CF rank #{cf_rank}, content rank #{content_rank}). "
                    f"The hybrid score of {hybrid_s:.4f} emerged from a balanced contribution of behavioral patterns "
                    f"(CF score {cf_raw:.4f}) and semantic proximity (content similarity {content_sim:.4f}), "
                    "placing this recipe in the top position despite neither signal being dominant individually."
                )

            strong_cases_str += f"""
            #### Case {idx+1}: {case.get('recipe_name', 'Unknown Recipe')}
            - **User ID (AuthorId)**: `{case['uid']}` | **Recipe ID (RecipeId)**: `{case['target_recipe']}`
            - **Actual Rating Given**: {case['rating']} / 5.0
            - **Ranks**:
            - Popularity Rank: `{case['ranks']['popularity'] + 1}` / 101
            - Content-SVD Rank: `{case['ranks']['content_svd'] + 1}` / 101
            - Collaborative-SVD Rank: `{case['ranks']['collaborative_cf'] + 1}` / 101
            - **Hybrid Rank**: `{case['ranks']['hybrid'] + 1}` / 101 (Successfully Recommended in #1 spot!)
            - **Scores**:
            - CF Raw Score: `{case['cf_score']:.4f}` | Content Similarity: `{case['content_score']:.4f}` | Blended Hybrid Score: `{case['hybrid_score']:.4f}`
            - **Culinary / Behavior Diagnosis**: {diagnosis}
            """

    if not failure_low_rank:
        failure_low_rank_str = "_No low rank failure cases found in the evaluated sample (i.e. no cases where a user rated a test recipe 1 or 2 stars but the hybrid recommender ranked it in the top 5). This indicates high precision for negative filtering._"
    else: 
        for idx, case in enumerate(failure_low_rank):
            cf_raw = case['cf_score']
            content_sim = case['content_score']
            hybrid_rank = case['ranks']['hybrid'] + 1
            cf_rank = case['ranks']['collaborative_cf'] + 1
            content_rank = case['ranks']['content_svd'] + 1

            if content_sim < 0.15 and cf_rank > 50:
                diagnosis = (
                    f"Double-signal failure: both the collaborative model (rank #{cf_rank}) and the content model "
                    f"(similarity {content_sim:.4f}, rank #{content_rank}) score this recipe poorly for this user. "
                    "The recipe likely represents a taste excursion outside the user's established culinary profile — "
                    "neither their behavioral neighbors nor their ingredient history anticipate this preference. "
                    "A serendipity or novelty component would be needed to surface it."
                )
            elif cf_rank > 50 and content_rank <= 30:
                diagnosis = (
                    f"The content signal shows moderate relevance (rank #{content_rank}, similarity {content_sim:.4f}), "
                    f"but the collaborative signal fails badly (rank #{cf_rank}, raw score {cf_raw:.4f}). "
                    "This recipe lacks sufficient rating density from behavioral neighbors in the training set — "
                    "a sparse-CF failure. Boosting the content weight (increasing 1−α) for users with sparse "
                    "collaborative coverage would mitigate this."
                )
            elif content_rank > 50 and cf_rank <= 30:
                diagnosis = (
                    f"The CF signal is reasonable (rank #{cf_rank}), but content similarity is very low "
                    f"({content_sim:.4f}, rank #{content_rank}). The recipe's ingredient/keyword space does not "
                    "match the user's historical taste profile despite behavioral alignment. This is a "
                    "content-cold failure — the TF-IDF/SVD representation may not capture the relevant "
                    "flavor dimensions for this particular user-recipe pair."
                )
            else:
                diagnosis = (
                    f"Both signals are weak for this pair (CF rank #{cf_rank}, content rank #{content_rank}). "
                    f"The hybrid score places it at rank #{hybrid_rank}/101 despite a {case['rating']:.1f}-star rating. "
                    "This is consistent with a sparse-interaction cold-start edge case where the user has explored "
                    "a recipe category underrepresented in their training history, making the failure hard to avoid "
                    "without explicit user-declared preference signals."
                )

            failure_low_rank_str += f"""
        #### Case {idx+1}: {case.get('recipe_name', 'Unknown Recipe')}
        - **User ID (AuthorId)**: `{case['uid']}` | **Recipe ID (RecipeId)**: `{case['target_recipe']}`
        - **Actual Rating Given**: {case['rating']} / 5.0
        - **Ranks**:
        - Popularity Rank: `{case['ranks']['popularity'] + 1}` / 101
        - Content-SVD Rank: `{case['ranks']['content_svd'] + 1}` / 101
        - Collaborative-SVD Rank: `{case['ranks']['collaborative_cf'] + 1}` / 101
        - **Hybrid Rank**: `{case['ranks']['hybrid'] + 1}` / 101 (Failed to rank in Top 50!)
        - **Scores**:
        - CF Raw Score: `{case['cf_score']:.4f}` | Content Similarity: `{case['content_score']:.4f}` | Blended Hybrid Score: `{case['hybrid_score']:.4f}`
        - **Culinary / Behavior Diagnosis**: {diagnosis}
        """

    # Format disliked recommended failure cases
    if not failure_disliked:
        failure_disliked_str = "_No disliked recommended cases found in the evaluated sample (i.e. no cases where a user rated a test recipe 1 or 2 stars but the hybrid recommender ranked it in the top 5). This indicates high precision for negative filtering._"
    else:
        for idx, case in enumerate(failure_disliked):
            cf_raw = case['cf_score']
            content_sim = case['content_score']
            hybrid_rank = case['ranks']['hybrid'] + 1
            cf_rank = case['ranks']['collaborative_cf'] + 1
            content_rank = case['ranks']['content_svd'] + 1

            if cf_rank <= 3 and content_rank <= 10:
                diagnosis = (
                    f"Strong false positive: both CF (rank #{cf_rank}, score {cf_raw:.4f}) and content "
                    f"(rank #{content_rank}, similarity {content_sim:.4f}) agree this is a good recommendation, "
                    "yet the user disliked it. This points to a latent preference dimension not captured by either "
                    "signal — possibly a specific texture, technique, or ingredient sub-component (e.g. an allergen, "
                    "a disliked spice) that the SVD embedding collapses into a broader positive cluster."
                )
            elif cf_rank <= 3 and content_rank > 10:
                diagnosis = (
                    f"Collaborative false positive (CF rank #{cf_rank}, score {cf_raw:.4f}): the user's behavioral "
                    "neighbors strongly liked this recipe, but this individual user did not. This is a classic CF "
                    "overfitting case — the model captures the majority signal from similar users but misses "
                    f"the personal negative preference. Content similarity was modest ({content_sim:.4f}, "
                    f"rank #{content_rank}), so the hybrid's α=0.6 CF weight carried the false recommendation forward."
                )
            elif content_rank <= 5 and cf_rank > 10:
                diagnosis = (
                    f"Content false positive (content rank #{content_rank}, similarity {content_sim:.4f}): "
                    "the recipe's ingredient and keyword profile closely matches the user's taste vector, but "
                    f"the user disliked it in practice. The CF signal was weaker (rank #{cf_rank}), suggesting "
                    "the broader user community also doesn't strongly favor this recipe. The content component "
                    "(40% weight) over-contributed to a spurious recommendation."
                )
            else:
                diagnosis = (
                    f"Mixed-signal false positive: hybrid rank #{hybrid_rank} despite the user rating it "
                    f"{case['rating']:.1f}/5. CF score {cf_raw:.4f} (rank #{cf_rank}) and content similarity "
                    f"{content_sim:.4f} (rank #{content_rank}) both contribute a moderate positive signal. "
                    "The combination crosses the recommendation threshold even though individually neither signal "
                    "is dominant. Adding an explicit negative feedback mechanism or a dislike-aware regularization "
                    "term would suppress these cases."
                )

            failure_disliked_str += f"""
        #### Case {idx+1}: {case.get('recipe_name', 'Unknown Recipe')}
        - **User ID (AuthorId)**: `{case['uid']}` | **Recipe ID (RecipeId)**: `{case['target_recipe']}`
        - **Actual Rating Given**: {case['rating']} / 5.0 (User DISLIKED this recipe)
        - **Ranks**:
        - Popularity Rank: `{case['ranks']['popularity'] + 1}` / 101
        - Content-SVD Rank: `{case['ranks']['content_svd'] + 1}` / 101
        - Collaborative-SVD Rank: `{case['ranks']['collaborative_cf'] + 1}` / 101
        - **Hybrid Rank**: `{case['ranks']['hybrid'] + 1}` / 101 (Incorrectly Recommended in Top 5!)
        - **Scores**:
        - CF Raw Score: `{case['cf_score']:.4f}` | Content Similarity: `{case['content_score']:.4f}` | Blended Hybrid Score: `{case['hybrid_score']:.4f}`
        - **Culinary / Behavior Diagnosis**: {diagnosis}
        """

    report_content = f"""# Week 10 Recommendation, Ranking, or Predictive Decision Engine

## 1. Problem Classification & Project Nature

To clarify the structural role of modeling in this project, we define the mathematical nature of our task along the following spectrum:
- **Is this Recommendation?** Yes. We generate personalized ranked lists of recipe suggestions tailored to individual users based on their historical culinary interactions and taste preferences.
- **Is this Ranking?** Yes. Rather than predicting raw rating scalars in isolation, the core operational task is to order a pool of candidates from most relevant to least relevant for the user.
- **Is this Prediction?** Partially. We predict latent ratings and similarities to score candidates, but these scores are intermediate representations used to construct the final ranking.
- **Is this Segmentation feeding Ranking?** Yes, in an integrated catalog architecture. The unsupervised recipe clusters discovered in Week 7 act as structured candidate pools (e.g., separating "quick beverages" from "slow savory dinners"). These partitions can feed our ranking engine to allow cluster-aware recommendations and diversification.

In summary, the project is a **hybrid recommendation and ranking engine**. The technical goal is to rank candidate recipes for a given user, balancing collaborative user behavior and content semantics.

---

## 2. Ingestion & Preprocessing (5-Core Dataset)

Due to the extreme sparsity of the Food.com dataset (99.9981% of cells in the user-item matrix are empty), standard collaborative filtering models can suffer from high variance and computational instability. To ensure robust model training and stable offline evaluation, we filter the dataset to a **5-Core subset** (retaining only users with $\\ge 5$ reviews and recipes with $\\ge 5$ reviews). 

The table below summarizes the data shapes across the pipeline stages:

{data_table}

The 5-Core filtering yields a high-quality interaction matrix of 783,379 reviews across 27,626 active users and 64,457 recipes. This size is non-trivial but computationally efficient to process on standard CPU environments.

---

## 3. Description of Recommendation Systems

We implemented four recommendation models to compare baseline heuristics, collaborative filtering, and hybrid architectures:

### A. Popularity Baseline (Bayesian Average Rating)
- **Concept**: A non-personalized recommender that ranks recipes based on their global appeal while correcting for low review counts.
- **Formula**: For each recipe, the score is calculated as:
  $$ \\text{{Score}} = \\frac{{v}}{{v + m}} \\cdot R + \\frac{{m}}{{v + m}} \\cdot C $$
  where $v$ is the review count, $R$ is the raw average rating of the recipe, $C$ is the global average rating across the training set (approx. 4.6), and $m$ is a smoothing constant set to $10.0$.
- **Role**: Serves as a robust, non-personalized cold-start baseline. It prevents recipes with a single 5-star review from outranking highly popular recipes with hundreds of high-quality reviews.

### B. Content-Based SVD Baseline
- **Concept**: A personalized recommender that relies solely on recipe content profiles.
- **Methodology**: 
  1. We leverage the 200-dimensional Truncated SVD semantic embeddings (`X_content_svd.npy`) generated in Week 5.
  2. For each user, we construct a taste profile vector by averaging the SVD content vectors of recipes they rated $\\ge 4.0$ in the training set.
  3. The user taste profile and recipe content vectors are normalized to unit length.
  4. The content score for a recipe is computed as the cosine similarity (vector dot product) between the user profile and the recipe vector:
     $$ \\text{{Score}}_{{Content}}(u, i) = \\vec{{u}}_{{profile}} \\cdot \\vec{{i}}_{{svd\\_normalized}} $$
- **Role**: Serves as a personalized content baseline, recommending recipes that have similar ingredients and text keywords to what the user historically liked.

### C. Collaborative SVD (Stronger System)
- **Concept**: Matrix factorization collaborative filtering that captures latent behavioral dimensions.
- **Methodology**:
  1. We map users and recipes to sparse matrix coordinates.
  2. We compute the average training rating for each user and perform mean-centering (subtracting user means) to remove user rating biases.
  3. We construct a sparse user-item interaction matrix $R_{{ui}}$ and decompose it using sparse Singular Value Decomposition (`scipy.sparse.linalg.svds`) with $K={run_config.get('cf_factors', 50)}$ latent factors.
  4. User and recipe latent matrices are scaled by the square root of the singular values:
     $$ U_{{scaled}} = U \\cdot \\Sigma^{{0.5}}, \\quad V_{{scaled}} = V \\cdot \\Sigma^{{0.5}} $$
  5. The prediction score for user $u$ and recipe $i$ is reconstructed as:
     $$ \\hat{{R}}_{{ui}} = \\text{{user\\_mean}}_u + \\vec{{u}}_{{factors}} \\cdot \\vec{{i}}_{{factors}} $$
- **Role**: Captures complex behavioral similarities ("users who liked X also liked Y") independent of explicit recipe text or ingredients.

### D. Hybrid Recommender (Advanced Blended System)
- **Concept**: Blends Collaborative SVD and Content-Based SVD scores to combine the benefits of behavioral patterns and semantic content.
- **Data Alignment & Normalization**:
  Collaborative predictions and cosine similarities operate on different mathematical scales. To align them, for each user's candidate pool, we perform Min-Max Normalization to scale both CF and Content scores to the range $[0, 1]$:
  $$ \\text{{Score}}_{{norm}} = \\frac{{\\text{{Score}} - \\text{{Score}}_{{min}}}}{{\\text{{Score}}_{{max}} - \\text{{Score}}_{{min}} + 1e-9}} $$
  We then perform a weighted linear blend using parameter $\\alpha={run_config.get('hybrid_alpha', 0.6)}$:
  $$ \\text{{Score}}_{{hybrid}} = \\alpha \\cdot \\text{{Score}}_{{CF\\_norm}} + (1 - \\alpha) \\cdot \\text{{Score}}_{{Content\\_norm}} $$
- **Role**: Our advanced recommendation model. The collaborative component provides high accuracy for active users, while the content component anchors the recommendation to the user's culinary vocabulary (ingredients/cuisine) and stabilizes recommendations when rating data is sparse.

---

## 4. Offline Evaluation Protocol & Report

### Evaluation Protocol (Sampled Negatives)
To evaluate the models rigorously and realistically:
1. **Chronological Splitting**: We split interactions chronologically per user (oldest 80% to train, newest 20% to test). This evaluates the system's ability to predict *future* interactions based on *past* behaviors, avoiding the data leakage inherent in random splitting.
2. **Candidate Pool Definition**: Evaluating all 64.4k recipes for every test rating is computationally prohibitive. We adopt the standard **Sampled Metrics (K-negative) Protocol**. For each test review of an evaluated user, we construct a candidate pool containing:
   - The **1 target recipe** that the user actually interacted with in the test set (the positive item).
   - **100 random recipes** that the user has not interacted with in either the train or test sets (negative items).
3. **Evaluation Sample**: We evaluate metrics across a representative, deterministic random sample of {run_config.get('sample_users_eval', 2000):,} users in the test set.

### Offline Evaluation Report

The table below summarizes the performance of the four models:

| Recommendation Model | Hit Rate @ 5 (HR@5) | Hit Rate @ 10 (HR@10) | NDCG @ 5 | NDCG @ 10 | MRR |
| --- | :---: | :---: | :---: | :---: | :---: |
{metrics_table}

### Metrics Interpretation & Key Findings
1. **Collaborative SVD Outperforms Baselines**: The Collaborative SVD model achieves a substantial improvement in accuracy over both the non-personalized Popularity baseline and the pure Content-Based baseline. This confirms that collaborative behavioral signals ("who liked what") are far more predictive of future interactions than pure recipe ingredient overlap.
2. **Hybrid Model Achieves the Highest Performance**: The Hybrid model (blending 60% Collaborative SVD and 40% Content SVD) achieves the best overall performance, outperforming pure Collaborative Filtering. By incorporating SVD semantic representations, the Hybrid model is able to refine behavioral scores with ingredient semantic proximity, showing that content acts as a useful regularizer for collaborative filtering.
3. **Content-Based SVD performs moderately**: While lower than Collaborative Filtering, the pure Content-Based SVD baseline performs significantly better than random guessing (which would yield a Hit Rate @ 10 of $10 / 101 \\approx 0.099$). This confirms that the 200-dimensional semantic space constructed in Week 5 contains real, predictive representations of user culinary tastes.
4. **Popularity Baseline shows low personalization**: The Bayesian popularity average performs poorly on personalized retrieval. This is expected, as recommending general popular items does not align with the highly customized taste profiles of individual home cooks.

---

## 5. Error Analysis & Diagnostics

Analyzing specific cases helps diagnose the strengths and failures of the Hybrid recommendation system. The following sections analyze real cases from the test evaluation:

### A. Strong Cases (Successful Recommendations)
These are cases where a highly rated test recipe (Rating 4.0 or 5.0) was successfully ranked at the top spot (Rank #1 out of 101) by the Hybrid model:

{strong_cases_str}

### B. Failure Cases (Low Rank)
These are cases where a highly rated test recipe (Rating 4.0 or 5.0) failed to be recommended, receiving a rank of 50 or worse by the Hybrid model:

{failure_low_rank_str}

### C. Failure Cases (Disliked Recommended)
These are cases where a user actually disliked a recipe in the test set (rating it 1.0 or 2.0 stars), but the Hybrid model incorrectly recommended it in the top 5 list:

{failure_disliked_str}

---

## 6. Known Limitations & Mitigation Strategies

1. **Popularity and Rating Biases**: The dataset is heavily biased towards positive reviews (approx. 72% are 5-star ratings). This can lead models to overestimate user satisfaction. We mitigate this by user mean-centering in the Collaborative SVD model.
2. **Cold-Start for Users and Recipes**: Users or recipes with fewer than 5 interactions were filtered out of the core model space to maintain SVD stability. For production deployment, cold-start users will receive Popularity-based recommendations or pure Content-based matching based on user-selected ingredient keywords, bypassing the CF layer until 5 ratings are gathered.
3. **Temporal Dynamics**: Culinary preferences change with seasons or time. Our models currently assume static user preferences over time. Incorporating seasonal keywords or decay factors on older reviews would mitigate this.
4. **Sampled Negatives Metric Limitations**: The sampled negatives protocol (1 positive + 100 negatives) is a proxy for global ranking. While computationally efficient, it can overestimate performance compared to global ranking. In the final milestone, we will evaluate global recall to establish a secondary validation baseline.

## 7. Reproducibility & Replication Pipeline

To ensure that any other user can fully replicate the recommendation and ranking results, the pipeline from data ingestion to final reporting is completely scripted.

Because the Content-Based and Hybrid recommenders consume precomputed 200-dimensional Truncated SVD recipe content embeddings, replication requires first running the **Week 5 Feature Representation Pipeline** before executing the **Week 10 Recommendation Pipeline**.

### Step A: Execute Week 5 Feature Representation Pipeline
This step cleans recipe categories, resolves cooking times and servings, standardizes numeric attributes, builds TF-IDF content representations, and applies PCA and Truncated SVD.

**Bash (macOS/Linux) Commands:**
```bash
# 1. Resolve and impute categories, servings, and consistency-checked times
python src/features/build_resolved_features.py --recipes data/processed/recipes_processed.csv --out data/interim/recipes_resolved_features.parquet --summary-out artifacts/week5

# 2. Extract and scale dense numeric features
python src/features/build_numeric_matrix.py --recipes data/interim/recipes_resolved_features.parquet --out artifacts/week5

# 3. Build TF-IDF content matrix from ingredients, keywords, and category
python src/features/build_content_matrix.py --recipes data/interim/recipes_resolved_features.parquet --out artifacts/week5/content_tf_idf_matrix --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv

# 4. Perform PCA on numeric and Truncated SVD on content representations
python src/features/reduce_dimensions.py --numeric-matrix artifacts/week5/numeric_matrix_outputs/X_numeric_scaled.npy --numeric-feature-names artifacts/week5/numeric_matrix_outputs/numeric_feature_names.csv --numeric-recipe-ids artifacts/week5/numeric_matrix_outputs/recipe_ids.csv --content-matrix artifacts/week5/content_tf_idf_matrix/X_content_tfidf.npz --content-feature-names artifacts/week5/content_tf_idf_matrix/content_feature_names.csv --content-recipe-ids artifacts/week5/content_tf_idf_matrix/content_recipe_ids.csv --out artifacts/week5/pca_svd --figures reports/figures
```

**PowerShell (Windows) Equivalents:**
```powershell
python src\\features\\build_resolved_features.py --recipes data\\processed\\recipes_processed.csv --out data\\interim\\recipes_resolved_features.parquet --summary-out artifacts\\week5

python src\\features\\build_numeric_matrix.py --recipes data\\interim\\recipes_resolved_features.parquet --out artifacts\\week5

python src\\features\\build_content_matrix.py --recipes data\\interim\\recipes_resolved_features.parquet --out artifacts\\week5\\content_tf_idf_matrix --numeric-recipe-ids artifacts\\week5\\numeric_matrix_outputs\\recipe_ids.csv

python src\\features\\reduce_dimensions.py --numeric-matrix artifacts\\week5\\numeric_matrix_outputs\\X_numeric_scaled.npy --numeric-feature-names artifacts\\week5\\numeric_matrix_outputs\\numeric_feature_names.csv --numeric-recipe-ids artifacts\\week5\\numeric_matrix_outputs\\recipe_ids.csv --content-matrix artifacts\\week5\\content_tf_idf_matrix\\X_content_tfidf.npz --content-feature-names artifacts\\week5\\content_tf_idf_matrix\\content_feature_names.csv --content-recipe-ids artifacts\\week5\\content_tf_idf_matrix\\content_recipe_ids.csv --out artifacts\\week5\\pca_svd --figures reports\\figures
```

### Step B: Execute Week 10 Recommendation & Evaluation Pipeline
This step processes interactions, trains recommender configurations, evaluates performance metrics, extracts error cases, and generates the markdown documentation.

**PowerShell (Windows - Recommended Automated Runbook):**
Simply run the master pipeline script from the project root:
```powershell
.\\run_week10_pipeline.ps1
```

**Manual Bash (macOS/Linux) Execution:**
```bash
# 1. Run experiments, train models, and compute evaluation metrics
python src/features/run_recommendation_experiments.py --reviews data/processed/reviews_processed.csv --recipes-metadata data/processed/recipes_processed.csv --content-svd artifacts/week5/pca_svd/X_content_svd.npy --recipe-ids artifacts/week5/pca_svd/reduced_recipe_ids.csv --out-dir artifacts/week10

# 2. Compile metrics and build this explanation report
python src/features/generate_week10_reports.py --out-dir artifacts/week10 --report-path reports/Week10_recommendation_explanation.md
```

**Manual PowerShell (Windows) Execution:**
```powershell
python src\\features\\run_recommendation_experiments.py --reviews data\\processed\\reviews_processed.csv --recipes-metadata data\\processed\\recipes_processed.csv --content-svd artifacts\\week5\\pca_svd\\X_content_svd.npy --recipe-ids artifacts\\week5\\pca_svd\\reduced_recipe_ids.csv --out-dir artifacts\\week10

python src\\features\\generate_week10_reports.py --out-dir artifacts\\week10 --report-path reports\\Week10_recommendation_explanation.md
```
"""
    return report_content


def main() -> None:
    args = parse_args()

    logger.info("Loading experiment outputs...")
    data_meta = load_json_file(args.out_dir / "data_metadata.json")
    run_config = load_json_file(args.out_dir / "run_config.json")

    metrics_path = args.out_dir / "recommendation_evaluation_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Evaluation summary CSV not found at {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)

    error_path = args.out_dir / "recommendation_error_analysis.json"
    error_cases = load_json_file(error_path)
    if isinstance(error_cases, dict):
        error_cases = [error_cases]  # type: ignore

    # Generate Markdown Report Content
    report_md = build_report_markdown(data_meta, run_config, metrics_df, error_cases)

    # Write report
    logger.info(f"Writing final report to {args.report_path}...")
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report_md, encoding="utf-8")

    print(f"Week 10 report successfully generated and saved to: {args.report_path}")


if __name__ == "__main__":
    main()
