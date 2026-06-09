# Week 10 Recommendation, Ranking, or Predictive Decision Engine

## 1. Problem Classification & Project Nature

To clarify the structural role of modeling in this project, we define the mathematical nature of our task along the following spectrum:
- **Is this Recommendation?** Yes. We generate personalized ranked lists of recipe suggestions tailored to individual users based on their historical culinary interactions and taste preferences.
- **Is this Ranking?** Yes. Rather than predicting raw rating scalars in isolation, the core operational task is to order a pool of candidates from most relevant to least relevant for the user.
- **Is this Prediction?** Partially. We predict latent ratings and similarities to score candidates, but these scores are intermediate representations used to construct the final ranking.
- **Is this Segmentation feeding Ranking?** Yes, in an integrated catalog architecture. The unsupervised recipe clusters discovered in Week 7 act as structured candidate pools (e.g., separating "quick beverages" from "slow savory dinners"). These partitions can feed our ranking engine to allow cluster-aware recommendations and diversification.

In summary, the project is a **hybrid recommendation and ranking engine**. The technical goal is to rank candidate recipes for a given user, balancing collaborative user behavior and content semantics.

---

## 2. Ingestion & Preprocessing (5-Core Dataset)

Due to the extreme sparsity of the Food.com dataset (99.9981% of cells in the user-item matrix are empty), standard collaborative filtering models can suffer from high variance and computational instability. To ensure robust model training and stable offline evaluation, we filter the dataset to a **5-Core subset** (retaining only users with $\ge 5$ reviews and recipes with $\ge 5$ reviews). 

The table below summarizes the data shapes across the pipeline stages:

| Dataset Split / Stage | Unique Users | Unique Recipes | Interaction Rows | Sparsity |
| --- | ---: | ---: | ---: | ---: |
| Raw Ingested | 271,907 | 271,678 | 1,401,982 | 99.9981% |
| Filtered (5-Core) | 21,792 | 50,835 | 719,687 | 99.9350% |
| Training Set (80%) | 21,792 | - | 567,342 | - |
| Testing Set (20%) | 21,792 | - | 152,345 | - |

The 5-Core filtering yields a high-quality interaction matrix of 783,379 reviews across 27,626 active users and 64,457 recipes. This size is non-trivial but computationally efficient to process on standard CPU environments.

---

## 3. Description of Recommendation Systems

We implemented four recommendation models to compare baseline heuristics, collaborative filtering, and hybrid architectures:

### A. Popularity Baseline (Bayesian Average Rating)
- **Concept**: A non-personalized recommender that ranks recipes based on their global appeal while correcting for low review counts.
- **Formula**: For each recipe, the score is calculated as:
  $$ \text{Score} = \frac{v}{v + m} \cdot R + \frac{m}{v + m} \cdot C $$
  where $v$ is the review count, $R$ is the raw average rating of the recipe, $C$ is the global average rating across the training set (approx. 4.6), and $m$ is a smoothing constant set to $10.0$.
- **Role**: Serves as a robust, non-personalized cold-start baseline. It prevents recipes with a single 5-star review from outranking highly popular recipes with hundreds of high-quality reviews.

### B. Content-Based SVD Baseline
- **Concept**: A personalized recommender that relies solely on recipe content profiles.
- **Methodology**: 
  1. We leverage the 200-dimensional Truncated SVD semantic embeddings (`X_content_svd.npy`) generated in Week 5.
  2. For each user, we construct a taste profile vector by averaging the SVD content vectors of recipes they rated $\ge 4.0$ in the training set.
  3. The user taste profile and recipe content vectors are normalized to unit length.
  4. The content score for a recipe is computed as the cosine similarity (vector dot product) between the user profile and the recipe vector:
     $$ \text{Score}_{Content}(u, i) = \vec{u}_{profile} \cdot \vec{i}_{svd\_normalized} $$
- **Role**: Serves as a personalized content baseline, recommending recipes that have similar ingredients and text keywords to what the user historically liked.

### C. Collaborative SVD (Stronger System)
- **Concept**: Matrix factorization collaborative filtering that captures latent behavioral dimensions.
- **Methodology**:
  1. We map users and recipes to sparse matrix coordinates.
  2. We compute the average training rating for each user and perform mean-centering (subtracting user means) to remove user rating biases.
  3. We construct a sparse user-item interaction matrix $R_{ui}$ and decompose it using sparse Singular Value Decomposition (`scipy.sparse.linalg.svds`) with $K=50$ latent factors.
  4. User and recipe latent matrices are scaled by the square root of the singular values:
     $$ U_{scaled} = U \cdot \Sigma^{0.5}, \quad V_{scaled} = V \cdot \Sigma^{0.5} $$
  5. The prediction score for user $u$ and recipe $i$ is reconstructed as:
     $$ \hat{R}_{ui} = \text{user\_mean}_u + \vec{u}_{factors} \cdot \vec{i}_{factors} $$
- **Role**: Captures complex behavioral similarities ("users who liked X also liked Y") independent of explicit recipe text or ingredients.

### D. Hybrid Recommender (Advanced Blended System)
- **Concept**: Blends Collaborative SVD and Content-Based SVD scores to combine the benefits of behavioral patterns and semantic content.
- **Data Alignment & Normalization**:
  Collaborative predictions and cosine similarities operate on different mathematical scales. To align them, for each user's candidate pool, we perform Min-Max Normalization to scale both CF and Content scores to the range $[0, 1]$:
  $$ \text{Score}_{norm} = \frac{\text{Score} - \text{Score}_{min}}{\text{Score}_{max} - \text{Score}_{min} + 1e-9} $$
  We then perform a weighted linear blend using parameter $\alpha=0.6$:
  $$ \text{Score}_{hybrid} = \alpha \cdot \text{Score}_{CF\_norm} + (1 - \alpha) \cdot \text{Score}_{Content\_norm} $$
- **Role**: Our advanced recommendation model. The collaborative component provides high accuracy for active users, while the content component anchors the recommendation to the user's culinary vocabulary (ingredients/cuisine) and stabilizes recommendations when rating data is sparse.

---

## 4. Offline Evaluation Protocol & Report

### Evaluation Protocol (Sampled Negatives)
To evaluate the models rigorously and realistically:
1. **Chronological Splitting**: We split interactions chronologically per user (oldest 80% to train, newest 20% to test). This evaluates the system's ability to predict *future* interactions based on *past* behaviors, avoiding the data leakage inherent in random splitting.
2. **Candidate Pool Definition**: Evaluating all 64.4k recipes for every test rating is computationally prohibitive. We adopt the standard **Sampled Metrics (K-negative) Protocol**. For each test review of an evaluated user, we construct a candidate pool containing:
   - The **1 target recipe** that the user actually interacted with in the test set (the positive item).
   - **100 random recipes** that the user has not interacted with in either the train or test sets (negative items).
3. **Evaluation Sample**: We evaluate metrics across a representative, deterministic random sample of 2,000 users in the test set.

### Offline Evaluation Report

The table below summarizes the performance of the four models:

| Recommendation Model | Hit Rate @ 5 (HR@5) | Hit Rate @ 10 (HR@10) | NDCG @ 5 | NDCG @ 10 | MRR |
| --- | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline (Bayesian)** | 0.0835 | 0.1373 | 0.0544 | 0.0717 | 0.0726 |
| **Collaborative SVD (Stronger)** | 0.1418 | 0.2074 | 0.0977 | 0.1188 | 0.1108 |
| **Content-Based SVD (Baseline)** | 0.1159 | 0.1999 | 0.0715 | 0.0984 | 0.0922 |
| **Hybrid Recommender (CF + Content)** | 0.1440 | 0.2250 | 0.0978 | 0.1238 | 0.1154 |

### Metrics Interpretation & Key Findings
1. **Collaborative SVD Outperforms Baselines**: The Collaborative SVD model achieves a substantial improvement in accuracy over both the non-personalized Popularity baseline and the pure Content-Based baseline. This confirms that collaborative behavioral signals ("who liked what") are far more predictive of future interactions than pure recipe ingredient overlap.
2. **Hybrid Model Achieves the Highest Performance**: The Hybrid model (blending 60% Collaborative SVD and 40% Content SVD) achieves the best overall performance, outperforming pure Collaborative Filtering. By incorporating SVD semantic representations, the Hybrid model is able to refine behavioral scores with ingredient semantic proximity, showing that content acts as a useful regularizer for collaborative filtering.
3. **Content-Based SVD performs moderately**: While lower than Collaborative Filtering, the pure Content-Based SVD baseline performs significantly better than random guessing (which would yield a Hit Rate @ 10 of $10 / 101 \approx 0.099$). This confirms that the 200-dimensional semantic space constructed in Week 5 contains real, predictive representations of user culinary tastes.
4. **Popularity Baseline shows low personalization**: The Bayesian popularity average performs poorly on personalized retrieval. This is expected, as recommending general popular items does not align with the highly customized taste profiles of individual home cooks.

---

## 5. Error Analysis & Diagnostics

Analyzing specific cases helps diagnose the strengths and failures of the Hybrid recommendation system. The following sections analyze real cases from the test evaluation:

### A. Strong Cases (Successful Recommendations)
These are cases where a highly rated test recipe (Rating 4.0 or 5.0) was successfully ranked at the top spot (Rank #1 out of 101) by the Hybrid model:


            #### Case 1: Jim's Microwave Scrambled Eggs
            - **User ID (AuthorId)**: `178117` | **Recipe ID (RecipeId)**: `131426`
            - **Actual Rating Given**: 5.0 / 5.0
            - **Ranks**:
            - Popularity Rank: `82` / 101
            - Content-SVD Rank: `25` / 101
            - Collaborative-SVD Rank: `2` / 101
            - **Hybrid Rank**: `1` / 101 (Successfully Recommended in #1 spot!)
            - **Scores**:
            - CF Raw Score: `4.0844` | Content Similarity: `0.2169` | Blended Hybrid Score: `0.7714`
            - **Culinary / Behavior Diagnosis**: Success driven primarily by the collaborative signal (CF rank #2, raw score 4.0844). The content similarity was moderate (0.2169, rank #25), suggesting the recipe shares behavioral fans with items this user liked, even if its ingredient keywords are not an exact profile match. The hybrid blend (α=0.6 toward CF) correctly up-ranked it.
            
            #### Case 2: Ghostly Green Brew
            - **User ID (AuthorId)**: `293001` | **Recipe ID (RecipeId)**: `188331`
            - **Actual Rating Given**: 5.0 / 5.0
            - **Ranks**:
            - Popularity Rank: `30` / 101
            - Content-SVD Rank: `1` / 101
            - Collaborative-SVD Rank: `15` / 101
            - **Hybrid Rank**: `1` / 101 (Successfully Recommended in #1 spot!)
            - **Scores**:
            - CF Raw Score: `4.7778` | Content Similarity: `0.4107` | Blended Hybrid Score: `0.8598`
            - **Culinary / Behavior Diagnosis**: Success driven primarily by content semantics (content rank #1, similarity 0.4107). The CF signal was weaker (rank #15, score 4.7778), but the recipe's ingredient and keyword space closely mirrors the user's historical taste profile. The content component (40% weight) was sufficient to pull the hybrid score above the candidate pool.
            
            #### Case 3: Kittencal's Spinach &amp; Four-Cheese Manicotti (Vegetarian)
            - **User ID (AuthorId)**: `135887` | **Recipe ID (RecipeId)**: `72308`
            - **Actual Rating Given**: 4.0 / 5.0
            - **Ranks**:
            - Popularity Rank: `61` / 101
            - Content-SVD Rank: `47` / 101
            - Collaborative-SVD Rank: `2` / 101
            - **Hybrid Rank**: `1` / 101 (Successfully Recommended in #1 spot!)
            - **Scores**:
            - CF Raw Score: `4.1237` | Content Similarity: `0.2461` | Blended Hybrid Score: `0.7631`
            - **Culinary / Behavior Diagnosis**: Success driven primarily by the collaborative signal (CF rank #2, raw score 4.1237). The content similarity was moderate (0.2461, rank #47), suggesting the recipe shares behavioral fans with items this user liked, even if its ingredient keywords are not an exact profile match. The hybrid blend (α=0.6 toward CF) correctly up-ranked it.
            
            #### Case 4: Yummy Baked Potato Skins
            - **User ID (AuthorId)**: `135887` | **Recipe ID (RecipeId)**: `43908`
            - **Actual Rating Given**: 5.0 / 5.0
            - **Ranks**:
            - Popularity Rank: `1` / 101
            - Content-SVD Rank: `1` / 101
            - Collaborative-SVD Rank: `14` / 101
            - **Hybrid Rank**: `1` / 101 (Successfully Recommended in #1 spot!)
            - **Scores**:
            - CF Raw Score: `4.0736` | Content Similarity: `0.4993` | Blended Hybrid Score: `0.7212`
            - **Culinary / Behavior Diagnosis**: Success driven primarily by content semantics (content rank #1, similarity 0.4993). The CF signal was weaker (rank #14, score 4.0736), but the recipe's ingredient and keyword space closely mirrors the user's historical taste profile. The content component (40% weight) was sufficient to pull the hybrid score above the candidate pool.
            
            #### Case 5: Marshall Field's Chicken Salad (With Sandwich Variations)
            - **User ID (AuthorId)**: `135887` | **Recipe ID (RecipeId)**: `115767`
            - **Actual Rating Given**: 5.0 / 5.0
            - **Ranks**:
            - Popularity Rank: `82` / 101
            - Content-SVD Rank: `16` / 101
            - Collaborative-SVD Rank: `1` / 101
            - **Hybrid Rank**: `1` / 101 (Successfully Recommended in #1 spot!)
            - **Scores**:
            - CF Raw Score: `4.0903` | Content Similarity: `0.3227` | Blended Hybrid Score: `0.8833`
            - **Culinary / Behavior Diagnosis**: Success driven primarily by the collaborative signal (CF rank #1, raw score 4.0903). The content similarity was moderate (0.3227, rank #16), suggesting the recipe shares behavioral fans with items this user liked, even if its ingredient keywords are not an exact profile match. The hybrid blend (α=0.6 toward CF) correctly up-ranked it.
            

### B. Failure Cases (Low Rank)
These are cases where a highly rated test recipe (Rating 4.0 or 5.0) failed to be recommended, receiving a rank of 50 or worse by the Hybrid model:


        #### Case 1: Southern Cinnamon Sugared Pecans
        - **User ID (AuthorId)**: `1443141` | **Recipe ID (RecipeId)**: `323372`
        - **Actual Rating Given**: 5.0 / 5.0
        - **Ranks**:
        - Popularity Rank: `8` / 101
        - Content-SVD Rank: `86` / 101
        - Collaborative-SVD Rank: `66` / 101
        - **Hybrid Rank**: `85` / 101 (Failed to rank in Top 50!)
        - **Scores**:
        - CF Raw Score: `4.4545` | Content Similarity: `0.0621` | Blended Hybrid Score: `0.1649`
        - **Culinary / Behavior Diagnosis**: Double-signal failure: both the collaborative model (rank #66) and the content model (similarity 0.0621, rank #86) score this recipe poorly for this user. The recipe likely represents a taste excursion outside the user's established culinary profile — neither their behavioral neighbors nor their ingredient history anticipate this preference. A serendipity or novelty component would be needed to surface it.
        
        #### Case 2: Pete's Scratch Pancakes
        - **User ID (AuthorId)**: `1443141` | **Recipe ID (RecipeId)**: `5170`
        - **Actual Rating Given**: 5.0 / 5.0
        - **Ranks**:
        - Popularity Rank: `24` / 101
        - Content-SVD Rank: `21` / 101
        - Collaborative-SVD Rank: `101` / 101
        - **Hybrid Rank**: `101` / 101 (Failed to rank in Top 50!)
        - **Scores**:
        - CF Raw Score: `4.4480` | Content Similarity: `0.2297` | Blended Hybrid Score: `0.2610`
        - **Culinary / Behavior Diagnosis**: The content signal shows moderate relevance (rank #21, similarity 0.2297), but the collaborative signal fails badly (rank #101, raw score 4.4480). This recipe lacks sufficient rating density from behavioral neighbors in the training set — a sparse-CF failure. Boosting the content weight (increasing 1−α) for users with sparse collaborative coverage would mitigate this.
        
        #### Case 3: Hummus
        - **User ID (AuthorId)**: `394144` | **Recipe ID (RecipeId)**: `11424`
        - **Actual Rating Given**: 5.0 / 5.0
        - **Ranks**:
        - Popularity Rank: `83` / 101
        - Content-SVD Rank: `99` / 101
        - Collaborative-SVD Rank: `71` / 101
        - **Hybrid Rank**: `96` / 101 (Failed to rank in Top 50!)
        - **Scores**:
        - CF Raw Score: `4.8889` | Content Similarity: `0.0509` | Blended Hybrid Score: `0.4922`
        - **Culinary / Behavior Diagnosis**: Double-signal failure: both the collaborative model (rank #71) and the content model (similarity 0.0509, rank #99) score this recipe poorly for this user. The recipe likely represents a taste excursion outside the user's established culinary profile — neither their behavioral neighbors nor their ingredient history anticipate this preference. A serendipity or novelty component would be needed to surface it.
        
        #### Case 4: Diet Soup
        - **User ID (AuthorId)**: `498829` | **Recipe ID (RecipeId)**: `21892`
        - **Actual Rating Given**: 5.0 / 5.0
        - **Ranks**:
        - Popularity Rank: `88` / 101
        - Content-SVD Rank: `49` / 101
        - Collaborative-SVD Rank: `96` / 101
        - **Hybrid Rank**: `67` / 101 (Failed to rank in Top 50!)
        - **Scores**:
        - CF Raw Score: `4.9615` | Content Similarity: `0.1997` | Blended Hybrid Score: `0.5007`
        - **Culinary / Behavior Diagnosis**: Both signals are weak for this pair (CF rank #96, content rank #49). The hybrid score places it at rank #67/101 despite a 5.0-star rating. This is consistent with a sparse-interaction cold-start edge case where the user has explored a recipe category underrepresented in their training history, making the failure hard to avoid without explicit user-declared preference signals.
        
        #### Case 5: Pumpkin Dog Cookies
        - **User ID (AuthorId)**: `498829` | **Recipe ID (RecipeId)**: `133062`
        - **Actual Rating Given**: 5.0 / 5.0
        - **Ranks**:
        - Popularity Rank: `1` / 101
        - Content-SVD Rank: `97` / 101
        - Collaborative-SVD Rank: `61` / 101
        - **Hybrid Rank**: `93` / 101 (Failed to rank in Top 50!)
        - **Scores**:
        - CF Raw Score: `4.9615` | Content Similarity: `0.0743` | Blended Hybrid Score: `0.4325`
        - **Culinary / Behavior Diagnosis**: Double-signal failure: both the collaborative model (rank #61) and the content model (similarity 0.0743, rank #97) score this recipe poorly for this user. The recipe likely represents a taste excursion outside the user's established culinary profile — neither their behavioral neighbors nor their ingredient history anticipate this preference. A serendipity or novelty component would be needed to surface it.
        

### C. Failure Cases (Disliked Recommended)
These are cases where a user actually disliked a recipe in the test set (rating it 1.0 or 2.0 stars), but the Hybrid model incorrectly recommended it in the top 5 list:


        #### Case 1: The Best Brownies
        - **User ID (AuthorId)**: `1443141` | **Recipe ID (RecipeId)**: `54225`
        - **Actual Rating Given**: 2.0 / 5.0 (User DISLIKED this recipe)
        - **Ranks**:
        - Popularity Rank: `50` / 101
        - Content-SVD Rank: `33` / 101
        - Collaborative-SVD Rank: `1` / 101
        - **Hybrid Rank**: `1` / 101 (Incorrectly Recommended in Top 5!)
        - **Scores**:
        - CF Raw Score: `4.4585` | Content Similarity: `0.1895` | Blended Hybrid Score: `0.7920`
        - **Culinary / Behavior Diagnosis**: Collaborative false positive (CF rank #1, score 4.4585): the user's behavioral neighbors strongly liked this recipe, but this individual user did not. This is a classic CF overfitting case — the model captures the majority signal from similar users but misses the personal negative preference. Content similarity was modest (0.1895, rank #33), so the hybrid's α=0.6 CF weight carried the false recommendation forward.
        
        #### Case 2: Kittencal's Strawberry Shortcake
        - **User ID (AuthorId)**: `135887` | **Recipe ID (RecipeId)**: `223104`
        - **Actual Rating Given**: 2.0 / 5.0 (User DISLIKED this recipe)
        - **Ranks**:
        - Popularity Rank: `1` / 101
        - Content-SVD Rank: `18` / 101
        - Collaborative-SVD Rank: `5` / 101
        - **Hybrid Rank**: `5` / 101 (Incorrectly Recommended in Top 5!)
        - **Scores**:
        - CF Raw Score: `4.0920` | Content Similarity: `0.3186` | Blended Hybrid Score: `0.5070`
        - **Culinary / Behavior Diagnosis**: Mixed-signal false positive: hybrid rank #5 despite the user rating it 2.0/5. CF score 4.0920 (rank #5) and content similarity 0.3186 (rank #18) both contribute a moderate positive signal. The combination crosses the recommendation threshold even though individually neither signal is dominant. Adding an explicit negative feedback mechanism or a dislike-aware regularization term would suppress these cases.
        
        #### Case 3: Magnolia Bakery Vanilla Cupcakes
        - **User ID (AuthorId)**: `1164770` | **Recipe ID (RecipeId)**: `133767`
        - **Actual Rating Given**: 1.0 / 5.0 (User DISLIKED this recipe)
        - **Ranks**:
        - Popularity Rank: `85` / 101
        - Content-SVD Rank: `3` / 101
        - Collaborative-SVD Rank: `36` / 101
        - **Hybrid Rank**: `5` / 101 (Incorrectly Recommended in Top 5!)
        - **Scores**:
        - CF Raw Score: `4.2501` | Content Similarity: `0.3378` | Blended Hybrid Score: `0.5653`
        - **Culinary / Behavior Diagnosis**: Content false positive (content rank #3, similarity 0.3378): the recipe's ingredient and keyword profile closely matches the user's taste vector, but the user disliked it in practice. The CF signal was weaker (rank #36), suggesting the broader user community also doesn't strongly favor this recipe. The content component (40% weight) over-contributed to a spurious recommendation.
        
        #### Case 4: Crock Pot Cream Cheese Chicken
        - **User ID (AuthorId)**: `722619` | **Recipe ID (RecipeId)**: `12458`
        - **Actual Rating Given**: 0.0 / 5.0 (User DISLIKED this recipe)
        - **Ranks**:
        - Popularity Rank: `76` / 101
        - Content-SVD Rank: `3` / 101
        - Collaborative-SVD Rank: `2` / 101
        - **Hybrid Rank**: `1` / 101 (Incorrectly Recommended in Top 5!)
        - **Scores**:
        - CF Raw Score: `4.5561` | Content Similarity: `0.3775` | Blended Hybrid Score: `0.9434`
        - **Culinary / Behavior Diagnosis**: Strong false positive: both CF (rank #2, score 4.5561) and content (rank #3, similarity 0.3775) agree this is a good recommendation, yet the user disliked it. This points to a latent preference dimension not captured by either signal — possibly a specific texture, technique, or ingredient sub-component (e.g. an allergen, a disliked spice) that the SVD embedding collapses into a broader positive cluster.
        
        #### Case 5: Japanese Mum's Chicken
        - **User ID (AuthorId)**: `2148404` | **Recipe ID (RecipeId)**: `68955`
        - **Actual Rating Given**: 0.0 / 5.0 (User DISLIKED this recipe)
        - **Ranks**:
        - Popularity Rank: `91` / 101
        - Content-SVD Rank: `4` / 101
        - Collaborative-SVD Rank: `1` / 101
        - **Hybrid Rank**: `1` / 101 (Incorrectly Recommended in Top 5!)
        - **Scores**:
        - CF Raw Score: `3.5162` | Content Similarity: `0.2527` | Blended Hybrid Score: `0.9103`
        - **Culinary / Behavior Diagnosis**: Strong false positive: both CF (rank #1, score 3.5162) and content (rank #4, similarity 0.2527) agree this is a good recommendation, yet the user disliked it. This points to a latent preference dimension not captured by either signal — possibly a specific texture, technique, or ingredient sub-component (e.g. an allergen, a disliked spice) that the SVD embedding collapses into a broader positive cluster.
        

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
python src\features\build_resolved_features.py --recipes data\processed\recipes_processed.csv --out data\interim\recipes_resolved_features.parquet --summary-out artifacts\week5

python src\features\build_numeric_matrix.py --recipes data\interim\recipes_resolved_features.parquet --out artifacts\week5

python src\features\build_content_matrix.py --recipes data\interim\recipes_resolved_features.parquet --out artifacts\week5\content_tf_idf_matrix --numeric-recipe-ids artifacts\week5\numeric_matrix_outputs\recipe_ids.csv

python src\features\reduce_dimensions.py --numeric-matrix artifacts\week5\numeric_matrix_outputs\X_numeric_scaled.npy --numeric-feature-names artifacts\week5\numeric_matrix_outputs\numeric_feature_names.csv --numeric-recipe-ids artifacts\week5\numeric_matrix_outputs\recipe_ids.csv --content-matrix artifacts\week5\content_tf_idf_matrix\X_content_tfidf.npz --content-feature-names artifacts\week5\content_tf_idf_matrix\content_feature_names.csv --content-recipe-ids artifacts\week5\content_tf_idf_matrix\content_recipe_ids.csv --out artifacts\week5\pca_svd --figures reports\figures
```

### Step B: Execute Week 10 Recommendation & Evaluation Pipeline
This step processes interactions, trains recommender configurations, evaluates performance metrics, extracts error cases, and generates the markdown documentation.

**PowerShell (Windows - Recommended Automated Runbook):**
Simply run the master pipeline script from the project root:
```powershell
.\run_week10_pipeline.ps1
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
python src\features\run_recommendation_experiments.py --reviews data\processed\reviews_processed.csv --recipes-metadata data\processed\recipes_processed.csv --content-svd artifacts\week5\pca_svd\X_content_svd.npy --recipe-ids artifacts\week5\pca_svd\reduced_recipe_ids.csv --out-dir artifacts\week10

python src\features\generate_week10_reports.py --out-dir artifacts\week10 --report-path reports\Week10_recommendation_explanation.md
```
