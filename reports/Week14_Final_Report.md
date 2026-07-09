# Week 14: Final Integrated Delivery and Defense

## 1. Problem Statement
The goal of this project is to build a "domain discovery, recommendation, and graph intelligence" system for recipes using the Food.com dataset. This system helps users answer: What recipes are similar? What latent segments exist? What to cook next? Which ingredients are structurally central to this dataset?

## 2. Domain Context
Food.com is a large recipe-sharing platform where users submit recipes, rate them, and leave reviews. The domain combines structured ingredients with unstructured instructions and user interactions.

## 3. Dataset Sources and Access Conditions
The dataset is the public Food.com dataset (formerly GeniusKitchen) available on Kaggle. It is an open dataset containing recipes and user interactions up to 2020.
Access condition: Open access, public domain research release.

## 4. Schema and Data Dictionary
- **Recipes Table**: `RecipeId`, `Name`, `AuthorId`, `RecipeCategory`, `RecipeIngredientParts`, `Calories`, `ReviewCount`.
- **Reviews Table**: `ReviewId`, `RecipeId`, `AuthorId`, `Rating`, `DateSubmitted`.
Entity grain: One row per recipe / One row per user review.

## 5. Preprocessing and Feature Engineering
We extracted TF-IDF from ingredients (`content_tf_idf_matrix`) and scaled numeric fields like Calories, Fat, and Sodium (`numeric_matrix`). Features were resolved and normalized through `build_resolved_features.py` to handle outliers and missing values.

## 6. Dimensionality and Representation Analysis
We applied Truncated SVD to the TF-IDF matrix and PCA to the numeric matrix to reduce sparsity. SVD components captured primary culinary dimensions (e.g., baking vs savory, specific cuisines), and PCA captured caloric/nutritional density. This generated our dense representation for clustering and recommendation.

## 7. Clustering Analysis
Using K-Means on the concatenated SVD+PCA representations, we segmented recipes into stable clusters. Validated with inertia and silhouette scores, the clusters successfully represented distinct culinary profiles (e.g., low-calorie meals, high-sugar baked goods).

## 8. Recommendation System
We built a collaborative filtering model using Sparse SVD matrix factorization on the user-recipe interaction graph (5-core filtered). We compared this against a Bayesian popularity baseline. A hybrid recommendation approach successfully combined collaborative scores with content similarity.

## 9. Graph Analytics
We modeled an undirected, weighted Ingredient-Ingredient co-occurrence graph. 
Centrality metrics (PageRank, PPMI weighted degree) revealed the backbone of the dataset (e.g., salt, butter, sugar) while normalized Jaccard/PPMI highlighted distinctive flavor affinities, effectively answering structural questions about the domain's culinary composition.

## 10. Evaluation Protocol
- **Clustering:** Inertia and Silhouette scores.
- **Recommendation:** Chronological 80/20 train/test split. Evaluated using HitRate@10 and NDCG@10 via sampled negatives.

## 11. Pipeline and Reproducibility
The entire pipeline is orchestratable via `run_all_pipeline.ps1`, spanning ingestion, feature building, clustering, recommendation experiments, and graph analytics. There is no hidden notebook state. The project strictly adheres to the raw/interim/processed data lifecycle.

## 12. Ethics and Limitations
**Ethics and Access Note**: The dataset is public, but we ensured no direct PII is exposed. User IDs are anonymized numbers.
**Limitations**:
- Cold-start problem for new recipes.
- Ingredients are unweighted in the graph (quantity is ignored).
- Graph co-occurrence does not imply perfect culinary substitution.

## 13. Final Conclusions
The project successfully progressed from raw relational tables to a fully functional intelligence system. We proved that multi-modal data (ingredients, numeric macros, user ratings) can be unified into a robust recommender and that graph algorithms offer complementary insights into the composition of the dataset.

## 14. Monitoring and Operationalization Plan
If operationalized:
- **Drift Tracking**: Monitor rating distributions over time to track shift in user preferences.
- **Retraining**: Weekly retraining of the SVD model on the updated 5-core graph.
- **Logging**: Log predicted vs actual interactions.
