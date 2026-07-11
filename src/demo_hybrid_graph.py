#!/usr/bin/env python3
"""
src/demo_hybrid_graph.py
Interactive CLI demo combining Graph Analytics and the Hybrid Recommendation Model (Option 3).

Workflow:
1. Load Graph edges (PPMI/Jaccard weights) and SVD/CF models.
2. Select or input a User ID and a seed ingredient (e.g., 'chicken', 'fish', 'chocolate').
3. Graph expansion: Find the top 5 related ingredients using PPMI/Jaccard edge weights.
4. Candidate generation: Filter recipes containing the seed ingredient + at least one related ingredient.
5. Hybrid SVD Ranking: Rank candidate recipes for the selected user combining:
   - Collaborative SVD rating prediction (trained on 5-core reviews).
   - Content SVD cosine similarity (user profile vs recipe content embeddings).
6. Display Top 5 personalized recipes.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
REVIEWS_PATH = Path("data/processed/reviews_processed.csv")
RECIPES_PATH = Path("data/processed/recipes_processed.csv")
GRAPH_EDGES_PATH = Path("artifacts/week12/ingredient_graph/ingredient_edges.csv")
CONTENT_SVD_PATH = Path("artifacts/week5/pca_svd/X_content_svd.npy")
RECIPE_IDS_PATH = Path("artifacts/week5/pca_svd/reduced_recipe_ids.csv")

CF_MODEL_PATH = Path("artifacts/week10/svd_cf_model.npz")
CF_MAPPINGS_PATH = Path("artifacts/week10/svd_cf_mappings.json")

def parse_list(cell):
    """Convert a string representation of a list into a Python list."""
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    if s.startswith('c(') and s.endswith(')'):
        s = s[2:-1]
    try:
        s_normalised = s.replace('\\"', '"').replace('\"', '"')
        if not (s_normalised.startswith('[') or s_normalised.startswith('(')):
            s_normalised = f'[{s_normalised}]'
        value = ast.literal_eval(s_normalised)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]
    except Exception:
        return [v.strip().strip('"') for v in s.split(',') if v]

def get_5_core(df, k=5):
    logger.info(f"Filtering reviews dataset to {k}-core...")
    df_core = df.copy()
    while True:
        u_counts = df_core["AuthorId"].value_counts()
        r_counts = df_core["RecipeId"].value_counts()
        keep_u = u_counts[u_counts >= k].index
        keep_r = r_counts[r_counts >= k].index
        filtered = df_core[df_core["AuthorId"].isin(keep_u) & df_core["RecipeId"].isin(keep_r)]
        if len(filtered) == len(df_core):
            break
        df_core = filtered
    return df_core.reset_index(drop=True)

class GraphHelper:
    def __init__(self, edges_path: Path):
        logger.info(f"Loading Graph edges from {edges_path}...")
        self.edges_df = pd.read_csv(edges_path)
        logger.info(f"Loaded {len(self.edges_df)} edges.")
        
    def find_related_ingredients(self, seed_ingredient: str, top_n: int = 5, metric: str = "ppmi") -> list[tuple[str, float]]:
        """Find the top_n associated ingredients using ppmi or jaccard weights."""
        seed = seed_ingredient.strip().lower().replace(" ", "_")
        
        # Look for matches where seed is source or target
        matches = self.edges_df[
            (self.edges_df["source"] == seed) | (self.edges_df["target"] == seed)
        ].copy()
        
        if matches.empty:
            return []
            
        # Determine the neighbor name and weight
        related = []
        for _, row in matches.iterrows():
            neighbor = row["target"] if row["source"] == seed else row["source"]
            weight = row[metric]
            related.append((neighbor, weight))
            
        # Sort and return top_n
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:top_n]

class RecommendationEngine:
    def __init__(self, n_factors=50, alpha=0.6):
        self.n_factors = n_factors
        self.alpha = alpha
        self.user_to_idx = {}
        self.recipe_to_idx = {}
        self.user_means = np.array([])
        self.user_factors = np.array([])
        self.recipe_factors = np.array([])
        self.global_mean = 4.0
        
        # Content model fields
        self.X_content_norm = None
        self.recipe_to_svd_idx = {}
        self.user_profiles = {}
        
    def load_content_embeddings(self, content_path: Path, recipe_ids_path: Path):
        logger.info(f"Loading content embeddings from {content_path}...")
        X_content = np.load(content_path)
        recipe_ids_df = pd.read_csv(recipe_ids_path)
        
        if "row_index" in recipe_ids_df.columns:
            self.recipe_to_svd_idx = {
                int(row["RecipeId"]): int(row["row_index"]) for _, row in recipe_ids_df.iterrows()
            }
        else:
            self.recipe_to_svd_idx = {
                int(row["RecipeId"]): int(idx) for idx, row in recipe_ids_df.iterrows()
            }
            
        norms = np.linalg.norm(X_content, axis=1, keepdims=True)
        self.X_content_norm = X_content / (norms + 1e-9)
        logger.info(f"Loaded content SVD for {len(self.recipe_to_svd_idx)} recipes.")

    def fit_or_load_cf(self, train_df: pd.DataFrame):
        if CF_MODEL_PATH.exists() and CF_MAPPINGS_PATH.exists():
            logger.info("Loading Collaborative Filtering SVD model from cache...")
            data = np.load(CF_MODEL_PATH)
            self.user_factors = data["user_factors"]
            self.recipe_factors = data["recipe_factors"]
            self.user_means = data["user_means"]
            self.global_mean = float(data["global_mean"][0])
            
            with open(CF_MAPPINGS_PATH, "r") as f:
                mappings = json.load(f)
            self.user_to_idx = {int(k): int(v) for k, v in mappings["user_to_idx"].items()}
            self.recipe_to_idx = {int(k): int(v) for k, v in mappings["recipe_to_idx"].items()}
            logger.info("CF model loaded successfully.")
        else:
            logger.info("CF model cache not found. Training Collaborative SVD on-the-fly...")
            self.global_mean = float(train_df["Rating"].mean())
            
            unique_users = train_df["AuthorId"].unique()
            unique_recipes = train_df["RecipeId"].unique()
            
            self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
            self.recipe_to_idx = {rid: idx for idx, rid in enumerate(unique_recipes)}
            
            n_users = len(unique_users)
            n_recipes = len(unique_recipes)
            
            u_indices = np.array([self.user_to_idx[uid] for uid in train_df["AuthorId"]])
            r_indices = np.array([self.recipe_to_idx[rid] for rid in train_df["RecipeId"]])
            ratings = train_df["Rating"].values
            
            user_ratings_sum = np.zeros(n_users)
            user_ratings_count = np.zeros(n_users)
            for u, r_val in zip(u_indices, ratings):
                user_ratings_sum[u] += r_val
                user_ratings_count[u] += 1
                
            self.user_means = user_ratings_sum / (user_ratings_count + 1e-9)
            centered_ratings = ratings - self.user_means[u_indices]
            
            R = sp.coo_matrix(
                (centered_ratings, (u_indices, r_indices)),
                shape=(n_users, n_recipes),
                dtype=np.float32,
            ).tocsr()
            
            k = min(self.n_factors, min(n_users, n_recipes) - 2)
            U, s, Vt = svds(R, k=k)
            
            sort_idx = np.argsort(s)[::-1]
            U = U[:, sort_idx]
            s = s[sort_idx]
            Vt = Vt[sort_idx, :]
            
            sqrt_s = np.sqrt(s)
            self.user_factors = U * sqrt_s
            self.recipe_factors = Vt.T * sqrt_s
            
            # Save cache
            logger.info(f"Saving SVD CF model to {CF_MODEL_PATH}...")
            CF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez(CF_MODEL_PATH, user_factors=self.user_factors, recipe_factors=self.recipe_factors, user_means=self.user_means, global_mean=np.array([self.global_mean]))
            mappings = {
                "user_to_idx": {int(k): int(v) for k, v in self.user_to_idx.items()},
                "recipe_to_idx": {int(k): int(v) for k, v in self.recipe_to_idx.items()}
            }
            with open(CF_MAPPINGS_PATH, "w") as f:
                json.dump(mappings, f)
                
            logger.info("CF model trained and cached successfully.")

    def fit_content_profiles(self, train_df: pd.DataFrame):
        logger.info("Building User Content SVD taste profiles...")
        liked_df = train_df[train_df["Rating"] >= 4.0]
        user_grouped = liked_df.groupby("AuthorId")
        fallback_grouped = train_df.groupby("AuthorId")
        
        uids = train_df["AuthorId"].unique()
        dim = self.X_content_norm.shape[1]
        
        for uid in uids:
            rids = []
            if uid in user_grouped.groups:
                rids = liked_df.loc[user_grouped.groups[uid], "RecipeId"].values
            else:
                rids = train_df.loc[fallback_grouped.groups[uid], "RecipeId"].values
                
            svd_indices = [self.recipe_to_svd_idx[rid] for rid in rids if rid in self.recipe_to_svd_idx]
            if svd_indices:
                profile = self.X_content_norm[svd_indices].mean(axis=0)
                norm = np.linalg.norm(profile)
                self.user_profiles[uid] = profile / (norm + 1e-9)
            else:
                self.user_profiles[uid] = np.zeros(dim, dtype=np.float32)

    def predict_hybrid_rankings(self, user_id: int, candidate_recipe_ids: list[int]) -> list[tuple[int, float]]:
        """Score candidate recipes using Hybrid recommender."""
        scored = []
        u_idx = self.user_to_idx.get(user_id)
        user_profile = self.user_profiles.get(user_id)
        
        cf_scores = []
        content_scores = []
        valid_candidates = []
        
        for rid in candidate_recipe_ids:
            # Collaborative score
            r_idx = self.recipe_to_idx.get(rid)
            if u_idx is None or r_idx is None:
                cf_val = self.global_mean
            else:
                cf_val = self.user_means[u_idx] + np.dot(self.user_factors[u_idx], self.recipe_factors[r_idx])
                
            # Content score
            c_idx = self.recipe_to_svd_idx.get(rid)
            if user_profile is None or c_idx is None or np.all(user_profile == 0):
                content_val = 0.0
            else:
                content_val = float(np.dot(user_profile, self.X_content_norm[c_idx]))
                
            cf_scores.append(cf_val)
            content_scores.append(content_val)
            valid_candidates.append(rid)
            
        if not valid_candidates:
            return []
            
        # Min-max normalization for blending
        cf_arr = np.array(cf_scores)
        content_arr = np.array(content_scores)
        
        cf_min, cf_max = cf_arr.min(), cf_arr.max()
        if cf_max > cf_min:
            cf_norm = (cf_arr - cf_min) / (cf_max - cf_min)
        else:
            cf_norm = np.ones_like(cf_arr) * 0.5
            
        c_min, c_max = content_arr.min(), content_arr.max()
        if c_max > c_min:
            c_norm = (content_arr - c_min) / (c_max - c_min)
        else:
            c_norm = np.ones_like(content_arr) * 0.5
            
        hybrid_scores = self.alpha * cf_norm + (1.0 - self.alpha) * c_norm
        
        for rid, score in zip(valid_candidates, hybrid_scores):
            scored.append((rid, float(score)))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

def main():
    print("================================================================")
    print("      RECIPE INTELLIGENCE DEMO - GRAPH & HYBRID MODEL")
    print("================================================================\n")
    
    # 1. Load Data
    logger.info("Loading recipes dataset...")
    recipes = pd.read_csv(RECIPES_PATH)
    recipes["ParsedIngredients"] = recipes["RecipeIngredientParts"].apply(parse_list)
    
    logger.info("Loading reviews dataset...")
    reviews = pd.read_csv(REVIEWS_PATH)
    reviews_core = get_5_core(reviews, k=5)
    
    # 2. Initialize Models
    graph = GraphHelper(GRAPH_EDGES_PATH)
    recommender = RecommendationEngine(n_factors=50, alpha=0.6)
    recommender.load_content_embeddings(CONTENT_SVD_PATH, RECIPE_IDS_PATH)
    recommender.fit_or_load_cf(reviews_core)
    recommender.fit_content_profiles(reviews_core)
    
    # Select popular users to prompt the user
    top_users = reviews_core["AuthorId"].value_counts().head(5).index.tolist()
    print("\n----------------------------------------------------------------")
    print("SYSTEM LOADED SUCCESSFULLY!")
    print(f"Top 5 active User IDs in dataset: {top_users}")
    print("----------------------------------------------------------------\n")
    
    # Interactive loop
    while True:
        try:
            user_input_id = input(f"Enter User ID (press Enter to use default {top_users[0]}): ").strip()
            if not user_input_id:
                user_id = top_users[0]
            else:
                user_id = int(user_input_id)
                
            if user_id not in recommender.user_to_idx:
                print(f"[!] User ID {user_id} not found in the 5-core interactions list. Standard fallback will be applied.")
                
            seed_ingredient = input("Enter an ingredient search seed (e.g., 'chicken', 'fish', 'chocolate'): ").strip()
            if not seed_ingredient:
                seed_ingredient = "chicken"
                
            print(f"\n[Step 1] Querying Graph for related ingredients to: '{seed_ingredient}'...")
            related = graph.find_related_ingredients(seed_ingredient, top_n=5, metric="ppmi")
            
            if not related:
                print(f"[!] No related ingredients found in the graph for '{seed_ingredient}'.")
                related_names = []
            else:
                print("Top related ingredients found by PPMI co-occurrence in the Graph:")
                for neighbor, score in related:
                    print(f"  * {neighbor} (PPMI Association Score: {score:.4f})")
                related_names = [r[0] for r in related]
                
            # Filter Candidates
            print(f"\n[Step 2] Finding recipe candidates containing '{seed_ingredient}'...")
            # We want recipes that contain the seed ingredient
            seed_norm = seed_ingredient.strip().lower().replace(" ", "_")
            
            # Simple word-matching filter on parsed ingredients
            mask = recipes["ParsedIngredients"].apply(lambda x: any(seed_norm in ing.replace(" ", "_") for ing in x))
            candidates_df = recipes[mask].copy()
            
            # Now, filter or prioritize recipes that also have at least one related ingredient to narrow down
            if related_names:
                related_set = set(related_names)
                candidates_df["graph_overlap"] = candidates_df["ParsedIngredients"].apply(
                    lambda x: len(related_set.intersection([ing.replace(" ", "_") for ing in x]))
                )
                # Keep those with at least 1 overlapping ingredient from the graph backbone
                filtered_candidates = candidates_df[candidates_df["graph_overlap"] > 0]
                if not filtered_candidates.empty:
                    candidates_df = filtered_candidates
                    print(f"  -> Retained {len(candidates_df)} candidates overlapping with Graph recommendations.")
                else:
                    print("  -> No candidates had direct overlaps with graph ingredients. Using all matching recipes.")
                    candidates_df["graph_overlap"] = 0
            
            if candidates_df.empty:
                print("[!] No candidate recipes found.")
                continue
                
            candidate_ids = candidates_df["RecipeId"].head(200).tolist() # Limit candidate pool for speed
            
            print(f"\n[Step 3] Ranking {len(candidate_ids)} candidates using the Hybrid Recommender (SVD Collaborative + Content Similarity) for User {user_id}...")
            rankings = recommender.predict_hybrid_rankings(user_id, candidate_ids)
            
            print(f"\n========================================================")
            print(f"       TOP 5 PERSONALIZED RECOMMENDATIONS FOR USER {user_id}")
            print(f"========================================================\n")
            
            top_rankings = rankings[:5]
            for rank, (rid, score) in enumerate(top_rankings, 1):
                recipe_row = recipes[recipes["RecipeId"] == rid].iloc[0]
                print(f"{rank}. [{rid}] {recipe_row['Name']} (Category: {recipe_row['RecipeCategory']})")
                print(f"   Ingredients: {recipe_row['RecipeIngredientParts']}")
                print(f"   Hybrid Relevance Score: {score:.4f}")
                if "graph_overlap" in candidates_df.columns:
                    overlap_cnt = int(candidates_df[candidates_df["RecipeId"] == rid]["graph_overlap"].iloc[0])
                    print(f"   Graph overlap: {overlap_cnt} related ingredient(s)")
                print()
                
            print("========================================================\n")
            
            cont = input("Do you want to search again? (y/n): ").strip().lower()
            if cont != 'y':
                print("Exiting demo. Have a great day!")
                break
                
        except KeyboardInterrupt:
            print("\nExiting demo.")
            break
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()
