#!/usr/bin/env python3
"""
Run reproducible Week 10 recommendation experiments.

This script loads the processed reviews, constructs a 5-core subset, splits the
data chronologically, trains baseline and collaborative filtering models,
evaluates them using a sampled negatives protocol, and performs error analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate recommendation models.")
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/processed/reviews_processed.csv"),
        help="Path to reviews_processed.csv",
    )
    parser.add_argument(
        "--recipes-metadata",
        type=Path,
        default=Path("data/processed/recipes_processed.csv"),
        help="Path to recipes_processed.csv",
    )
    parser.add_argument(
        "--content-svd",
        type=Path,
        default=Path("artifacts/week5/pca_svd/X_content_svd.npy"),
        help="Path to X_content_svd.npy",
    )
    parser.add_argument(
        "--recipe-ids",
        type=Path,
        default=Path("artifacts/week5/pca_svd/reduced_recipe_ids.csv"),
        help="Path to reduced_recipe_ids.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/week10"),
        help="Output directory for model outputs and metrics",
    )
    parser.add_argument(
        "--cf-factors",
        type=int,
        default=50,
        help="Number of latent factors for SVD Collaborative Filtering",
    )
    parser.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.6,
        help="Weight assigned to Collaborative SVD score in the Hybrid model (0.0 to 1.0)",
    )
    parser.add_argument(
        "--sample-users-eval",
        type=int,
        default=2000,
        help="Number of users to evaluate to keep execution time fast",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_reviews(path: Path) -> pd.DataFrame:
    logger.info("Loading reviews dataset...")
    cols = ["ReviewId", "RecipeId", "AuthorId", "Rating", "DateSubmitted"]
    df = pd.read_csv(path, usecols=cols)
    df["DateSubmitted"] = pd.to_datetime(df["DateSubmitted"])
    return df


def get_5_core(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    logger.info(f"Filtering dataset to {k}-core...")
    df_core = df.copy()
    iteration = 0
    while True:
        u_counts = df_core["AuthorId"].value_counts()
        r_counts = df_core["RecipeId"].value_counts()
        keep_u = u_counts[u_counts >= k].index
        keep_r = r_counts[r_counts >= k].index
        filtered = df_core[df_core["AuthorId"].isin(keep_u) & df_core["RecipeId"].isin(keep_r)]
        iteration += 1
        logger.info(
            f"Iteration {iteration}: {len(filtered)} interactions remaining "
            f"({filtered['AuthorId'].nunique()} users, {filtered['RecipeId'].nunique()} recipes)"
        )
        if len(filtered) == len(df_core):
            break
        df_core = filtered
    return df_core.reset_index(drop=True)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Performing chronological 80/20 train/test split per user...")
    df_sorted = df.sort_values(by=["AuthorId", "DateSubmitted"]).copy()
    df_sorted["user_idx"] = df_sorted.groupby("AuthorId").cumcount()
    df_sorted["user_total"] = df_sorted.groupby("AuthorId")["AuthorId"].transform("count")

    # Split: oldest 80% to train, newest 20% to test.
    # Enforce that every user has at least 1 item in train and 1 item in test.
    train_cutoff = (df_sorted["user_total"] * 0.8).astype(int).clip(1, df_sorted["user_total"] - 1)
    train_mask = df_sorted["user_idx"] < train_cutoff

    df_train = df_sorted[train_mask].copy()
    df_test = df_sorted[~train_mask].copy()

    logger.info(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    return df_train, df_test


class PopularityRecommender:
    """Popularity-based recommendation using Bayesian average ratings."""

    def __init__(self, m: float = 10.0) -> None:
        self.m = m
        self.recipe_scores: dict[int, float] = {}
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        logger.info("Fitting Popularity Recommender...")
        stats = train_df.groupby("RecipeId")["Rating"].agg(["count", "mean"])
        self.global_mean = float(train_df["Rating"].mean())

        # Bayesian average formula:
        # Score = (v * R + m * C) / (v + m)
        stats["score"] = (stats["count"] * stats["mean"] + self.m * self.global_mean) / (
            stats["count"] + self.m
        )
        self.recipe_scores = stats["score"].to_dict()

    def predict_score(self, recipe_id: int) -> float:
        return self.recipe_scores.get(recipe_id, self.global_mean)


class CollaborativeSVDRecommender:
    """Matrix Factorization Collaborative Filtering using sparse SVD."""

    def __init__(self, n_factors: int = 50) -> None:
        self.n_factors = n_factors
        self.user_to_idx: dict[int, int] = {}
        self.recipe_to_idx: dict[int, int] = {}
        self.user_means: np.ndarray = np.array([])
        self.user_factors: np.ndarray = np.array([])
        self.recipe_factors: np.ndarray = np.array([])
        self.global_mean = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        logger.info("Fitting SVD Collaborative Filtering...")
        self.global_mean = float(train_df["Rating"].mean())

        # Map IDs to continuous indices
        unique_users = train_df["AuthorId"].unique()
        unique_recipes = train_df["RecipeId"].unique()

        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.recipe_to_idx = {rid: idx for idx, rid in enumerate(unique_recipes)}

        n_users = len(unique_users)
        n_recipes = len(unique_recipes)

        # Compute user means for mean-centering
        user_ratings_sum = np.zeros(n_users)
        user_ratings_count = np.zeros(n_users)

        u_indices = np.array([self.user_to_idx[uid] for uid in train_df["AuthorId"]])
        r_indices = np.array([self.recipe_to_idx[rid] for rid in train_df["RecipeId"]])
        ratings = train_df["Rating"].values

        for u, r_val in zip(u_indices, ratings):
            user_ratings_sum[u] += r_val
            user_ratings_count[u] += 1

        self.user_means = user_ratings_sum / (user_ratings_count + 1e-9)
        centered_ratings = ratings - self.user_means[u_indices]

        # Construct sparse rating matrix
        R = sp.coo_matrix(
            (centered_ratings, (u_indices, r_indices)),
            shape=(n_users, n_recipes),
            dtype=np.float32,
        ).tocsr()

        # Run Sparse SVD
        k = min(self.n_factors, min(n_users, n_recipes) - 2)
        logger.info(f"Computing sparse SVD with k={k} factors...")
        U, s, Vt = svds(R, k=k)

        # Sort SVD components in descending order of singular values
        sort_idx = np.argsort(s)[::-1]
        U = U[:, sort_idx]
        s = s[sort_idx]
        Vt = Vt[sort_idx, :]

        # Compute user and item latent matrices
        # We scale both user and item factors by sqrt(singular_values)
        sqrt_s = np.sqrt(s)
        self.user_factors = U * sqrt_s
        self.recipe_factors = Vt.T * sqrt_s

    def predict_score(self, author_id: int, recipe_id: int) -> float:
        u_idx = self.user_to_idx.get(author_id)
        r_idx = self.recipe_to_idx.get(recipe_id)

        if u_idx is None and r_idx is None:
            return self.global_mean
        elif u_idx is None:
            # Cold-start user: return global mean rating
            return self.global_mean
        elif r_idx is None:
            # Cold-start recipe: return user mean rating
            return self.user_means[u_idx]

        # Predict as user_mean + dot_product(user_factors, recipe_factors)
        pred = self.user_means[u_idx] + np.dot(self.user_factors[u_idx], self.recipe_factors[r_idx])
        return float(pred)


class ContentSVDRecommender:
    """Content-based recommender using precomputed SVD embeddings from Week 5."""

    def __init__(self, content_svd_path: Path, recipe_ids_path: Path) -> None:
        logger.info(f"Loading Content SVD embeddings from {content_svd_path}...")
        self.X_content = np.load(content_svd_path)
        logger.info(f"Content SVD shape: {self.X_content.shape}")

        # DESPUÉS
        recipe_ids_df = pd.read_csv(recipe_ids_path)
        # Detect the index column robustly: prefer 'row_index', fallback to positional index
        if "row_index" in recipe_ids_df.columns:
            self.recipe_to_svd_idx = {
                int(row["RecipeId"]): int(row["row_index"]) for _, row in recipe_ids_df.iterrows()
            }
        else:
            # The CSV may have been written with the dataframe index as the row position
            self.recipe_to_svd_idx = {
                int(row["RecipeId"]): int(idx) for idx, row in recipe_ids_df.iterrows()
            }
        if not self.recipe_to_svd_idx:
            raise ValueError(
                f"recipe_to_svd_idx is empty after reading {recipe_ids_path}. "
                "Check that the file contains a 'RecipeId' column."
            )
        logger.info(f"Loaded {len(self.recipe_to_svd_idx)} recipe SVD index mappings.")

        # Normalize the content vectors to unit length for fast cosine similarity via dot product
        norms = np.linalg.norm(self.X_content, axis=1, keepdims=True)
        self.X_content_normalized = self.X_content / (norms + 1e-9)

        self.user_profiles: dict[int, np.ndarray] = {}
        self.global_profile = np.zeros(self.X_content.shape[1], dtype=np.float32)

    def fit(self, train_df: pd.DataFrame) -> None:
        logger.info("Computing User Content Profiles from high-rated recipes...")

        # Filter training ratings to those >= 4.0 (liked recipes) to construct profiles
        liked_df = train_df[train_df["Rating"] >= 4.0]

        # Fallback to all training reviews if user has no ratings >= 4.0
        fallback_df = train_df

        # Group by user and collect SVD vectors
        user_grouped = liked_df.groupby("AuthorId")
        fallback_grouped = fallback_df.groupby("AuthorId")

        all_profiles = []
        uids = train_df["AuthorId"].unique()

        for uid in uids:
            # Get recipes for user
            rids = []
            if uid in user_grouped.groups:
                rids = liked_df.loc[user_grouped.groups[uid], "RecipeId"].values
            else:
                rids = fallback_df.loc[fallback_grouped.groups[uid], "RecipeId"].values

            # Look up SVD indices
            svd_indices = [self.recipe_to_svd_idx[rid] for rid in rids if rid in self.recipe_to_svd_idx]

            if svd_indices:
                profile = self.X_content_normalized[svd_indices].mean(axis=0)
                profile_norm = np.linalg.norm(profile)
                if profile_norm > 0:
                    profile = profile / profile_norm
                self.user_profiles[uid] = profile
                all_profiles.append(profile)

        if all_profiles:
            self.global_profile = np.mean(all_profiles, axis=0)
            gp_norm = np.linalg.norm(self.global_profile)
            if gp_norm > 0:
                self.global_profile = self.global_profile / gp_norm

    def predict_score(self, author_id: int, recipe_id: int) -> float:
        profile = self.user_profiles.get(author_id, self.global_profile)
        svd_idx = self.recipe_to_svd_idx.get(recipe_id)

        if svd_idx is None:
            return 0.0  # Missing content representation: return neutral similarity

        # Cosine similarity is just the dot product because both are normalized
        sim = np.dot(profile, self.X_content_normalized[svd_idx])
        return float(sim)


def evaluate_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    popularity_model: PopularityRecommender,
    svd_model: CollaborativeSVDRecommender,
    content_model: ContentSVDRecommender,
    hybrid_alpha: float,
    sample_users_count: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    logger.info("Starting offline evaluation...")
    rng = np.random.default_rng(seed)

    # Get active users in test who are also in train
    train_users = set(train_df["AuthorId"].unique())
    test_users_all = test_df["AuthorId"].unique()
    eval_users = [uid for uid in test_users_all if uid in train_users]

    if len(eval_users) > sample_users_count:
        eval_users = rng.choice(eval_users, size=sample_users_count, replace=False).tolist()

    logger.info(f"Evaluating on a representative sample of {len(eval_users)} test users...")

    # All candidate recipes (5-core)
    all_recipes = np.array(list(set(train_df["RecipeId"].unique()).union(test_df["RecipeId"].unique())))

    # Dictionary of user interaction history (train + test) to filter out of negatives
    user_history: dict[int, set[int]] = {}
    for row in train_df.itertuples():
        user_history.setdefault(row.AuthorId, set()).add(row.RecipeId)
    for row in test_df.itertuples():
        user_history.setdefault(row.AuthorId, set()).add(row.RecipeId)

    # Metric accumulators
    metrics = {
        "popularity": {"hit@5": 0.0, "hit@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
        "collaborative_cf": {"hit@5": 0.0, "hit@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
        "content_svd": {"hit@5": 0.0, "hit@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
        "hybrid": {"hit@5": 0.0, "hit@10": 0.0, "ndcg@5": 0.0, "ndcg@10": 0.0, "mrr": 0.0},
    }

    test_by_user = test_df.groupby("AuthorId")
    total_eval_items = 0

    # For error analysis tracking
    error_cases: list[dict[str, Any]] = []

    progress_step = max(1, len(eval_users) // 10)

    for idx, uid in enumerate(eval_users):
        if idx % progress_step == 0:
            logger.info(f"Evaluated {idx}/{len(eval_users)} users ({(idx/len(eval_users))*100:.0f}%)")

        history = user_history.get(uid, set())
        if uid not in test_by_user.groups:
            logger.warning(f"No test interactions for user {uid}")
            continue
        user_test_items = test_by_user.get_group(uid)

        for row in user_test_items.itertuples():
            target_recipe = row.RecipeId
            target_rating = row.Rating

            # Sample 100 negatives
            # Filter all recipes to those not in user history
            allowed_negatives = all_recipes[~np.isin(all_recipes, list(history))]
            if len(allowed_negatives) < 100:
                # Fallback if too few recipes left
                negatives = rng.choice(all_recipes, size=100, replace=True)
            else:
                negatives = rng.choice(allowed_negatives, size=100, replace=False)

            candidates = [target_recipe] + list(negatives)

            # Predict scores
            pop_scores = np.array([popularity_model.predict_score(r) for r in candidates])
            cf_scores = np.array([svd_model.predict_score(uid, r) for r in candidates])
            content_scores = np.array([content_model.predict_score(uid, r) for r in candidates])

            # Normalize CF and Content scores to [0, 1] within candidate pool for hybrid alignment
            min_cf, max_cf = cf_scores.min(), cf_scores.max()
            cf_scores_norm = (
                (cf_scores - min_cf) / (max_cf - min_cf + 1e-9)
                if max_cf > min_cf
                else np.ones_like(cf_scores)
            )

            min_content, max_content = content_scores.min(), content_scores.max()
            content_scores_norm = (
                (content_scores - min_content) / (max_content - min_content + 1e-9)
                if max_content > min_content
                else np.ones_like(content_scores)
            )

            # Hybrid score
            hybrid_scores = hybrid_alpha * cf_scores_norm + (1.0 - hybrid_alpha) * content_scores_norm

            # Rank candidates (descending) and compute metrics
            models_scores = {
                "popularity": pop_scores,
                "collaborative_cf": cf_scores,
                "content_svd": content_scores,
                "hybrid": hybrid_scores,
            }

            ranks = {}
            for name, scores in models_scores.items():
                # Get ranks (descending order).
                # We sort args descending: argsort of negative scores
                sorted_indices = np.argsort(-scores)
                # Find rank of target recipe (which is index 0 in candidates)
                rank = int(np.where(sorted_indices == 0)[0][0])
                ranks[name] = rank

                # Accumulate metrics
                if rank < 5:
                    metrics[name]["hit@5"] += 1.0
                    metrics[name]["ndcg@5"] += 1.0 / np.log2(rank + 2)
                if rank < 10:
                    metrics[name]["hit@10"] += 1.0
                    metrics[name]["ndcg@10"] += 1.0 / np.log2(rank + 2)
                metrics[name]["mrr"] += 1.0 / (rank + 1)

            total_eval_items += 1

            # Save interesting cases for error analysis
            # We track user, target, rank, scores, ratings
            # Case 1: Strong Case (CF and Hybrid rank a highly rated target in Top 1)
            if target_rating >= 4.0 and ranks["hybrid"] == 0:
                if len([c for c in error_cases if c["type"] == "strong_case"]) < 5:
                    error_cases.append(
                        {
                            "type": "strong_case",
                            "uid": int(uid),
                            "target_recipe": int(target_recipe),
                            "rating": float(target_rating),
                            "ranks": ranks,
                            "cf_score": float(cf_scores[0]),
                            "content_score": float(content_scores[0]),
                            "hybrid_score": float(hybrid_scores[0]),
                        }
                    )

            # Case 2: Failure Case (Highly rated target ranked low, e.g. > 50)
            if target_rating >= 4.0 and ranks["hybrid"] >= 50:
                if len([c for c in error_cases if c["type"] == "failure_low_rank"]) < 5:
                    error_cases.append(
                        {
                            "type": "failure_low_rank",
                            "uid": int(uid),
                            "target_recipe": int(target_recipe),
                            "rating": float(target_rating),
                            "ranks": ranks,
                            "cf_score": float(cf_scores[0]),
                            "content_score": float(content_scores[0]),
                            "hybrid_score": float(hybrid_scores[0]),
                        }
                    )

            # Case 3: Failure Case (User disliked the target, rated 1 or 2 stars, but Hybrid recommended it in Top 5)
            if target_rating <= 2.0 and ranks["hybrid"] < 5:
                if len([c for c in error_cases if c["type"] == "failure_disliked_recommended"]) < 5:
                    error_cases.append(
                        {
                            "type": "failure_disliked_recommended",
                            "uid": int(uid),
                            "target_recipe": int(target_recipe),
                            "rating": float(target_rating),
                            "ranks": ranks,
                            "cf_score": float(cf_scores[0]),
                            "content_score": float(content_scores[0]),
                            "hybrid_score": float(hybrid_scores[0]),
                        }
                    )

    # Average metrics
    avg_metrics_rows = []
    for model_name, m_dict in metrics.items():
        row = {"model": model_name}
        for metric_name, val in m_dict.items():
            row[metric_name] = val / total_eval_items
        avg_metrics_rows.append(row)

    avg_metrics_df = pd.DataFrame(avg_metrics_rows)
    logger.info("Evaluation complete. Metrics summary:\n" + str(avg_metrics_df))
    return avg_metrics_df, error_cases


def add_recipe_names_to_errors(
    error_cases: list[dict[str, Any]], recipes_metadata_path: Path
) -> list[dict[str, Any]]:
    logger.info("Enriching error cases with recipe titles...")
    if not recipes_metadata_path.exists():
        logger.warning(
            f"Recipes metadata path {recipes_metadata_path} not found. Skipping title lookup."
        )
        return error_cases

    # Read processed recipe metadata to map RecipeId to Name
    recipes_df = pd.read_csv(recipes_metadata_path, usecols=["RecipeId", "Name"])
    id_to_name = {
        int(row["RecipeId"]): str(row["Name"]) for _, row in recipes_df.iterrows()
    }

    for case in error_cases:
        case["recipe_name"] = id_to_name.get(case["target_recipe"], "Unknown Recipe")

    return error_cases


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Week 10 Recommendation Engine Pipeline...")

    # 1. Ingest/Load data
    reviews_df = load_reviews(args.reviews)
    logger.info(f"Loaded {len(reviews_df)} reviews.")

    # 2. Filter to 5-core
    core_df = get_5_core(reviews_df, k=5)

    # 3. Stratified Chronological Train-Test split
    train_df, test_df = chronological_split(core_df)

    # 4. Save train/test counts and details in a json configuration metadata
    data_meta = {
        "total_reviews": len(reviews_df),
        "core_5_reviews": len(core_df),
        "train_reviews": len(train_df),
        "test_reviews": len(test_df),
        "unique_users_core": core_df["AuthorId"].nunique(),
        "unique_recipes_core": core_df["RecipeId"].nunique(),
    }
    with open(args.out_dir / "data_metadata.json", "w", encoding="utf-8") as f:
        json.dump(data_meta, f, indent=2)

    # 5. Fit Popularity Baseline
    pop_model = PopularityRecommender(m=10.0)
    pop_model.fit(train_df)

    # 6. Fit SVD Collaborative Filtering Model
    cf_model = CollaborativeSVDRecommender(n_factors=args.cf_factors)
    cf_model.fit(train_df)

    # 7. Fit Content SVD Model
    content_model = ContentSVDRecommender(
        content_svd_path=args.content_svd, recipe_ids_path=args.recipe_ids
    )
    content_model.fit(train_df)

    # 8. Evaluate models
    avg_metrics_df, error_cases = evaluate_models(
        train_df=train_df,
        test_df=test_df,
        popularity_model=pop_model,
        svd_model=cf_model,
        content_model=content_model,
        hybrid_alpha=args.hybrid_alpha,
        sample_users_count=args.sample_users_eval,
        seed=args.seed,
    )

    # Save metrics
    avg_metrics_df.to_csv(args.out_dir / "recommendation_evaluation_summary.csv", index=False)

    # 9. Enrich and Save Error Analysis cases
    error_cases = add_recipe_names_to_errors(error_cases, args.recipes_metadata)
    with open(args.out_dir / "recommendation_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)

    # Save additional run configurations
    run_config = {
        "cf_factors": args.cf_factors,
        "hybrid_alpha": args.hybrid_alpha,
        "sample_users_eval": args.sample_users_eval,
        "random_seed": args.seed,
    }
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Pipeline run completed successfully. Outputs saved under:")
    logger.info(f"- {args.out_dir}")


if __name__ == "__main__":
    main()
