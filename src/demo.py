#!/usr/bin/env python3
"""
Week 14 Final Demo Artifact
A simple CLI tool to demonstrate the end-to-end outputs of the Recipe Recommendation Project.
"""

import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Recipe Intelligence Demo CLI")
    parser.add_argument("--query", type=str, help="Search keyword for a recipe", default="chocolate")
    args = parser.parse_args()

    print("=========================================")
    print("Recipe Intelligence System - Final Demo")
    print("=========================================\n")

    # Paths
    recipes_path = Path("data/processed/recipes_processed.csv")
    clusters_path = Path("artifacts/week7/clustering_reports/cluster_profile_summary.csv")
    graph_path = Path("artifacts/week12/ingredient_graph/top_ingredients_by_pagerank.csv")

    try:
        # Load Data
        recipes = pd.read_csv(recipes_path)
        clusters = pd.read_csv(clusters_path) if clusters_path.exists() else None
        graph = pd.read_csv(graph_path) if graph_path.exists() else None

        print(f"[*] Loaded {len(recipes)} recipes from {recipes_path.name}")

        # Search Recipe
        print(f"\n[1] Searching for recipes matching: '{args.query}'")
        matches = recipes[recipes['Name'].str.contains(args.query, case=False, na=False)].head(5)
        
        if matches.empty:
            print("No recipes found.")
        else:
            for _, row in matches.iterrows():
                print(f"  - [{row['RecipeId']}] {row['Name']} ({row['RecipeCategory']})")
                print(f"    Ingredients: {row['RecipeIngredientParts']}")
                print()

        # Display Cluster profiles
        if clusters is not None:
            print("[2] Discovered Recipe Clusters (Week 7):")
            for _, row in clusters.head(5).iterrows():
                print(f"  - Cluster {row['cluster']}: Top Category: {row['dominant_category']} | Avg Cal: {row['avg_calories']:.1f}")
        else:
            print("[!] Cluster summary not found.")

        # Display Top Ingredients
        if graph is not None:
            print("\n[3] Most Central Ingredients by PageRank (Week 12):")
            for _, row in graph.head(5).iterrows():
                print(f"  - {row['ingredient']} (Score: {row['pagerank_weighted']:.4f})")
        else:
            print("\n[!] Graph output not found.")

        print("\n=========================================")
        print("Demo completed successfully!")
        print("=========================================")

    except Exception as e:
        print(f"Error during demo: {e}")

if __name__ == "__main__":
    main()
