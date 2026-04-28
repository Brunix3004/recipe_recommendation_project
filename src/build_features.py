"""
build_features.py
Generates feature matrices, applies dimensionality reduction (PCA and SVD)
and exports explained variance plots for the Week 5 report.
"""

import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuración visual para Seaborn
sns.set_theme(style="whitegrid")

def safe_eval_list(x):
    """Converts the list string to space-separated string for TF-IDF"""
    try:
        lst = ast.literal_eval(x)
        if isinstance(lst, list):
            return " ".join([str(i).replace(" ", "_") for i in lst])
    except:
        pass
    return ""

def main():
    print("Loading processed dataset...")
    data_path = os.path.join("data", "raw", "recipes_processed.csv")
    df = pd.read_csv(data_path)

    # Ensure directory to save plots and artifacts
    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    os.makedirs(os.path.join("artifacts"), exist_ok=True)

    # ---------------------------------------------------------
    # 1. NUMERIC MATRIX AND PCA
    # ---------------------------------------------------------
    print("Processing numeric features...")
    num_cols = [
        'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
        'ProteinContent', 'CookTime_Minutes', 'PrepTime_Minutes'
    ]
    
    # Impute nulls with median to avoid data loss
    df_num = df[num_cols].fillna(df[num_cols].median())
    
    # Scale data (Crucial for PCA)
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df_num)
    
    # Apply PCA
    print("Applying PCA to numeric data...")
    pca = PCA()
    pca.fit(num_scaled)
    
    # Explained Variance Plot - PCA
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.axhline(y=0.90, color='r', linestyle='-', label='90% Variance')
    plt.title('PCA - Cumulative Explained Variance (Nutritional and Time Features)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "pca_variance.png"))
    plt.close()

    # ---------------------------------------------------------
    # 2. TEXT MATRIX (INGREDIENTS) AND TRUNCATED SVD
    # ---------------------------------------------------------
    print("Processing ingredients with TF-IDF...")
    # Fill nulls and convert lists to treatable strings
    df['RecipeIngredientParts'] = df['RecipeIngredientParts'].fillna("[]")
    ingredients_text = df['RecipeIngredientParts'].apply(safe_eval_list)
    
    # Vectorize using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, min_df=5)
    tfidf_matrix = tfidf.fit_transform(ingredients_text)
    
    # Apply TruncatedSVD (PCA for sparse matrices)
    print("Applying TruncatedSVD to the TF-IDF matrix...")
    n_components_svd = 100
    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
    svd.fit(tfidf_matrix)
    
    # Explained Variance Plot - SVD
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(svd.explained_variance_ratio_) + 1), 
             np.cumsum(svd.explained_variance_ratio_), color='green')
    plt.title('Truncated SVD - Cumulative Explained Variance (Ingredients TF-IDF)')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.tight_layout()
    plt.savefig(os.path.join("reports", "figures", "svd_variance.png"))
    plt.close()

    # ---------------------------------------------------------
    # 3. EXPORT REDUCED MATRICES
    # ---------------------------------------------------------
    print("Exporting reduced representations...")
    # We save only the first 6 PCA components (~90% variance)
    num_pca_reduced = pca.transform(num_scaled)[:, :6]
    svd_reduced = svd.transform(tfidf_matrix)

    np.save(os.path.join("artifacts", "num_pca_features.npy"), num_pca_reduced)
    np.save(os.path.join("artifacts", "ingredients_svd_features.npy"), svd_reduced)
    
    print(f"Variance explained by 6 PCA components: {np.sum(pca.explained_variance_ratio_[:6]):.2f}")
    print(f"Variance explained by {n_components_svd} SVD components: {np.sum(svd.explained_variance_ratio_):.2f}")
    print("Pipeline completed. Plots saved in reports/figures/.")

if __name__ == "__main__":
    main()