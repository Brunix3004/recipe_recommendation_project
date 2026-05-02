"""
Build the Week 5 sparse content TF-IDF matrix for Food.com recipes.

This script is Step 4 only. It builds a sparse recipe-level content matrix from
ingredients, keywords, cleaned categories, and interpretable yield-unit tokens.
It does not run TruncatedSVD, PCA, clustering, or recommendations. TruncatedSVD
will be applied later to this sparse matrix to create dense recipe embeddings
for clustering and content-based recommendation.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


INGREDIENT_COL = "RecipeIngredientParts"
RECIPE_ID_COL = "RecipeId"
KEYWORD_CANDIDATES = ["RecipeKeywords", "Keywords"]
CATEGORY_COL = "RecipeCategory_clean"
YIELD_COL = "yield_unit_token"

GROUP_WEIGHTS = {
    "ingredients": 1.0,
    "keywords": 0.6,
    "category": 0.4,
    "yield": 0.2,
}

TIME_KEYWORD_PATTERNS = [
    re.compile(r"^\d+_mins?$"),
    re.compile(r"^\d+_minutes?$"),
    re.compile(r"^\d+_hours?$"),
    re.compile(r"^\d+_hours_or_less$"),
    re.compile(r"^time_to_make$"),
    re.compile(r"^weeknight$"),
]

EXCLUDED_RAW_COLUMNS = [
    "RecipeYield",
    "AggregatedRating",
    "ReviewCount",
    "Rating",
]


def load_recipes(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Reading Parquet requires pyarrow. Install it with "
            "`pip install pyarrow` and rerun this script."
        ) from exc
    except ValueError as exc:
        if "parquet" in str(exc).lower() or "pyarrow" in str(exc).lower():
            raise ValueError(
                "Unable to read the Parquet input. Install pyarrow with "
                "`pip install pyarrow` and verify the file is a valid Parquet file."
            ) from exc
        raise


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def safe_parse_list(value: Any) -> list[str]:
    """Parse Food.com list-like cells without failing on malformed rows."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, tuple):
            return [str(item) for item in parsed]
        if parsed is None:
            return []
        return [str(parsed)]
    except (SyntaxError, ValueError):
        return [text]


def normalize_token(token: str) -> str:
    token = str(token).strip().lower()
    if token in {"", "nan", "none", "null"}:
        return ""

    token = token.replace("&", " and ")
    token = token.replace("'", "")
    token = re.sub(r"[^\w\s/-]+", " ", token)
    token = re.sub(r"[\s-]+", "_", token)
    token = re.sub(r"_+", "_", token)
    token = token.strip("_")

    if token in {"", "nan", "none", "null"}:
        return ""
    return token


def dedupe_sorted(tokens: list[str]) -> list[str]:
    return sorted(set(token for token in tokens if token))


def build_ingredient_documents(df: pd.DataFrame) -> pd.Series:
    # Ingredients are the core recipe-content signal.
    docs = []
    for value in df[INGREDIENT_COL]:
        tokens = [normalize_token(token) for token in safe_parse_list(value)]
        docs.append(" ".join(dedupe_sorted(tokens)))
    return pd.Series(docs, index=df.index, dtype="string")


def is_time_like_keyword(token: str) -> bool:
    return any(pattern.match(token) for pattern in TIME_KEYWORD_PATTERNS)


def build_keyword_documents(
    df: pd.DataFrame, keyword_col: str
) -> tuple[pd.Series, pd.DataFrame]:
    # Keywords provide tag-level semantics, but they are noisier than
    # ingredients and can overlap with numeric time features, so time-like tags
    # are removed and the resulting matrix is downweighted.
    docs = []
    removed_counter: Counter[str] = Counter()

    for value in df[keyword_col]:
        kept_tokens = []
        for raw_token in safe_parse_list(value):
            token = normalize_token(raw_token)
            if not token:
                continue
            if is_time_like_keyword(token):
                removed_counter[token] += 1
                continue
            kept_tokens.append(token)
        docs.append(" ".join(dedupe_sorted(kept_tokens)))

    removed = pd.DataFrame(
        [
            {"keyword_token": token, "removed_count": count}
            for token, count in removed_counter.most_common()
        ],
        columns=["keyword_token", "removed_count"],
    )
    return pd.Series(docs, index=df.index, dtype="string"), removed


def build_category_documents(df: pd.DataFrame) -> pd.Series:
    # RecipeCategory is sparse binary metadata. Missing category is not imputed
    # and no category__unknown token is created.
    if CATEGORY_COL not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    docs = []
    for value in df[CATEGORY_COL]:
        if value is None or pd.isna(value):
            docs.append("")
            continue
        token = normalize_token(str(value))
        docs.append(f"category__{token}" if token else "")
    return pd.Series(docs, index=df.index, dtype="string")


def build_yield_documents(df: pd.DataFrame) -> pd.Series:
    # Yield units are optional auxiliary content signals and are included only
    # when interpretable. Raw RecipeYield is deliberately excluded, and the
    # curated yield__ tokens from Step 2 are kept as-is.
    if YIELD_COL not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="string")

    docs = []
    excluded = {"", "yield__missing", "yield__other"}
    for value in df[YIELD_COL]:
        if value is None or pd.isna(value):
            docs.append("")
            continue
        token = str(value).strip().lower()
        if token in excluded or not re.fullmatch(r"yield__[a-z0-9_/]+", token):
            docs.append("")
            continue
        docs.append(token)
    return pd.Series(docs, index=df.index, dtype="string")


def make_vectorizer(group: str) -> TfidfVectorizer:
    common = {
        "token_pattern": r"(?u)\b[\w/]+\b",
        "dtype": np.float32,
    }
    if group == "ingredients":
        return TfidfVectorizer(
            min_df=10,
            max_df=0.80,
            sublinear_tf=True,
            norm="l2",
            **common,
        )
    if group == "keywords":
        return TfidfVectorizer(
            min_df=10,
            max_df=0.60,
            sublinear_tf=True,
            norm="l2",
            **common,
        )
    if group == "category":
        return TfidfVectorizer(
            min_df=1,
            max_df=1.0,
            use_idf=False,
            norm=None,
            binary=True,
            **common,
        )
    if group == "yield":
        return TfidfVectorizer(
            min_df=50,
            max_df=1.0,
            use_idf=False,
            norm=None,
            binary=True,
            **common,
        )
    raise ValueError(f"Unknown vectorizer group: {group}")


def empty_csv(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def vectorizer_settings(vectorizer: TfidfVectorizer) -> dict[str, Any]:
    return {
        "min_df": vectorizer.min_df,
        "max_df": vectorizer.max_df,
        "sublinear_tf": vectorizer.sublinear_tf,
        "norm": vectorizer.norm,
        "use_idf": vectorizer.use_idf,
        "binary": vectorizer.binary,
        "token_pattern": vectorizer.token_pattern,
        "dtype": "float32",
    }


def matrix_density(matrix: csr_matrix) -> float:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return 0.0
    return float(matrix.nnz / (matrix.shape[0] * matrix.shape[1]))


def fit_group_matrix(
    docs: pd.Series,
    group: str,
    source_column: str,
) -> tuple[csr_matrix | None, TfidfVectorizer | None, dict[str, Any], list[str]]:
    n_docs = len(docs)
    non_empty = int(docs.str.len().gt(0).sum())
    vectorizer = make_vectorizer(group)

    base_summary = {
        "feature_group": group,
        "source_column": source_column,
        "included": False,
        "n_documents": n_docs,
        "non_empty_documents": non_empty,
        "non_empty_document_percent": round(non_empty / n_docs * 100, 4) if n_docs else 0.0,
        "vocab_size": 0,
        "matrix_nonzero_count": 0,
        "matrix_density": 0.0,
        "group_weight": GROUP_WEIGHTS[group],
        "vectorizer_min_df": vectorizer.min_df,
        "vectorizer_max_df": vectorizer.max_df,
    }

    if non_empty == 0:
        return None, None, base_summary, []

    try:
        matrix = vectorizer.fit_transform(docs).astype(np.float32).tocsr()
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower() or "no terms remain" in str(exc).lower():
            return None, None, base_summary, []
        raise

    vocab = vectorizer.get_feature_names_out().tolist()
    weighted_matrix = (matrix * np.float32(GROUP_WEIGHTS[group])).tocsr()
    base_summary.update(
        {
            "included": True,
            "vocab_size": len(vocab),
            "matrix_nonzero_count": int(weighted_matrix.nnz),
            "matrix_density": matrix_density(weighted_matrix),
        }
    )
    return weighted_matrix, vectorizer, base_summary, vocab


def prefixed_feature_name(group: str, token: str) -> str:
    if group == "ingredients":
        return f"ingredient__{token}"
    if group == "keywords":
        return f"keyword__{token}"
    if group == "category":
        return token
    if group == "yield":
        return token
    raise ValueError(f"Unknown feature group: {group}")


def token_summary(
    matrix: csr_matrix | None,
    vocab: list[str],
    token_col: str,
) -> pd.DataFrame:
    columns = [token_col, "document_frequency", "total_tfidf_weight"]
    if matrix is None or not vocab:
        return empty_csv(columns)

    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    total_weight = np.asarray(matrix.sum(axis=0)).ravel()
    summary = pd.DataFrame(
        {
            token_col: vocab,
            "document_frequency": doc_freq.astype(np.int64),
            "total_tfidf_weight": total_weight.astype(float),
        }
    )
    return summary.sort_values(
        ["document_frequency", "total_tfidf_weight", token_col],
        ascending=[False, False, True],
    )


def count_single_token_docs(docs: pd.Series, token_col: str) -> pd.DataFrame:
    columns = [token_col, "recipe_count"]
    tokens = docs[docs.str.len().gt(0)]
    if tokens.empty:
        return empty_csv(columns)
    counts = tokens.value_counts().rename_axis(token_col).reset_index(name="recipe_count")
    return counts.sort_values(["recipe_count", token_col], ascending=[False, True])


def select_keyword_column(df: pd.DataFrame) -> str | None:
    for col in KEYWORD_CANDIDATES:
        if col in df.columns:
            return col
    return None


def resolve_numeric_recipe_ids_path(path: Path | None, output_dir: Path) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path

    fallback = output_dir.parent / "numeric_matrix_outputs" / "recipe_ids.csv"
    if fallback.exists():
        print(
            "WARNING: numeric recipe IDs path does not exist: "
            f"{path}. Using fallback: {fallback}"
        )
        return fallback

    print(f"WARNING: numeric recipe IDs path does not exist, skipping alignment check: {path}")
    return None


def validate_recipe_ids(
    df: pd.DataFrame, numeric_recipe_ids_path: Path | None
) -> pd.DataFrame:
    if df[RECIPE_ID_COL].isna().any():
        raise ValueError("RecipeId contains null values.")
    if df[RECIPE_ID_COL].duplicated().any():
        duplicate_count = int(df[RECIPE_ID_COL].duplicated().sum())
        raise ValueError(f"RecipeId contains duplicates: {duplicate_count} rows.")

    content_ids = pd.DataFrame(
        {
            "row_index": np.arange(len(df), dtype=np.int64),
            "RecipeId": df[RECIPE_ID_COL].to_numpy(),
        }
    )

    if numeric_recipe_ids_path is not None:
        numeric_ids = pd.read_csv(numeric_recipe_ids_path)
        validate_required_columns(numeric_ids, ["row_index", "RecipeId"])
        if len(numeric_ids) != len(content_ids):
            raise ValueError(
                "Numeric recipe ID mapping row count does not match content input: "
                f"{len(numeric_ids)} != {len(content_ids)}"
            )
        if not numeric_ids["RecipeId"].reset_index(drop=True).equals(
            content_ids["RecipeId"].reset_index(drop=True)
        ):
            raise ValueError("RecipeId order does not match numeric matrix recipe IDs.")

    return content_ids


def build_content_matrix(
    df: pd.DataFrame,
) -> tuple[
    csr_matrix,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, TfidfVectorizer],
    dict[str, Any],
]:
    ingredient_docs = build_ingredient_documents(df)
    keyword_col = select_keyword_column(df)
    if keyword_col is None:
        print("WARNING: neither RecipeKeywords nor Keywords exists. Skipping keywords.")
        keyword_docs = pd.Series([""] * len(df), index=df.index, dtype="string")
        removed_time_keywords = empty_csv(["keyword_token", "removed_count"])
    else:
        keyword_docs, removed_time_keywords = build_keyword_documents(df, keyword_col)

    category_docs = build_category_documents(df)
    yield_docs = build_yield_documents(df)

    group_specs = [
        ("ingredients", ingredient_docs, INGREDIENT_COL),
        ("keywords", keyword_docs, keyword_col or ""),
        ("category", category_docs, CATEGORY_COL if CATEGORY_COL in df.columns else ""),
        ("yield", yield_docs, YIELD_COL if YIELD_COL in df.columns else ""),
    ]

    matrices: list[csr_matrix] = []
    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    vectorizers: dict[str, TfidfVectorizer] = {}
    group_outputs: dict[str, dict[str, Any]] = {}
    feature_index = 0

    for group, docs, source_column in group_specs:
        matrix, vectorizer, summary, vocab = fit_group_matrix(docs, group, source_column)
        summary_rows.append(summary)
        group_outputs[group] = {
            "matrix": matrix,
            "vocab": vocab,
            "docs": docs,
            "source_column": source_column,
            "included": bool(summary["included"]),
            "vectorizer_settings": vectorizer_settings(make_vectorizer(group)),
        }

        if matrix is None or vectorizer is None:
            if group in {"keywords", "yield"}:
                print(f"WARNING: {group} matrix skipped.")
            continue

        matrices.append(matrix)
        vectorizers[group] = vectorizer
        for token in vocab:
            feature_rows.append(
                {
                    "feature_index": feature_index,
                    "feature_name": prefixed_feature_name(group, token),
                    "feature_group": group,
                    "source_column": source_column,
                    "group_weight": GROUP_WEIGHTS[group],
                }
            )
            feature_index += 1

    if not matrices:
        raise ValueError("No content feature groups produced a non-empty matrix.")

    X_content = hstack(matrices, format="csr", dtype=np.float32)
    feature_names = pd.DataFrame(feature_rows)
    summary_rows.append(
        {
            "feature_group": "combined_content_matrix",
            "source_column": "",
            "included": True,
            "n_documents": len(df),
            "non_empty_documents": "",
            "non_empty_document_percent": "",
            "vocab_size": X_content.shape[1],
            "matrix_nonzero_count": int(X_content.nnz),
            "matrix_density": matrix_density(X_content),
            "group_weight": "",
            "vectorizer_min_df": "",
            "vectorizer_max_df": "",
        }
    )

    ingredient_top = token_summary(
        group_outputs["ingredients"]["matrix"],
        group_outputs["ingredients"]["vocab"],
        "token",
    )
    keyword_top = token_summary(
        group_outputs["keywords"]["matrix"],
        group_outputs["keywords"]["vocab"],
        "token",
    )
    category_counts = count_single_token_docs(category_docs, "category_token")
    yield_counts = count_single_token_docs(yield_docs, "yield_token")

    metadata = {
        "keyword_column_used": keyword_col,
        "feature_groups_included": [
            group for group, output in group_outputs.items() if output["included"]
        ],
        "group_outputs": group_outputs,
    }

    return (
        X_content,
        feature_names,
        pd.DataFrame(summary_rows),
        ingredient_top,
        keyword_top,
        category_counts,
        yield_counts,
        removed_time_keywords,
        vectorizers,
        metadata,
    )


def validate_content_outputs(
    df: pd.DataFrame,
    X_content: csr_matrix,
    feature_names: pd.DataFrame,
    content_recipe_ids: pd.DataFrame,
) -> None:
    if len(df) != X_content.shape[0]:
        raise ValueError("Matrix row count does not match input dataframe row count.")
    if X_content.shape[1] != len(feature_names):
        raise ValueError("Matrix column count does not match content_feature_names.csv rows.")
    if not np.isfinite(X_content.data).all():
        raise ValueError("Content matrix data contains NaN or infinite values.")

    expected_index = np.arange(len(feature_names), dtype=np.int64)
    if not np.array_equal(feature_names["feature_index"].to_numpy(), expected_index):
        raise ValueError("feature_index is not contiguous from 0 to n_features - 1.")

    if not content_recipe_ids["RecipeId"].reset_index(drop=True).equals(
        df[RECIPE_ID_COL].reset_index(drop=True)
    ):
        raise ValueError("content_recipe_ids row order does not match dataframe row order.")


def print_validation_warnings(summary: pd.DataFrame, X_content: csr_matrix) -> None:
    group_rows = summary[
        summary["feature_group"].ne("combined_content_matrix") & summary["included"].eq(True)
    ]
    for _, row in group_rows.iterrows():
        pct = float(row["non_empty_document_percent"])
        if pct < 1.0:
            print(
                "WARNING: feature group has less than 1% non-empty documents: "
                f"{row['feature_group']} ({pct:.4f}%)."
            )

    density = matrix_density(X_content)
    if density > 0.05:
        print(f"WARNING: combined content matrix density is high: {density:.6f}.")


def save_outputs(
    output_dir: Path,
    X_content: csr_matrix,
    content_recipe_ids: pd.DataFrame,
    feature_names: pd.DataFrame,
    matrix_summary: pd.DataFrame,
    ingredient_top: pd.DataFrame,
    keyword_top: pd.DataFrame,
    category_counts: pd.DataFrame,
    yield_counts: pd.DataFrame,
    removed_time_keywords: pd.DataFrame,
    vectorizers: dict[str, TfidfVectorizer],
    config: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "matrix": output_dir / "X_content_tfidf.npz",
        "recipe_ids": output_dir / "content_recipe_ids.csv",
        "feature_names": output_dir / "content_feature_names.csv",
        "summary": output_dir / "content_matrix_summary.csv",
        "top_ingredients": output_dir / "top_ingredient_tokens.csv",
        "top_keywords": output_dir / "top_keyword_tokens.csv",
        "category_counts": output_dir / "category_token_counts.csv",
        "yield_counts": output_dir / "yield_token_counts.csv",
        "removed_time_keywords": output_dir / "removed_time_keyword_counts.csv",
        "vectorizers": output_dir / "content_vectorizers.joblib",
        "config": output_dir / "content_preprocessing_config.json",
    }

    save_npz(paths["matrix"], X_content)
    content_recipe_ids.to_csv(paths["recipe_ids"], index=False)
    feature_names.to_csv(paths["feature_names"], index=False)
    matrix_summary.to_csv(paths["summary"], index=False)
    ingredient_top.to_csv(paths["top_ingredients"], index=False)
    keyword_top.to_csv(paths["top_keywords"], index=False)
    category_counts.to_csv(paths["category_counts"], index=False)
    yield_counts.to_csv(paths["yield_counts"], index=False)
    removed_time_keywords.to_csv(paths["removed_time_keywords"], index=False)
    joblib.dump(vectorizers, paths["vectorizers"])
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths


def build_config(
    input_path: Path,
    output_dir: Path,
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    group_outputs = metadata["group_outputs"]
    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_recipes": len(df),
        "ingredient_column": INGREDIENT_COL,
        "keyword_column_used": metadata["keyword_column_used"],
        "category_column_used": CATEGORY_COL if CATEGORY_COL in df.columns else None,
        "yield_column_used": YIELD_COL if YIELD_COL in df.columns else None,
        "feature_groups_included": metadata["feature_groups_included"],
        "group_weights": GROUP_WEIGHTS,
        "vectorizer_settings": {
            group: output["vectorizer_settings"]
            for group, output in group_outputs.items()
        },
        "removed_time_keywords": True,
        "excluded_raw_columns": EXCLUDED_RAW_COLUMNS,
        "reason_for_excluding_raw_yield": (
            "RecipeYield mixes servings, product units, volume units, and informal text, "
            "so only normalized yield_unit_token is used."
        ),
        "reason_for_excluding_category_unknown": (
            "Missing category is metadata absence, not culinary content."
        ),
        "reason_for_separate_vectorizers": (
            "Ingredients, keywords, categories, and yield units represent different "
            "semantic sources and are weighted separately before combination."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the sparse content TF-IDF matrix for Week 5 recipes."
    )
    parser.add_argument(
        "--recipes",
        required=True,
        type=Path,
        help="Path to data/interim/recipes_resolved_features.parquet.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory, typically artifacts/week5/content_tf_idf_matrix.",
    )
    parser.add_argument(
        "--numeric-recipe-ids",
        type=Path,
        default=None,
        help="Optional Step 3 recipe_ids.csv for row-order alignment validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes_path = args.recipes
    output_dir = args.out

    df = load_recipes(recipes_path)
    input_shape = df.shape
    validate_required_columns(df, [RECIPE_ID_COL, INGREDIENT_COL])

    numeric_ids_path = resolve_numeric_recipe_ids_path(args.numeric_recipe_ids, output_dir)
    content_recipe_ids = validate_recipe_ids(df, numeric_ids_path)

    (
        X_content,
        feature_names,
        matrix_summary,
        ingredient_top,
        keyword_top,
        category_counts,
        yield_counts,
        removed_time_keywords,
        vectorizers,
        metadata,
    ) = build_content_matrix(df)

    validate_content_outputs(df, X_content, feature_names, content_recipe_ids)
    print_validation_warnings(matrix_summary, X_content)

    config = build_config(recipes_path, output_dir, df, metadata)
    paths = save_outputs(
        output_dir=output_dir,
        X_content=X_content,
        content_recipe_ids=content_recipe_ids,
        feature_names=feature_names,
        matrix_summary=matrix_summary,
        ingredient_top=ingredient_top,
        keyword_top=keyword_top,
        category_counts=category_counts,
        yield_counts=yield_counts,
        removed_time_keywords=removed_time_keywords,
        vectorizers=vectorizers,
        config=config,
    )

    def vocab_size(group: str) -> int | str:
        row = matrix_summary.loc[matrix_summary["feature_group"].eq(group)]
        if row.empty or not bool(row["included"].iloc[0]):
            return "skipped"
        return int(row["vocab_size"].iloc[0])

    print("Content TF-IDF matrix build complete.")
    print(f"Input shape: {input_shape}")
    print(f"Output directory: {output_dir}")
    print(f"Number of recipes: {len(df)}")
    print(f"Ingredient vocabulary size: {vocab_size('ingredients')}")
    print(f"Keyword vocabulary size: {vocab_size('keywords')}")
    print(f"Category vocabulary size: {vocab_size('category')}")
    print(f"Yield vocabulary size: {vocab_size('yield')}")
    print(f"Combined matrix shape: {X_content.shape}")
    print(f"Combined matrix density: {matrix_density(X_content):.8f}")
    print(f"Content matrix: {paths['matrix']}")
    print(f"Feature names: {paths['feature_names']}")
    print(f"Matrix summary: {paths['summary']}")
    print(f"Vectorizers: {paths['vectorizers']}")


if __name__ == "__main__":
    main()
