"""
Ingest Food.com Recipes and Reviews dataset and produce a cleaned, processed
version of the data suitable for downstream analytics.  This script assumes
that the dataset can be downloaded via the `kagglehub` library.  It cleans
durations encoded in ISO 8601 format, parses columns that are stored as
stringified lists, and derives a few convenience features.  Processed tables
are written into a `processed` subdirectory adjacent to the downloaded data.

Example usage::

    python ingest_foodcom_data.py --output-dir ./data

The script creates two CSV files: ``recipes_processed.csv`` and
``reviews_processed.csv``.

.. note::
    This script does not require any Kaggle credentials when the dataset
    is publicly accessible.  If Kaggle enforces authentication, configure
    the environment variables ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` before
    running this script.
"""

import argparse
import ast
import os
import shutil
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

try:
    # ``kagglehub`` is a lightweight client for downloading Kaggle datasets.
    import kagglehub
except ImportError:
    kagglehub = None  # type: ignore

try:
    import isodate
except ImportError:
    isodate = None  # type: ignore


def parse_list(cell: Any) -> List[str]:
    """Convert a string representation of a list into a Python list.

    The Food.com CSV files store certain columns (``Images``, ``Keywords``,
    ``RecipeIngredientQuantities``, ``RecipeIngredientParts``,
    ``RecipeInstructions``) as string representations.  Some of these
    strings are wrapped with the R-like ``c(...)`` syntax.  This helper
    strips the ``c(...)`` wrapper if present and safely evaluates the list.

    Parameters
    ----------
    cell : Any
        The cell value read from the CSV.  If this value is already a list,
        it is returned unchanged.  If it is NaN or missing, an empty list is
        returned.

    Returns
    -------
    List[str]
        A Python list of strings parsed from the original representation.
    """
    if isinstance(cell, list):
        return cell
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    # Remove the R wrapper ``c(...)`` if present
    if s.startswith('c(') and s.endswith(')'):
        s = s[2:-1]
    try:
        # Replace escaped quotes ``\"`` with standard quotes
        s_normalised = s.replace('\\"', '"').replace('\"', '"')
        # ast.literal_eval can parse Python-style lists; for comma-separated
        # values without surrounding brackets we add brackets manually.
        if not (s_normalised.startswith('[') or s_normalised.startswith('(')):
            s_normalised = f'[{s_normalised}]'
        value = ast.literal_eval(s_normalised)
        # Flatten nested tuples/lists and cast all elements to string
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        # If a single value is returned, wrap it in a list
        return [str(value)]
    except Exception:
        # Fallback: split on comma
        return [v.strip().strip('"') for v in s.split(',') if v]


def parse_duration(duration: Any) -> Optional[float]:
    """Parse ISO 8601 duration strings into minutes.

    The raw dataset encodes ``CookTime``, ``PrepTime``, and ``TotalTime``
    columns using ISO 8601 durations (e.g. ``PT30M`` for 30 minutes, ``PT1H``
    for one hour).  This helper uses the ``isodate`` library when available.

    Parameters
    ----------
    duration : Any
        Raw value from the CSV.

    Returns
    -------
    Optional[float]
        The duration in minutes if parseable, otherwise ``None``.
    """
    if duration is None or (isinstance(duration, float) and pd.isna(duration)):
        return None
    s = str(duration)
    if not s:
        return None
    if isodate is None:
        # ``isodate`` not installed; return None or try to parse manually
        return None
    try:
        td = isodate.parse_duration(s)
        return td.total_seconds() / 60.0  # convert seconds to minutes
    except Exception:
        return None


def ingest_foodcom(data_dir: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """Download and process the Food.com dataset.

    When ``data_dir`` is ``None``, the dataset is downloaded using ``kagglehub``.
    Processed files are written under ``output_dir`` or ``data/processed`` at
    the repository root by default.
    """
    repo_root = Path(__file__).resolve().parent.parent
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if data_dir is None:
        if kagglehub is None:
            raise ImportError(
                "kagglehub is required to download the dataset but is not installed."
            )
        data_dir = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
    data_dir_path = Path(data_dir)
    source_recipes = data_dir_path / "recipes.csv"
    source_reviews = data_dir_path / "reviews.csv"
    if not os.path.exists(source_recipes) or not os.path.exists(source_reviews):
        raise FileNotFoundError(
            f"Expected recipes.csv and reviews.csv in {data_dir_path}, but one or both files are missing."
        )

    recipes_path = raw_dir / "recipes.csv"
    reviews_path = raw_dir / "reviews.csv"
    shutil.copy2(source_recipes, recipes_path)
    shutil.copy2(source_reviews, reviews_path)
    # Read the raw tables
    recipes = pd.read_csv(recipes_path)
    reviews = pd.read_csv(reviews_path)
    # Clean list-like columns in recipes
    list_columns = [
        "Images",
        "Keywords",
        "RecipeIngredientQuantities",
        "RecipeIngredientParts",
        "RecipeInstructions",
    ]
    for col in list_columns:
        if col in recipes.columns:
            recipes[col] = recipes[col].apply(parse_list)
    # Parse durations into minutes
    duration_columns = ["CookTime", "PrepTime", "TotalTime"]
    for col in duration_columns:
        if col in recipes.columns:
            new_col = f"{col}_Minutes"
            recipes[new_col] = recipes[col].apply(parse_duration)
    # Derive additional features
    if "RecipeIngredientParts" in recipes.columns:
        recipes["NumIngredients"] = recipes["RecipeIngredientParts"].apply(lambda x: len(x))
    if "RecipeIngredientQuantities" in recipes.columns:
        recipes["NumQuantities"] = recipes["RecipeIngredientQuantities"].apply(lambda x: len(x))
    # Standardise column names to snake_case for processed outputs
    recipes.columns = [
        col.strip()
        .replace(".", "_")
        .replace(" ", "_")
        for col in recipes.columns
    ]
    reviews.columns = [
        col.strip()
        .replace(".", "_")
        .replace(" ", "_")
        for col in reviews.columns
    ]
    # Output directory
    if output_dir is None:
        processed_dir = repo_root / "data" / "processed"
    else:
        processed_dir = Path(output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    recipes_file = processed_dir / "recipes_processed.csv"
    reviews_file = processed_dir / "reviews_processed.csv"
    # Save processed tables
    recipes.to_csv(recipes_file, index=False)
    reviews.to_csv(reviews_file, index=False)
    return str(processed_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and process Food.com dataset")
    parser.add_argument(
        "--data-dir",
        help="Directory containing recipes.csv and reviews.csv (optional). If omitted, kagglehub will download the data.",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where processed data will be saved. Defaults to a 'processed' subfolder next to the data directory.",
        default=None,
    )
    args = parser.parse_args()
    ingest_foodcom(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()