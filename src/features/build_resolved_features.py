"""
Build resolved Week 5 recipe features for the Food.com recommendation project.

This script is intentionally limited to Step 2 feature resolution:
category cleaning, servings/yield resolution, time arithmetic resolution, and
yield-unit token extraction. It does not scale, clip outliers, log-transform,
or build TF-IDF/PCA/SVD representations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "RecipeId",
    "RecipeServings",
    "RecipeYield",
    "PrepTime_Minutes",
    "CookTime_Minutes",
    "TotalTime_Minutes",
]

FINAL_NON_NULL_COLUMNS = [
    "ResolvedServings_imputed",
    "CookTime_Minutes_imputed",
    "PrepTime_Minutes_imputed",
    "TotalTime_Minutes_imputed",
]

SUMMARY_SUBDIR = "resolved_features_summaries"
NUMBER_RE = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
SERVING_UNIT_RE = r"(?:servings?|serves?|people|persons?|portions?)"

YIELD_UNIT_PATTERNS = [
    ("yield__loaf", r"\bloaves\b|\bloaf\b"),
    ("yield__muffin", r"\bmuffins\b|\bmuffin\b"),
    ("yield__cookie", r"\bcookies\b|\bcookie\b"),
    ("yield__cup", r"\bcups\b|\bcup\b"),
    ("yield__pie", r"\bpies\b|\bpie\b"),
    ("yield__cake", r"\bcakes\b|\bcake\b"),
    ("yield__quart", r"\bquarts\b|\bquart\b|\bqts\b|\bqt\b"),
    ("yield__batch", r"\bbatches\b|\bbatch\b"),
    ("yield__piece", r"\bpieces\b|\bpiece\b|\bpcs\b|\bpc\b"),
    ("yield__slice", r"\bslices\b|\bslice\b"),
    ("yield__jar", r"\bjars\b|\bjar\b"),
    ("yield__bottle", r"\bbottles\b|\bbottle\b"),
    ("yield__pint", r"\bpints\b|\bpint\b|\bpts\b|\bpt\b"),
    ("yield__gallon", r"\bgallons\b|\bgallon\b|\bgals\b|\bgal\b"),
]


def percent(count: int | float, total: int) -> float:
    if total == 0:
        return 0.0
    return round(float(count) / total * 100, 4)


def require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def original_missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def clean_text_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.eq(""))


def build_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean RecipeCategory while preserving true missingness.

    Missing categories are deliberately kept as NaN. They are not imputed and
    are not converted to an "unknown" token because absence of category text is
    not culinary content.
    """
    out = df.copy()

    if "RecipeCategory" not in out.columns:
        print(
            "WARNING: RecipeCategory column is missing. Category-aware median "
            "imputation will fall back to global medians only."
        )
        out["category_missing_original"] = 1
        out["RecipeCategory_clean"] = np.nan
        return out

    missing = original_missing_mask(out["RecipeCategory"])
    cleaned = out["RecipeCategory"].astype("string").str.strip().str.lower()
    cleaned = cleaned.str.replace(r"[^\w\s]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    cleaned = cleaned.str.replace(" ", "_", regex=False)
    cleaned = cleaned.mask(missing | cleaned.eq(""))

    out["category_missing_original"] = missing.astype(int)
    out["RecipeCategory_clean"] = cleaned
    return out


def parse_number_token(token: str) -> float:
    token = token.strip()

    mixed = re.match(r"^(\d+)\s+(\d+)/(\d+)$", token)
    if mixed:
        whole = float(mixed.group(1))
        num = float(mixed.group(2))
        den = float(mixed.group(3))
        return whole + num / den

    frac = re.match(r"^(\d+)/(\d+)$", token)
    if frac:
        num = float(frac.group(1))
        den = float(frac.group(2))
        return num / den

    return float(token)


def average_range(a: str, b: str | None) -> float:
    low = parse_number_token(a)
    if b is None:
        return low
    high = parse_number_token(b)
    return (low + high) / 2.0


def parse_servings_from_yield(value: object) -> float:
    """
    Parse RecipeYield only when the number is directly attached to serving-like wording.

    This avoids taking unrelated product numbers from strings such as
    "1 9-inch pie, serves 8".
    """
    if value is None or pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if not text:
        return np.nan

    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"\s+", " ", text)

    # Case 1: "2 dozen servings"
    match = re.search(
        rf"(\d+(?:\.\d+)?)\s*(?:dozen|doz)\s*"
        rf"\b(?:servings?|people|persons?|portions?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1)) * 12.0

    # Case 2: "serves 2 dozen"
    match = re.search(
        r"\b(?:serves?)\b\s*(\d+(?:\.\d+)?)\s*(?:dozen|doz)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1)) * 12.0

    # Case 3: "serves 4", "serves 4-6", "serves 4 to 6"
    match = re.search(
        rf"\b(?:serves?|serving)\b\s*({NUMBER_RE})(?:\s*(?:-|to)\s*({NUMBER_RE}))?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return average_range(match.group(1), match.group(2))

    # Case 4: "4 servings", "4-6 servings", "4 to 6 portions", "8 people"
    match = re.search(
        rf"({NUMBER_RE})(?:\s*(?:-|to)\s*({NUMBER_RE}))?\s*"
        rf"\b(?:servings?|people|persons?|portions?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return average_range(match.group(1), match.group(2))

    return np.nan


def extract_yield_unit_token(value: object) -> str:
    """Return broad non-serving yield unit tokens for later optional text work."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""

    if not pd.isna(parse_servings_from_yield(text)):
        return ""

    for token, pattern in YIELD_UNIT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return token

    return ""


def impute_with_category_then_global(
    df: pd.DataFrame,
    value_col: str,
    category_col: str = "RecipeCategory_clean",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Impute missing values with category medians, then global median.

    Category medians are used only for rows with a known category and an
    available median in that category. Remaining missing values, including rows
    with missing categories, receive the global median. No synthetic "unknown"
    category is created.
    """
    if value_col not in df.columns:
        raise ValueError(f"Cannot impute missing value column: {value_col}")

    values = pd.to_numeric(df[value_col], errors="coerce")
    global_median = values.median(skipna=True)
    if pd.isna(global_median):
        raise ValueError(
            f"Cannot impute {value_col}: no non-missing values are available "
            "to compute a global median."
        )

    imputed = values.copy()
    missing = imputed.isna()
    category_indicator = pd.Series(0, index=df.index, dtype=int)

    if category_col in df.columns:
        categories = df[category_col]
        medians = df.groupby(category_col, dropna=True)[value_col].median()
        mapped_medians = categories.map(medians)
        use_category = missing & categories.notna() & mapped_medians.notna()
        imputed.loc[use_category] = mapped_medians.loc[use_category]
        category_indicator.loc[use_category] = 1

    use_global = imputed.isna()
    global_indicator = use_global.astype(int)
    imputed.loc[use_global] = global_median

    return imputed, category_indicator, global_indicator


def build_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["servings_missing_original"] = original_missing_mask(out["RecipeServings"]).astype(
        int
    )
    out["RecipeServings_clean"] = pd.to_numeric(
        out["RecipeServings"], errors="coerce"
    )
    out.loc[out["RecipeServings_clean"] <= 0, "RecipeServings_clean"] = np.nan

    out["RecipeYield_clean"] = clean_text_series(out["RecipeYield"])
    out["servings_parsed_from_yield"] = out["RecipeYield_clean"].apply(
        parse_servings_from_yield
    )

    # Prefer explicit RecipeServings. Use RecipeYield only when it clearly
    # refers to servings, people, or portions.
    out["ResolvedServings"] = out["RecipeServings_clean"]
    use_yield = (
        out["ResolvedServings"].isna() & out["servings_parsed_from_yield"].notna()
    )
    out.loc[use_yield, "ResolvedServings"] = out.loc[
        use_yield, "servings_parsed_from_yield"
    ]
    out["servings_from_yield"] = use_yield.astype(int)
    out["servings_missing_after_yield_parse"] = out["ResolvedServings"].isna().astype(
        int
    )

    (
        out["ResolvedServings_imputed"],
        out["servings_imputed_with_category_median"],
        out["servings_imputed_with_global_median"],
    ) = impute_with_category_then_global(out, "ResolvedServings")

    out["yield_unit_token"] = out["RecipeYield_clean"].apply(extract_yield_unit_token)
    return out


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["PrepTime_Minutes", "CookTime_Minutes", "TotalTime_Minutes"]:
        clean_col = f"{col}_clean"
        out[clean_col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[clean_col] < 0, clean_col] = np.nan

    out["preptime_missing_original"] = out["PrepTime_Minutes_clean"].isna().astype(int)
    out["cooktime_missing_original"] = out["CookTime_Minutes_clean"].isna().astype(int)
    out["totaltime_missing_original"] = out["TotalTime_Minutes_clean"].isna().astype(
        int
    )

    # Recover one missing component only when the arithmetic guarantees a
    # non-negative duration.
    out["CookTime_Minutes_resolved"] = out["CookTime_Minutes_clean"]
    derive_cook = (
        out["CookTime_Minutes_clean"].isna()
        & out["PrepTime_Minutes_clean"].notna()
        & out["TotalTime_Minutes_clean"].notna()
        & (out["TotalTime_Minutes_clean"] >= out["PrepTime_Minutes_clean"])
    )
    out.loc[derive_cook, "CookTime_Minutes_resolved"] = (
        out.loc[derive_cook, "TotalTime_Minutes_clean"]
        - out.loc[derive_cook, "PrepTime_Minutes_clean"]
    )
    out["cooktime_derived_from_total_prep"] = derive_cook.astype(int)

    out["PrepTime_Minutes_resolved"] = out["PrepTime_Minutes_clean"]
    derive_prep = (
        out["PrepTime_Minutes_clean"].isna()
        & out["CookTime_Minutes_resolved"].notna()
        & out["TotalTime_Minutes_clean"].notna()
        & (out["TotalTime_Minutes_clean"] >= out["CookTime_Minutes_resolved"])
    )
    out.loc[derive_prep, "PrepTime_Minutes_resolved"] = (
        out.loc[derive_prep, "TotalTime_Minutes_clean"]
        - out.loc[derive_prep, "CookTime_Minutes_resolved"]
    )
    out["preptime_derived_from_total_cook"] = derive_prep.astype(int)

    out["TotalTime_Minutes_resolved"] = out["TotalTime_Minutes_clean"]
    derive_total = (
        out["TotalTime_Minutes_clean"].isna()
        & out["PrepTime_Minutes_resolved"].notna()
        & out["CookTime_Minutes_resolved"].notna()
    )
    out.loc[derive_total, "TotalTime_Minutes_resolved"] = (
        out.loc[derive_total, "PrepTime_Minutes_resolved"]
        + out.loc[derive_total, "CookTime_Minutes_resolved"]
    )
    out["totaltime_derived_from_prep_cook"] = derive_total.astype(int)

    all_resolved = (
        out["PrepTime_Minutes_resolved"].notna()
        & out["CookTime_Minutes_resolved"].notna()
        & out["TotalTime_Minutes_resolved"].notna()
    )
    arithmetic_ok = np.isclose(
        out["TotalTime_Minutes_resolved"],
        out["PrepTime_Minutes_resolved"] + out["CookTime_Minutes_resolved"],
        rtol=1e-05,
        atol=1e-08,
        equal_nan=False,
    )
    out["time_arithmetic_consistent"] = (all_resolved & arithmetic_ok).astype(int)
    out["time_arithmetic_mismatch"] = (all_resolved & ~arithmetic_ok).astype(int)

    for source_col, prefix in [
        ("CookTime_Minutes_resolved", "cooktime"),
        ("PrepTime_Minutes_resolved", "preptime"),
        ("TotalTime_Minutes_resolved", "totaltime"),
    ]:
        imputed, cat_ind, global_ind = impute_with_category_then_global(out, source_col)
        target_col = source_col.replace("_resolved", "_imputed")
        out[target_col] = imputed
        out[f"{prefix}_imputed_with_category_median"] = cat_ind
        out[f"{prefix}_imputed_with_global_median"] = global_ind

    return out


def write_category_summary(df: pd.DataFrame, summary_dir: Path) -> None:
    total = len(df)
    missing = int(df["category_missing_original"].sum())
    present = total - missing
    summary = pd.DataFrame(
        [
            {
                "total_recipes": total,
                "category_missing_count": missing,
                "category_missing_percent": percent(missing, total),
                "category_present_count": present,
                "category_present_percent": percent(present, total),
            }
        ]
    )
    summary.to_csv(summary_dir / "category_missing_summary.csv", index=False)


def write_servings_summaries(df: pd.DataFrame, summary_dir: Path) -> None:
    total = len(df)
    servings_present = df["RecipeServings_clean"].notna()
    yield_present = df["RecipeYield_clean"].notna()

    summary_rows = []

    def add_count_metric(metric: str, count: int) -> None:
        summary_rows.append(
            {
                "metric": metric,
                "count": count,
                "percent_total_dataset": percent(count, total),
                "value": "",
            }
        )

    add_count_metric("total_recipes", total)
    add_count_metric("servings_only", int((servings_present & ~yield_present).sum()))
    add_count_metric("yield_only", int((~servings_present & yield_present).sum()))
    add_count_metric("both_present", int((servings_present & yield_present).sum()))
    add_count_metric("both_missing", int((~servings_present & ~yield_present).sum()))
    add_count_metric("parsed_from_yield", int(df["servings_from_yield"].sum()))
    add_count_metric(
        "still_missing_after_yield_parse",
        int(df["servings_missing_after_yield_parse"].sum()),
    )
    add_count_metric(
        "imputed_with_category_median",
        int(df["servings_imputed_with_category_median"].sum()),
    )
    add_count_metric(
        "imputed_with_global_median",
        int(df["servings_imputed_with_global_median"].sum()),
    )
    summary_rows.extend(
        [
            {
                "metric": "median_resolved_servings_before_imputation",
                "count": "",
                "percent_total_dataset": "",
                "value": float(df["ResolvedServings"].median(skipna=True)),
            },
            {
                "metric": "median_resolved_servings_after_imputation",
                "count": "",
                "percent_total_dataset": "",
                "value": float(df["ResolvedServings_imputed"].median(skipna=True)),
            },
        ]
    )
    pd.DataFrame(summary_rows).to_csv(
        summary_dir / "servings_yield_resolution_summary.csv", index=False
    )

    token_counts = (
        df.loc[df["yield_unit_token"].ne(""), "yield_unit_token"]
        .value_counts()
        .rename_axis("yield_unit_token")
        .reset_index(name="recipe_count")
    )
    token_counts.to_csv(summary_dir / "yield_unit_token_counts.csv", index=False)


def summary_row(
    metric: str,
    count: int,
    total: int,
    interpretation: str,
    relevant_total: int | None = None,
) -> dict[str, object]:
    if relevant_total is None:
        within = 100.0 if count > 0 else 0.0
    else:
        within = percent(count, relevant_total)
    return {
        "metric": metric,
        "count": count,
        "percent_total_dataset": percent(count, total),
        "percent_within_relevant_case": within,
        "interpretation": interpretation,
    }


def write_time_presence_summary(df: pd.DataFrame, summary_dir: Path) -> None:
    total = len(df)
    prep = df["PrepTime_Minutes_clean"].notna()
    cook = df["CookTime_Minutes_clean"].notna()
    total_time = df["TotalTime_Minutes_clean"].notna()

    rows = [
        (
            "all_three_present",
            int((prep & cook & total_time).sum()),
            "Prep, cook, and total time were all originally usable.",
        ),
        (
            "prep_cook_present_total_missing",
            int((prep & cook & ~total_time).sum()),
            "Total time can be recovered from prep plus cook time.",
        ),
        (
            "prep_total_present_cook_missing",
            int((prep & ~cook & total_time).sum()),
            "Cook time may be recoverable from total minus prep time.",
        ),
        (
            "cook_total_present_prep_missing",
            int((~prep & cook & total_time).sum()),
            "Prep time may be recoverable from total minus cook time.",
        ),
        (
            "only_prep_present",
            int((prep & ~cook & ~total_time).sum()),
            "Only prep time was originally usable.",
        ),
        (
            "only_cook_present",
            int((~prep & cook & ~total_time).sum()),
            "Only cook time was originally usable.",
        ),
        (
            "only_total_present",
            int((~prep & ~cook & total_time).sum()),
            "Only total time was originally usable.",
        ),
        (
            "all_three_missing",
            int((~prep & ~cook & ~total_time).sum()),
            "No usable time values were available before imputation.",
        ),
        ("total_recipes", total, "Total recipes in the input dataset."),
    ]

    pd.DataFrame(
        [
            {
                "metric": metric,
                "count": count,
                "percent_total_dataset": percent(count, total),
                "interpretation": interpretation,
            }
            for metric, count, interpretation in rows
        ]
    ).to_csv(summary_dir / "time_presence_summary.csv", index=False)


def write_time_consistency_summary(df: pd.DataFrame, summary_dir: Path) -> None:
    total = len(df)
    prep = df["PrepTime_Minutes_clean"]
    cook = df["CookTime_Minutes_clean"]
    total_time = df["TotalTime_Minutes_clean"]

    prep_present = prep.notna()
    cook_present = cook.notna()
    total_present = total_time.notna()

    all_three = prep_present & cook_present & total_present
    all_three_count = int(all_three.sum())
    total_equals = all_three & np.isclose(total_time, prep + cook, rtol=1e-05, atol=1e-08)
    total_mismatch = all_three & ~np.isclose(
        total_time, prep + cook, rtol=1e-05, atol=1e-08
    )

    cook_missing_case = prep_present & total_present & ~cook_present
    cook_missing_count = int(cook_missing_case.sum())
    prep_equals_total = cook_missing_case & np.isclose(
        prep, total_time, rtol=1e-05, atol=1e-08
    )
    prep_total_mismatch = cook_missing_case & (total_time < prep)

    prep_missing_case = cook_present & total_present & ~prep_present
    prep_missing_count = int(prep_missing_case.sum())
    cook_equals_total = prep_missing_case & np.isclose(
        cook, total_time, rtol=1e-05, atol=1e-08
    )
    cook_total_mismatch = prep_missing_case & (total_time < cook)

    recoverable_total = prep_present & cook_present & ~total_present
    recoverable_total_count = int(recoverable_total.sum())

    rows = [
        summary_row(
            "all_three_present",
            all_three_count,
            total,
            "All original clean time fields are available for arithmetic checks.",
            total,
        ),
        summary_row(
            "total_equals_prep_plus_cook",
            int(total_equals.sum()),
            total,
            "Original total time equals prep plus cook time.",
            all_three_count,
        ),
        summary_row(
            "total_mismatch_when_all_present",
            int(total_mismatch.sum()),
            total,
            "Original total time does not equal prep plus cook time.",
            all_three_count,
        ),
        summary_row(
            "cook_missing_prep_total_present",
            cook_missing_count,
            total,
            "Prep and total are available while cook time is missing.",
            total,
        ),
        summary_row(
            "prep_equals_total_when_cook_missing",
            int(prep_equals_total.sum()),
            total,
            "Cook time resolves to zero because prep equals total.",
            cook_missing_count,
        ),
        summary_row(
            "prep_total_mismatch_when_cook_missing",
            int(prep_total_mismatch.sum()),
            total,
            "Cook time cannot be derived because total is less than prep.",
            cook_missing_count,
        ),
        summary_row(
            "prep_missing_cook_total_present",
            prep_missing_count,
            total,
            "Cook and total are available while prep time is missing.",
            total,
        ),
        summary_row(
            "cook_equals_total_when_prep_missing",
            int(cook_equals_total.sum()),
            total,
            "Prep time resolves to zero because cook equals total.",
            prep_missing_count,
        ),
        summary_row(
            "cook_total_mismatch_when_prep_missing",
            int(cook_total_mismatch.sum()),
            total,
            "Prep time cannot be derived because total is less than cook.",
            prep_missing_count,
        ),
        summary_row(
            "recoverable_total_missing",
            recoverable_total_count,
            total,
            "Total time can be derived from prep plus cook time.",
            total,
        ),
    ]
    pd.DataFrame(rows).to_csv(summary_dir / "time_consistency_summary.csv", index=False)


def write_time_derivation_summary(df: pd.DataFrame, summary_dir: Path) -> None:
    total = len(df)
    rows = [
        {
            "metric": "can_derive_cooktime",
            "count": int(df["cooktime_derived_from_total_prep"].sum()),
            "percent_total_dataset": percent(
                int(df["cooktime_derived_from_total_prep"].sum()), total
            ),
            "rule": "CookTime = TotalTime - PrepTime when CookTime is missing and TotalTime >= PrepTime.",
        },
        {
            "metric": "can_derive_preptime",
            "count": int(df["preptime_derived_from_total_cook"].sum()),
            "percent_total_dataset": percent(
                int(df["preptime_derived_from_total_cook"].sum()), total
            ),
            "rule": "PrepTime = TotalTime - CookTime when PrepTime is missing and TotalTime >= CookTime.",
        },
        {
            "metric": "can_derive_totaltime",
            "count": int(df["totaltime_derived_from_prep_cook"].sum()),
            "percent_total_dataset": percent(
                int(df["totaltime_derived_from_prep_cook"].sum()), total
            ),
            "rule": "TotalTime = PrepTime + CookTime when TotalTime is missing and both components exist.",
        },
    ]
    pd.DataFrame(rows).to_csv(summary_dir / "time_derivation_summary.csv", index=False)


def write_time_imputation_summary(df: pd.DataFrame, summary_dir: Path) -> None:
    total = len(df)
    rows = []
    specs = [
        (
            "CookTime_Minutes",
            "CookTime_Minutes_clean",
            "CookTime_Minutes_resolved",
            "CookTime_Minutes_imputed",
            "cooktime_imputed_with_category_median",
            "cooktime_imputed_with_global_median",
        ),
        (
            "PrepTime_Minutes",
            "PrepTime_Minutes_clean",
            "PrepTime_Minutes_resolved",
            "PrepTime_Minutes_imputed",
            "preptime_imputed_with_category_median",
            "preptime_imputed_with_global_median",
        ),
        (
            "TotalTime_Minutes",
            "TotalTime_Minutes_clean",
            "TotalTime_Minutes_resolved",
            "TotalTime_Minutes_imputed",
            "totaltime_imputed_with_category_median",
            "totaltime_imputed_with_global_median",
        ),
    ]
    for metric, clean_col, resolved_col, imputed_col, cat_col, global_col in specs:
        original_missing = int(df[clean_col].isna().sum())
        resolved_missing = int(df[resolved_col].isna().sum())
        final_missing = int(df[imputed_col].isna().sum())
        rows.append(
            {
                "metric": metric,
                "original_missing_count": original_missing,
                "resolved_missing_count": resolved_missing,
                "imputed_with_category_median_count": int(df[cat_col].sum()),
                "imputed_with_global_median_count": int(df[global_col].sum()),
                "final_missing_count": final_missing,
                "original_missing_percent": percent(original_missing, total),
                "resolved_missing_percent": percent(resolved_missing, total),
                "final_missing_percent": percent(final_missing, total),
            }
        )
    pd.DataFrame(rows).to_csv(summary_dir / "time_imputation_summary.csv", index=False)


def write_summary_files(df: pd.DataFrame, summary_dir: Path) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_category_summary(df, summary_dir)
    write_servings_summaries(df, summary_dir)
    write_time_presence_summary(df, summary_dir)
    write_time_consistency_summary(df, summary_dir)
    write_time_derivation_summary(df, summary_dir)
    write_time_imputation_summary(df, summary_dir)


def resolve_summary_dir(summary_out: Path) -> Path:
    """Treat --summary-out as the Week 5 parent unless the subdir is explicit."""
    if summary_out.name == SUMMARY_SUBDIR:
        return summary_out
    return summary_out / SUMMARY_SUBDIR


def validate_output(input_df: pd.DataFrame, output_df: pd.DataFrame) -> None:
    if len(input_df) != len(output_df):
        raise ValueError(
            f"Row count changed: input={len(input_df)}, output={len(output_df)}"
        )

    input_unique = input_df["RecipeId"].nunique(dropna=False)
    output_unique = output_df["RecipeId"].nunique(dropna=False)
    if input_unique != output_unique:
        raise ValueError(
            "RecipeId uniqueness was not preserved: "
            f"input unique={input_unique}, output unique={output_unique}"
        )

    null_counts = output_df[FINAL_NON_NULL_COLUMNS].isna().sum()
    if int(null_counts.sum()) > 0:
        raise ValueError(
            "Final imputed columns contain nulls:\n"
            + null_counts[null_counts > 0].to_string()
        )

    negative_counts = (output_df[FINAL_NON_NULL_COLUMNS] < 0).sum()
    if int(negative_counts.sum()) > 0:
        raise ValueError(
            "Final imputed columns contain negative values:\n"
            + negative_counts[negative_counts > 0].to_string()
        )

    mismatch_count = int(output_df["time_arithmetic_mismatch"].sum())
    if mismatch_count > 0:
        print(
            "WARNING: time_arithmetic_mismatch is greater than zero: "
            f"{mismatch_count} rows."
        )


def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False, engine="pyarrow")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Saving Parquet requires pyarrow. Install it with "
            "`pip install pyarrow` and rerun this script."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resolved/imputed Week 5 recipe features."
    )
    parser.add_argument(
        "--recipes",
        required=True,
        type=Path,
        help="Path to data/processed/recipes_processed.csv.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path for the resolved Parquet dataset.",
    )
    parser.add_argument(
        "--summary-out",
        required=True,
        type=Path,
        help=(
            "Week 5 summary parent directory. CSVs are written under "
            "resolved_features_summaries/ inside this directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes_path = args.recipes
    out_path = args.out
    summary_dir = resolve_summary_dir(args.summary_out)

    df = pd.read_csv(recipes_path)
    input_shape = df.shape
    original_columns = set(df.columns)

    require_columns(df, REQUIRED_COLUMNS)

    resolved = build_category_features(df)
    resolved = build_serving_features(resolved)
    resolved = build_time_features(resolved)

    validate_output(df, resolved)
    write_summary_files(resolved, summary_dir)
    save_parquet(resolved, out_path)

    category_missing_count = int(resolved["category_missing_original"].sum())
    category_missing_percent = percent(category_missing_count, len(resolved))
    new_column_count = len(set(resolved.columns) - original_columns)

    print("Resolved feature build complete.")
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {resolved.shape}")
    print(f"New columns created: {new_column_count}")
    print(f"Saved parquet: {out_path}")
    print(f"Saved summary files: {summary_dir}")
    print(
        "RecipeCategory missing: "
        f"{category_missing_count} ({category_missing_percent}%)"
    )
    print(
        "ResolvedServings final missing count: "
        f"{int(resolved['ResolvedServings_imputed'].isna().sum())}"
    )
    print(
        "CookTime/PrepTime/TotalTime final missing counts: "
        f"{int(resolved['CookTime_Minutes_imputed'].isna().sum())}/"
        f"{int(resolved['PrepTime_Minutes_imputed'].isna().sum())}/"
        f"{int(resolved['TotalTime_Minutes_imputed'].isna().sum())}"
    )


if __name__ == "__main__":
    main()
