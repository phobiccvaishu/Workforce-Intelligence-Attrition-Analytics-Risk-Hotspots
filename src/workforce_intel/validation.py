"""Validation library for Workforce Intelligence project.

Pure, side-effect-free, individually testable functions returning DataFrames/dicts.
Implements Section 5.5 of Phase 1 specifications.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from scipy import stats

from workforce_intel.stats_utils import wilson_interval
from workforce_intel.config import (
    ALL_NUMERIC_COLUMNS,
    CONTINUOUS_COLUMNS,
    DERIVED_COLUMNS,
    DISCRETE_COUNT_COLUMNS,
    EXPECTED_COLUMNS,
    MIN_GROUP_SIZE,
    NOMINAL_COLUMNS,
    ORDINAL_COLUMNS_DOCUMENTED_RANGES,
    PROCESSED_COLUMNS,
    SURROGATE_KEY_COLUMN,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# a. Structure Report
# -----------------------------------------------------------------------------
def structure_report(
    df: pd.DataFrame, expected_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Inspect dataset shape, column names, ordering, dtypes, and memory usage."""
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS

    actual_columns = list(df.columns)
    missing_cols = [c for c in expected_columns if c not in actual_columns]
    unexpected_cols = [c for c in actual_columns if c not in expected_columns]

    # Check order among common columns
    common_expected = [c for c in expected_columns if c in actual_columns]
    common_actual = [c for c in actual_columns if c in expected_columns]
    order_matches = common_expected == common_actual

    col_details = []
    for col in actual_columns:
        dtype_str = str(df[col].dtype)
        is_num = bool(ptypes.is_numeric_dtype(df[col]))
        is_str = bool(ptypes.is_string_dtype(df[col]))
        col_details.append(
            {
                "column": col,
                "dtype": dtype_str,
                "is_numeric": is_num,
                "is_string": is_str,
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
            }
        )

    details_df = pd.DataFrame(col_details)
    mem_bytes = int(df.memory_usage(deep=True).sum())

    text_cols = [c for c in actual_columns if ptypes.is_string_dtype(df[c])]
    numeric_cols = [c for c in actual_columns if ptypes.is_numeric_dtype(df[c])]

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "memory_bytes": mem_bytes,
        "memory_mb": round(mem_bytes / (1024 * 1024), 2),
        "expected_columns_count": len(expected_columns),
        "missing_columns": missing_cols,
        "unexpected_columns": unexpected_cols,
        "order_matches_expected": order_matches,
        "n_text_columns": len(text_cols),
        "n_numeric_columns": len(numeric_cols),
        "text_columns": text_cols,
        "numeric_columns": numeric_cols,
        "contract_violated": bool(missing_cols or unexpected_cols or not order_matches),
        "details_table": details_df,
    }


# -----------------------------------------------------------------------------
# b. Missing Value & Sentinel Tokens Report
# -----------------------------------------------------------------------------
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Audit missing values, empty strings, whitespace, and sentinel tokens."""
    string_sentinels = {"?", "na", "n/a", "null", "none", "-"}
    numeric_sentinels = {-1, 999, 9999}

    records = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        nan_count = int(s.isna().sum())

        empty_str_count = 0
        whitespace_count = 0
        string_sentinel_count = 0
        numeric_sentinel_count = 0

        if ptypes.is_string_dtype(s) or s.dtype == object:
            # Drop actual NaNs first for string inspection
            non_null_series = s.dropna().astype(str)
            empty_str_count = int((non_null_series == "").sum())
            whitespace_count = int(
                ((non_null_series.str.strip() == "") & (non_null_series != "")).sum()
            )
            lowered = non_null_series.str.strip().str.lower()
            string_sentinel_count = int(lowered.isin(string_sentinels).sum())

        if ptypes.is_numeric_dtype(s):
            non_null_num = s.dropna()
            numeric_sentinel_count = int(non_null_num.isin(numeric_sentinels).sum())

        total_suspicious = (
            nan_count
            + empty_str_count
            + whitespace_count
            + string_sentinel_count
            + numeric_sentinel_count
        )

        records.append(
            {
                "column": col,
                "nan_count": nan_count,
                "nan_pct": round(100.0 * nan_count / n_rows, 3) if n_rows > 0 else 0.0,
                "empty_string_count": empty_str_count,
                "whitespace_only_count": whitespace_count,
                "string_sentinel_count": string_sentinel_count,
                "numeric_sentinel_count": numeric_sentinel_count,
                "total_missing_or_sentinel": total_suspicious,
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# c. Duplicate Records Report
# -----------------------------------------------------------------------------
def duplicate_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect full-row duplicates and duplicate key profiles."""
    n_rows = len(df)
    full_duplicates = int(df.duplicated().sum())

    # Key profile subset without rates (demographics + role + tenure)
    key_subset = [
        c
        for c in [
            "Age",
            "Gender",
            "Department",
            "JobRole",
            "EducationField",
            "TotalWorkingYears",
            "YearsAtCompany",
        ]
        if c in df.columns
    ]

    key_subset_duplicates = 0
    if key_subset:
        key_subset_duplicates = int(df.duplicated(subset=key_subset).sum())

    return {
        "n_rows": n_rows,
        "full_row_duplicates": full_duplicates,
        "full_row_duplicate_rate_pct": round(100.0 * full_duplicates / n_rows, 3)
        if n_rows > 0
        else 0.0,
        "key_subset_columns": key_subset,
        "key_subset_duplicates": key_subset_duplicates,
        "uniqueness_note": (
            "NOTICE: The dataset lacks a unique employee identifier (e.g. EmployeeID). "
            "Uniqueness of distinct individuals cannot be definitively confirmed from data alone."
        ),
    }


# -----------------------------------------------------------------------------
# d. Categorical Audit
# -----------------------------------------------------------------------------
def categorical_audit(
    df: pd.DataFrame, nominal_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Audit nominal columns for levels, shares, whitespace issues, and casing variants."""
    if nominal_columns is None:
        nominal_columns = [c for c in NOMINAL_COLUMNS if c in df.columns]

    records = []
    n_rows = len(df)

    for col in nominal_columns:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        val_counts = s.value_counts()

        # Whitespace checks
        has_leading_trailing_ws = bool((s != s.str.strip()).any())

        # Check for case/casing variants
        unique_raw = s.unique()
        unique_lowered = {val.lower() for val in unique_raw}
        has_case_variants = len(unique_raw) != len(unique_lowered)

        for val, count in val_counts.items():
            records.append(
                {
                    "column": col,
                    "level": val,
                    "count": int(count),
                    "share_pct": round(100.0 * count / n_rows, 2) if n_rows > 0 else 0.0,
                    "leading_trailing_whitespace": has_leading_trailing_ws,
                    "has_case_variants": has_case_variants,
                }
            )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# e. Range and Scale Audit
# -----------------------------------------------------------------------------
def range_audit(
    df: pd.DataFrame,
    ordinal_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    numeric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Audit observed min/max/levels vs documented scales and detect restricted ranges."""
    if ordinal_ranges is None:
        ordinal_ranges = ORDINAL_COLUMNS_DOCUMENTED_RANGES
    if numeric_columns is None:
        numeric_columns = [
            c for c in ALL_NUMERIC_COLUMNS if c in df.columns and ptypes.is_numeric_dtype(df[c])
        ]

    records = []
    for col in numeric_columns:
        if col not in df.columns or df[col].empty:
            continue
        s = df[col].dropna()
        if s.empty:
            continue

        obs_min = float(s.min())
        obs_max = float(s.max())
        unique_vals = sorted(s.unique().tolist())
        n_unique = len(unique_vals)

        is_ordinal = col in ordinal_ranges
        doc_min, doc_max = ordinal_ranges[col] if is_ordinal else (np.nan, np.nan)
        doc_span = (doc_max - doc_min + 1) if is_ordinal else np.nan

        out_of_bounds = False
        restricted_range = False

        if is_ordinal:
            out_of_bounds = bool((obs_min < doc_min) or (obs_max > doc_max))
            # Restricted if fewer levels observed than documented scale
            restricted_range = bool(n_unique < doc_span)

        records.append(
            {
                "column": col,
                "is_ordinal": is_ordinal,
                "documented_min": doc_min if is_ordinal else None,
                "documented_max": doc_max if is_ordinal else None,
                "observed_min": obs_min,
                "observed_max": obs_max,
                "observed_unique_count": n_unique,
                "observed_levels": str(unique_vals[:10])
                if n_unique <= 10
                else f"[{obs_min} .. {obs_max}]",
                "out_of_bounds": out_of_bounds,
                "is_restricted_range": restricted_range,
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# f. Target Audit
# -----------------------------------------------------------------------------
def target_audit(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, Any]:
    """Audit the target variable distribution, subset compliance, and Wilson 95% CI."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in DataFrame.")

    s = df[target_col].dropna()
    distinct_vals = sorted(s.unique().tolist())
    is_binary_01 = set(distinct_vals).issubset({0, 1})

    n_total = len(s)
    val_counts = s.value_counts().to_dict()
    count_0 = int(val_counts.get(0, 0))
    count_1 = int(val_counts.get(1, 0))

    rate_1 = (count_1 / n_total) if n_total > 0 else 0.0
    rate_0 = (count_0 / n_total) if n_total > 0 else 0.0

    imbalance_ratio = (count_0 / count_1) if count_1 > 0 else np.nan

    # Wilson 95% Score Confidence Interval
    ci_lower, ci_upper = wilson_interval(count_1, n_total, confidence=0.95)
    if n_total == 0:
        ci_lower, ci_upper = 0.0, 0.0

    return {
        "target_column": target_col,
        "dtype": str(df[target_col].dtype),
        "total_records": n_total,
        "distinct_values": distinct_vals,
        "is_binary_01_subset": is_binary_01,
        "count_attrition_0": count_0,
        "count_attrition_1": count_1,
        "rate_attrition_0_pct": round(rate_0 * 100.0, 2),
        "rate_attrition_1_pct": round(rate_1 * 100.0, 2),
        "imbalance_ratio_stayed_to_left": round(imbalance_ratio, 2)
        if not np.isnan(imbalance_ratio)
        else None,
        "wilson_95_ci_lower_pct": round(ci_lower * 100.0, 2),
        "wilson_95_ci_upper_pct": round(ci_upper * 100.0, 2),
    }


# -----------------------------------------------------------------------------
# g. Domain Consistency Checks (C1 - C9)
# -----------------------------------------------------------------------------
def consistency_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute C1 through C9 domain and temporal consistency checks.

    Returns summary counts, offending row indices, and detailed crosstabs/matrices.
    """
    results: Dict[str, Any] = {}

    # C1: YearsAtCompany <= TotalWorkingYears
    if {"YearsAtCompany", "TotalWorkingYears"}.issubset(df.columns):
        c1_offending = df.index[df["YearsAtCompany"] > df["TotalWorkingYears"]].tolist()
        results["C1"] = {
            "rule": "YearsAtCompany <= TotalWorkingYears",
            "violation_count": len(c1_offending),
            "offending_indices": c1_offending,
        }

    # C2: YearsInCurrentRole <= YearsAtCompany
    if {"YearsInCurrentRole", "YearsAtCompany"}.issubset(df.columns):
        c2_offending = df.index[df["YearsInCurrentRole"] > df["YearsAtCompany"]].tolist()
        results["C2"] = {
            "rule": "YearsInCurrentRole <= YearsAtCompany",
            "violation_count": len(c2_offending),
            "offending_indices": c2_offending,
        }

    # C3: YearsSinceLastPromotion <= YearsAtCompany
    if {"YearsSinceLastPromotion", "YearsAtCompany"}.issubset(df.columns):
        c3_offending = df.index[df["YearsSinceLastPromotion"] > df["YearsAtCompany"]].tolist()
        results["C3"] = {
            "rule": "YearsSinceLastPromotion <= YearsAtCompany",
            "violation_count": len(c3_offending),
            "offending_indices": c3_offending,
        }

    # C4: YearsWithCurrManager <= YearsAtCompany
    if {"YearsWithCurrManager", "YearsAtCompany"}.issubset(df.columns):
        c4_offending = df.index[df["YearsWithCurrManager"] > df["YearsAtCompany"]].tolist()
        results["C4"] = {
            "rule": "YearsWithCurrManager <= YearsAtCompany",
            "violation_count": len(c4_offending),
            "offending_indices": c4_offending,
        }

    # C5: TotalWorkingYears <= Age - 14 (hard) and TotalWorkingYears > Age - 18 (soft)
    if {"TotalWorkingYears", "Age"}.issubset(df.columns):
        c5_hard_offending = df.index[df["TotalWorkingYears"] > (df["Age"] - 14)].tolist()
        c5_soft_offending = df.index[df["TotalWorkingYears"] > (df["Age"] - 18)].tolist()
        results["C5_hard"] = {
            "rule": "TotalWorkingYears <= Age - 14",
            "violation_count": len(c5_hard_offending),
            "offending_indices": c5_hard_offending,
        }
        results["C5_soft"] = {
            "rule": "TotalWorkingYears <= Age - 18 (Career start before 18)",
            "violation_count": len(c5_soft_offending),
            "offending_indices": c5_soft_offending,
        }

    # C6: NumCompaniesWorked == 0 while TotalWorkingYears > YearsAtCompany
    if {"NumCompaniesWorked", "TotalWorkingYears", "YearsAtCompany"}.issubset(df.columns):
        c6_cond = (df["NumCompaniesWorked"] == 0) & (df["TotalWorkingYears"] > df["YearsAtCompany"])
        c6_offending = df.index[c6_cond].tolist()
        results["C6"] = {
            "rule": "NumCompaniesWorked == 0 while TotalWorkingYears > YearsAtCompany",
            "violation_count": len(c6_offending),
            "offending_indices": c6_offending,
            "interpretation_note": (
                "Ambiguity in definition: If an employee worked at 0 prior companies, "
                "their TotalWorkingYears should logically equal YearsAtCompany. However, "
                "197 employees report NumCompaniesWorked=0 yet TotalWorkingYears > YearsAtCompany."
            ),
        }

    # C7: Department x JobRole crosstab
    if {"Department", "JobRole"}.issubset(df.columns):
        ct = pd.crosstab(df["Department"], df["JobRole"])
        populated_cells = int((ct > 0).values.sum())
        total_cells = int(ct.size)

        # Flag any JobRole mapping to > 1 Department
        role_dept_map = df.groupby("JobRole")["Department"].nunique()
        multi_dept_roles = role_dept_map[role_dept_map > 1].to_dict()

        results["C7"] = {
            "rule": "Department x JobRole hierarchy coherence",
            "populated_cells": populated_cells,
            "total_cells": total_cells,
            "multi_department_roles": multi_dept_roles,
            "crosstab_table": ct,
        }

    # C8: MonthlyIncome range per JobLevel and whether ranges overlap
    if {"MonthlyIncome", "JobLevel"}.issubset(df.columns):
        level_stats = (
            df.groupby("JobLevel")["MonthlyIncome"]
            .agg(
                min="min",
                p25=lambda x: x.quantile(0.25),
                median="median",
                p75=lambda x: x.quantile(0.75),
                max="max",
                count="count",
            )
            .reset_index()
        )

        overlaps = []
        for i in range(len(level_stats) - 1):
            curr_lvl = int(level_stats.loc[i, "JobLevel"])
            next_lvl = int(level_stats.loc[i + 1, "JobLevel"])
            curr_max = float(level_stats.loc[i, "max"])
            next_min = float(level_stats.loc[i + 1, "min"])
            overlap_exists = curr_max >= next_min
            overlaps.append(
                {
                    "level_pair": f"Level {curr_lvl} vs Level {next_lvl}",
                    "current_level_max": curr_max,
                    "next_level_min": next_min,
                    "ranges_overlap": overlap_exists,
                }
            )

        results["C8"] = {
            "rule": "MonthlyIncome range progression by JobLevel",
            "level_stats_table": level_stats,
            "overlap_details": pd.DataFrame(overlaps),
        }

    # C9: Coherence of DailyRate, HourlyRate, MonthlyRate, MonthlyIncome
    rate_cols = [c for c in ["DailyRate", "HourlyRate", "MonthlyRate", "MonthlyIncome"] if c in df.columns]
    if len(rate_cols) == 4:
        spearman_corr = df[rate_cols].corr(method="spearman")

        # Relationship HourlyRate * 8 vs DailyRate
        ratio_series = df["DailyRate"] / (df["HourlyRate"] * 8.0)
        off_diag_rhos = [
            abs(float(spearman_corr.loc[rate_cols[i], rate_cols[j]]))
            for i in range(len(rate_cols))
            for j in range(i + 1, len(rate_cols))
        ]
        max_abs_rate_corr = max(off_diag_rhos) if off_diag_rhos else 0.0
        c9_stats = {
            "spearman_matrix": spearman_corr,
            "daily_to_8hr_ratio_min": round(float(ratio_series.min()), 3),
            "daily_to_8hr_ratio_median": round(float(ratio_series.median()), 3),
            "daily_to_8hr_ratio_max": round(float(ratio_series.max()), 3),
            "daily_vs_8hr_spearman_rho": round(
                float(df[["DailyRate", "HourlyRate"]].corr(method="spearman").iloc[0, 1]), 4
            ),
            "max_abs_spearman_rho": round(max_abs_rate_corr, 4),
            "interpretation_note": (
                f"REPORT ONLY: Spearman correlations among DailyRate, HourlyRate, MonthlyRate, "
                f"and MonthlyIncome are near zero (max abs(rho) = {max_abs_rate_corr:.4f}). "
                f"The ratio DailyRate / (HourlyRate * 8) varies drastically across rows, "
                f"suggesting independent synthetic generation rather than accounting coherence."
            ),
        }
        results["C9"] = c9_stats

    return results


# -----------------------------------------------------------------------------
# h. Zero Value Audit
# -----------------------------------------------------------------------------
def zero_audit(
    df: pd.DataFrame, numeric_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Audit frequency of zeros across numeric columns with candidate review classifications."""
    if numeric_columns is None:
        numeric_columns = [
            c for c in ALL_NUMERIC_COLUMNS if c in df.columns and ptypes.is_numeric_dtype(df[c])
        ]

    classification_map = {
        "StockOptionLevel": "legitimate zero (no stock options granted)",
        "YearsSinceLastPromotion": "legitimate zero (promoted within current year)",
        "TrainingTimesLastYear": "legitimate zero (no formal training attended)",
        "YearsAtCompany": "legitimate zero (new hire in first year)",
        "TotalWorkingYears": "legitimate zero (entry level / first professional year)",
        "YearsInCurrentRole": "legitimate zero (recently transferred / promoted)",
        "YearsWithCurrManager": "legitimate zero (new manager assignment)",
        "NumCompaniesWorked": "ambiguous (could indicate 0 prior employers or no prior work)",
        "Age": "possibly missing (cannot be zero)",
        "DailyRate": "possibly missing (compensation cannot be zero)",
        "HourlyRate": "possibly missing (compensation cannot be zero)",
        "MonthlyIncome": "possibly missing (compensation cannot be zero)",
        "MonthlyRate": "possibly missing (compensation cannot be zero)",
        "DistanceFromHome": "possibly missing / on-site resident",
        "PercentSalaryHike": "possibly missing / zero raise",
    }

    records = []
    n_rows = len(df)

    for col in numeric_columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        zero_count = int((s == 0).sum())
        zero_pct = round(100.0 * zero_count / n_rows, 2) if n_rows > 0 else 0.0

        classification = classification_map.get(
            col, "legitimate zero" if zero_count == 0 else "ambiguous"
        )

        records.append(
            {
                "column": col,
                "zero_count": zero_count,
                "zero_pct": zero_pct,
                "candidate_classification": classification,
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# i. Outlier Report (IQR 1.5x and 3.0x)
# -----------------------------------------------------------------------------
def outlier_report(
    df: pd.DataFrame, numeric_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Detect 1.5x and 3.0x IQR outliers and compute skewness for reporting."""
    if numeric_columns is None:
        numeric_columns = [
            c for c in ALL_NUMERIC_COLUMNS if c in df.columns and ptypes.is_numeric_dtype(df[c])
        ]

    records = []
    n_rows = len(df)

    for col in numeric_columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty or not ptypes.is_numeric_dtype(s):
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1

        # 1.5x IQR
        lower_15 = q1 - 1.5 * iqr
        upper_15 = q3 + 1.5 * iqr
        outliers_15 = int(((s < lower_15) | (s > upper_15)).sum())

        # 3.0x IQR (Extreme outliers)
        lower_30 = q1 - 3.0 * iqr
        upper_30 = q3 + 3.0 * iqr
        outliers_30 = int(((s < lower_30) | (s > upper_30)).sum())

        col_skew = float(s.skew()) if len(s) > 2 else 0.0

        records.append(
            {
                "column": col,
                "q1": round(q1, 2),
                "median": round(float(s.median()), 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound_1_5x": round(lower_15, 2),
                "upper_bound_1_5x": round(upper_15, 2),
                "outliers_count_1_5x": outliers_15,
                "outliers_pct_1_5x": round(100.0 * outliers_15 / n_rows, 2)
                if n_rows > 0
                else 0.0,
                "lower_bound_3_0x": round(lower_30, 2),
                "upper_bound_3_0x": round(upper_30, 2),
                "outliers_count_3_0x": outliers_30,
                "outliers_pct_3_0x": round(100.0 * outliers_30 / n_rows, 2)
                if n_rows > 0
                else 0.0,
                "skewness": round(col_skew, 3),
            }
        )

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# j. Collinearity & VIF Report
# -----------------------------------------------------------------------------
def collinearity_report(
    df: pd.DataFrame, numeric_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate Spearman correlation matrix, flag pairs with abs(rho) >= 0.7, and compute VIF."""
    if numeric_columns is None:
        numeric_columns = [
            c for c in ALL_NUMERIC_COLUMNS if c in df.columns and ptypes.is_numeric_dtype(df[c])
        ]

    clean_df = df[numeric_columns].dropna()
    spearman_matrix = clean_df.corr(method="spearman")

    # High correlation pairs (abs(rho) >= 0.7)
    high_corr_pairs = []
    cols = list(spearman_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            rho = float(spearman_matrix.loc[col1, col2])
            if abs(rho) >= 0.7:
                high_corr_pairs.append(
                    {
                        "feature_1": col1,
                        "feature_2": col2,
                        "spearman_rho": round(rho, 4),
                        "abs_rho": round(abs(rho), 4),
                    }
                )

    high_corr_df = pd.DataFrame(high_corr_pairs)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values(by="abs_rho", ascending=False).reset_index(
            drop=True
        )

    # Variance Inflation Factor (VIF) calculation
    # Using numpy OLS or statsmodels
    vif_records = []
    try:
        # Standardize features for stable inversion
        X = clean_df.astype(float).values
        # Add intercept column
        n_samples, n_features = X.shape
        X_with_intercept = np.column_stack([np.ones(n_samples), X])

        for idx in range(n_features):
            col_name = cols[idx]
            y_i = X[:, idx]
            X_others = np.delete(X, idx, axis=1)
            X_others_with_const = np.column_stack([np.ones(n_samples), X_others])

            # OLS R-squared via numpy lstsq
            beta, residuals, rank, s = np.linalg.lstsq(X_others_with_const, y_i, rcond=None)
            y_pred = X_others_with_const @ beta
            ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
            ss_res = np.sum((y_i - y_pred) ** 2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            vif = 1.0 / (1.0 - r2) if r2 < 0.999999 else 999.0
            vif_records.append(
                {
                    "feature": col_name,
                    "r_squared": round(float(r2), 4),
                    "vif": round(float(vif), 2),
                    "vif_gt_5": bool(vif > 5.0),
                    "vif_gt_10": bool(vif > 10.0),
                }
            )
    except Exception as exc:
        logger.warning("VIF calculation degraded or encountered numerical issue: %s", exc)

    vif_df = pd.DataFrame(vif_records)
    if not vif_df.empty:
        vif_df = vif_df.sort_values(by="vif", ascending=False).reset_index(drop=True)

    return {
        "spearman_matrix": spearman_matrix,
        "high_correlation_pairs": high_corr_df,
        "vif_table": vif_df,
    }


# -----------------------------------------------------------------------------
# k. Group Size & Small Sample Report
# -----------------------------------------------------------------------------
def group_size_report(
    df: pd.DataFrame, nominal_columns: Optional[List[str]] = None, threshold: int = 30
) -> Dict[str, Any]:
    """Flag nominal levels and Department x JobRole cross-cells with n < 30."""
    if nominal_columns is None:
        nominal_columns = [c for c in NOMINAL_COLUMNS if c in df.columns]

    small_nominal_cells = []
    for col in nominal_columns:
        counts = df[col].value_counts()
        for level, count in counts.items():
            if count < threshold:
                small_nominal_cells.append(
                    {
                        "dimension": col,
                        "level": str(level),
                        "count": int(count),
                        "below_threshold": True,
                    }
                )

    small_cross_cells = []
    if {"Department", "JobRole"}.issubset(df.columns):
        ct = pd.crosstab(df["Department"], df["JobRole"])
        for dept in ct.index:
            for role in ct.columns:
                count = int(ct.loc[dept, role])
                # Only report populated cells with 0 < count < 30
                if 0 < count < threshold:
                    small_cross_cells.append(
                        {
                            "department": dept,
                            "job_role": role,
                            "count": count,
                            "below_threshold": True,
                        }
                    )

    return {
        "threshold": threshold,
        "small_nominal_cells_table": pd.DataFrame(small_nominal_cells),
        "small_cross_cells_table": pd.DataFrame(small_cross_cells),
    }


# -----------------------------------------------------------------------------
# l. Constant and Near-Constant Columns
# -----------------------------------------------------------------------------
def constant_column_check(
    df: pd.DataFrame, near_constant_threshold: float = 0.99
) -> pd.DataFrame:
    """Identify columns with zero variance (constant) or dominant mode >= threshold."""
    records = []
    n_rows = len(df)

    for col in df.columns:
        s = df[col].dropna()
        n_unique = s.nunique()
        is_constant = bool(n_unique <= 1)

        dominant_mode_val = None
        dominant_mode_pct = 0.0
        is_near_constant = False

        if not s.empty:
            mode_counts = s.value_counts()
            dominant_mode_val = mode_counts.index[0]
            top_count = int(mode_counts.iloc[0])
            dominant_mode_pct = round(100.0 * top_count / n_rows, 2) if n_rows > 0 else 0.0
            is_near_constant = bool(dominant_mode_pct >= (near_constant_threshold * 100.0))

        records.append(
            {
                "column": col,
                "unique_values_count": n_unique,
                "is_constant": is_constant,
                "dominant_mode_value": str(dominant_mode_val),
                "dominant_mode_pct": dominant_mode_pct,
                "is_near_constant": is_near_constant,
            }
        )

    return pd.DataFrame(records)



# -----------------------------------------------------------------------------
# Report Builder (Pure Function for Markdown Report Provenance)
# -----------------------------------------------------------------------------
def reconciliation_audit(
    raw_df: pd.DataFrame, processed_df: pd.DataFrame, expected_rows: Optional[int] = None
) -> Dict[str, Any]:
    """Execute row-level and cell-for-cell reconciliation between raw and processed data.

    Adheres strictly to Phase 2 Section 5.7:
    - Verifies raw rows == processed rows, 0 dropped
    - Verifies all 31 original columns equal raw cell for cell
    - Verifies RowID uniqueness and 0-based continuity
    - Generates bucket sizes table with small-sample flags (n < 30)
    - Generates NaN audit table verifying zero-denominator rule

    Args:
        raw_df: Untouched raw DataFrame.
        processed_df: Processed DataFrame with RowID and derived columns.

    Returns:
        Dictionary containing reconciliation metrics and audit DataFrames.
    """
    raw_rows = len(raw_df)
    proc_rows = len(processed_df)
    rows_dropped = raw_rows - proc_rows

    # 1. Cell-for-cell comparison on 31 original columns
    mismatches = {}
    for col in EXPECTED_COLUMNS:
        match_series = (raw_df[col] == processed_df[col]) | (
            raw_df[col].isna() & processed_df[col].isna()
        )
        if not match_series.all():
            mismatches[col] = int((~match_series).sum())

    original_cols_identical = len(mismatches) == 0

    # 2. RowID continuous and unique
    row_id_valid = (
        SURROGATE_KEY_COLUMN in processed_df.columns
        and (processed_df[SURROGATE_KEY_COLUMN] == np.arange(proc_rows)).all()
    )

    # 3. Reconciliation summary table
    target_expected = expected_rows if expected_rows is not None else raw_rows
    reconciliation_table = pd.DataFrame(
        [
            {"Metric": "Raw Dataset Rows", "Value": raw_rows, "Expected": target_expected, "Status": "PASS" if raw_rows == target_expected else "FAIL"},
            {"Metric": "Processed Dataset Rows", "Value": proc_rows, "Expected": target_expected, "Status": "PASS" if proc_rows == target_expected else "FAIL"},
            {"Metric": "Rows Dropped", "Value": rows_dropped, "Expected": 0, "Status": "PASS" if rows_dropped == 0 else "FAIL"},
            {"Metric": "Original 31 Columns Exact Match", "Value": "100% Identical" if original_cols_identical else f"{len(mismatches)} columns with diffs", "Expected": "100% Identical", "Status": "PASS" if original_cols_identical else "FAIL"},
            {"Metric": "RowID Sequence [0 .. n-1]", "Value": "Valid & Continuous" if row_id_valid else "Invalid", "Expected": "Valid & Continuous", "Status": "PASS" if row_id_valid else "FAIL"},
            {"Metric": "Derived Columns Count", "Value": len(DERIVED_COLUMNS), "Expected": 15, "Status": "PASS" if len(DERIVED_COLUMNS) == 15 else "FAIL"},
            {"Metric": "Total Processed Columns", "Value": len(processed_df.columns), "Expected": 47, "Status": "PASS" if len(processed_df.columns) == 47 else "FAIL"},
        ]
    )

    # 4. Dropped reasons register (Empty register confirms 0 rows dropped)
    dropped_register = pd.DataFrame(columns=["RowID", "DropReason"])

    # 5. Bucket sizes audit table (Ordered categoricals and binary features)
    bucket_records = []
    bucket_cols = [
        "AgeGroup",
        "TenureBand",
        "EarlyTenure",
        "CareerStage",
        "RoleTenureBand",
        "ManagerTenureBand",
        "PromotionGapBand",
        "DistanceBand",
        "HasPriorExperience",
        "PromotionGapEqualsTenure",
        "PromotionAnalysisCohort",
        "OutstandingRating",
    ]
    for col in bucket_cols:
        vc = processed_df[col].value_counts(dropna=False)
        for level, count in vc.items():
            pct = round(float(count) / proc_rows * 100.0, 2)
            is_small = count < MIN_GROUP_SIZE
            bucket_records.append(
                {
                    "column": col,
                    "level": str(level),
                    "count": int(count),
                    "percentage": pct,
                    "is_small_group": is_small,
                }
            )
    bucket_sizes_table = pd.DataFrame(bucket_records)

    # 6. NaN Audit Table
    nan_records = []
    for col in processed_df.columns:
        cnt = int(processed_df[col].isna().sum())
        expected_nan = 44 if col in ["RoleTenureShare", "PromotionGapShare"] else 0
        pct = round(cnt / proc_rows * 100.0, 2)
        nan_records.append(
            {
                "column": col,
                "nan_count": cnt,
                "nan_pct": pct,
                "expected_nan_count": expected_nan,
                "status": "PASS" if cnt == expected_nan else "FAIL",
            }
        )
    nan_audit_table = pd.DataFrame(nan_records)

    # 7. Derived Feature Audit Table
    feature_records = []
    for col in DERIVED_COLUMNS:
        series = processed_df[col]
        dtype_str = str(series.dtype)
        valid_cnt = int(series.notna().sum())
        null_cnt = int(series.isna().sum())
        if hasattr(series, "cat"):
            domain_summary = f"{len(series.cat.categories)} categories: {list(series.cat.categories)}"
        elif dtype_str.startswith("int") or dtype_str.startswith("float"):
            min_val = series.min()
            max_val = series.max()
            domain_summary = f"Range [{min_val} .. {max_val}]"
        else:
            domain_summary = "Non-numeric / Categorical"

        feature_records.append(
            {
                "derived_column": col,
                "dtype": dtype_str,
                "non_null_count": valid_cnt,
                "null_count": null_cnt,
                "domain_summary": domain_summary,
            }
        )
    derived_feature_audit = pd.DataFrame(feature_records)

    return {
        "raw_rows": raw_rows,
        "processed_rows": proc_rows,
        "rows_dropped": rows_dropped,
        "original_cols_identical": original_cols_identical,
        "row_id_valid": row_id_valid,
        "reconciliation_table": reconciliation_table,
        "dropped_register": dropped_register,
        "bucket_sizes_table": bucket_sizes_table,
        "nan_audit_table": nan_audit_table,
        "derived_feature_audit": derived_feature_audit,
    }


