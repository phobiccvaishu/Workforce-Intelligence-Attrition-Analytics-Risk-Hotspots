"""Workforce Intelligence KPI Module.

Implements pure functions for evaluating 10 key performance indicators across
live exploratory filters and workforce-level static inferential benchmarks.
Adheres strictly to pre-registered contracts, statistical safeguards, and
multiplicity/governance constraints.
"""

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from workforce_intel.config import (
    ALLOWED_FILTER_DIMENSIONS,
    MIN_GROUP_SIZE,
    PROJECT_ROOT,
    STATIC_TABLE_CHECKSUMS,
)
from workforce_intel.stats_utils import wilson_interval

# Overall workforce baseline metrics (unfiltered constants)
WORKFORCE_TOTAL_N: int = 1470
WORKFORCE_TOTAL_LEAVERS: int = 237
WORKFORCE_BASELINE_RATE: float = 237 / 1470  # ~0.161224

COHORT_TOTAL_N: int = 890
COHORT_TOTAL_LEAVERS: int = 96
COHORT_BASELINE_RATE: float = 96 / 890  # ~0.107865

COHORT_DEFINITION_STR: str = (
    "PromotionAnalysisCohort: employees with TotalWorkingYears >= 6 and "
    "YearsAtCompany >= 6, excluding executive/senior management "
    "(JobLevel 4-5 or Director/Manager roles)."
)


def verify_static_source_integrity(filepath: Union[str, Path], expected_sha256: str) -> None:
    """Verify that a static source file matches its expected SHA-256 hash."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Static source file not found at: {path}")

    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_sha256:
        raise ValueError(
            f"Static source integrity failure for '{path.name}'. "
            f"Expected SHA-256 {expected_sha256}, but computed {actual_hash}."
        )


def apply_kpi_filter(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Filter workforce DataFrame by matching conjunction (AND) across key-value pairs.

    Validates that:
    1. Every filter key is in ALLOWED_FILTER_DIMENSIONS.
    2. Every filter value exists in the underlying dataset level domain.

    Returns the filtered DataFrame subset.
    """
    if filters is None or len(filters) == 0:
        return df.copy()

    filtered = df.copy()
    for col, target_val in filters.items():
        if col not in ALLOWED_FILTER_DIMENSIONS:
            raise ValueError(
                f"Dimension '{col}' is not an authorized filter dimension. "
                f"Permitted dimensions: {ALLOWED_FILTER_DIMENSIONS}"
            )
        if col not in df.columns:
            raise ValueError(f"Dimension '{col}' not found in DataFrame columns.")

        # Check domain existence
        existing_levels = set(df[col].dropna().astype(str).unique())
        target_str = str(target_val)
        if target_str not in existing_levels:
            raise ValueError(
                f"Level '{target_val}' does not exist in dimension '{col}'. "
                f"Available levels: {sorted(list(existing_levels))}"
            )

        filtered = filtered[filtered[col].astype(str) == target_str]

    return filtered


def _check_static_filters(filters: Optional[Dict[str, Any]], kpi_id: str) -> None:
    """Reject subpopulation filters on static benchmarks to avoid silent misuse."""
    if filters is not None and len(filters) > 0:
        raise ValueError(
            f"{kpi_id} is a workforce-level static benchmark and cannot accept "
            f"subpopulation filters. Passed filters: {filters}"
        )


# =============================================================================
# LIVE KPIs (1 through 5)
# =============================================================================


def kpi_01_attrition_rate(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> Dict[str, Any]:
    """KPI-01: Filtered Attrition Rate with Wilson 95% Score Interval."""
    filtered = apply_kpi_filter(df, filters)
    n = len(filtered)
    filter_dict = dict(filters) if filters else {}

    if n == 0:
        return {
            "kpi_id": "KPI-01",
            "name": "Attrition Rate",
            "value": None,
            "n": 0,
            "ci_low": None,
            "ci_high": None,
            "reliability": "no_data",
            "source": "src.workforce_intel.kpis.kpi_01_attrition_rate",
            "as_of_filter": filter_dict,
        }

    leavers = int(filtered["Attrition"].sum())
    rate = float(leavers / n)
    ci_l, ci_h = wilson_interval(leavers, n, confidence=0.95)
    reliability = "low_n" if n < min_group_size else "reliable"

    return {
        "kpi_id": "KPI-01",
        "name": "Attrition Rate",
        "value": rate,
        "n": n,
        "ci_low": ci_l,
        "ci_high": ci_h,
        "reliability": reliability,
        "source": "src.workforce_intel.kpis.kpi_01_attrition_rate",
        "as_of_filter": filter_dict,
    }


def kpi_02_headcount_leavers(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    total_headcount: int = WORKFORCE_TOTAL_N,
    total_leavers: int = WORKFORCE_TOTAL_LEAVERS,
    min_group_size: int = MIN_GROUP_SIZE,
) -> Dict[str, Any]:
    """KPI-02: Headcount and Leavers breakdown with workforce proportions."""
    filtered = apply_kpi_filter(df, filters)
    n = len(filtered)
    filter_dict = dict(filters) if filters else {}

    if n == 0:
        return {
            "kpi_id": "KPI-02",
            "name": "Headcount and Leavers Breakdown",
            "value": None,
            "n": 0,
            "ci_low": None,
            "ci_high": None,
            "reliability": "no_data",
            "source": "src.workforce_intel.kpis.kpi_02_headcount_leavers",
            "as_of_filter": filter_dict,
        }

    leavers = int(filtered["Attrition"].sum())
    headcount_share = float(n / total_headcount) if total_headcount > 0 else 0.0
    leavers_share = float(leavers / total_leavers) if total_leavers > 0 else 0.0
    reliability = "low_n" if n < min_group_size else "reliable"

    return {
        "kpi_id": "KPI-02",
        "name": "Headcount and Leavers Breakdown",
        "value": {
            "headcount": n,
            "leavers": leavers,
            "headcount_share": headcount_share,
            "leavers_share": leavers_share,
        },
        "n": n,
        "ci_low": None,
        "ci_high": None,
        "reliability": reliability,
        "source": "src.workforce_intel.kpis.kpi_02_headcount_leavers",
        "as_of_filter": filter_dict,
    }


def kpi_03_relative_attrition(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    baseline_rate: float = WORKFORCE_BASELINE_RATE,
    min_group_size: int = MIN_GROUP_SIZE,
) -> Dict[str, Any]:
    """KPI-03: Relative Attrition vs Baseline (percentage points, ratio, interval flag)."""
    filtered = apply_kpi_filter(df, filters)
    n = len(filtered)
    filter_dict = dict(filters) if filters else {}

    if n == 0:
        return {
            "kpi_id": "KPI-03",
            "name": "Relative Attrition vs Baseline",
            "value": None,
            "n": 0,
            "ci_low": None,
            "ci_high": None,
            "reliability": "no_data",
            "source": "src.workforce_intel.kpis.kpi_03_relative_attrition",
            "as_of_filter": filter_dict,
        }

    leavers = int(filtered["Attrition"].sum())
    rate = float(leavers / n)
    ci_l, ci_h = wilson_interval(leavers, n, confidence=0.95)

    rate_diff = float(rate - baseline_rate)
    rate_ratio = float(rate / baseline_rate) if baseline_rate > 0 else None

    # Interval separation rule:
    # lower bound > baseline -> above_baseline
    # upper bound < baseline -> below_baseline
    # otherwise -> no_separation
    if ci_l > baseline_rate:
        separation_flag = "above_baseline"
    elif ci_h < baseline_rate:
        separation_flag = "below_baseline"
    else:
        separation_flag = "no_separation"

    reliability = "low_n" if n < min_group_size else "reliable"

    return {
        "kpi_id": "KPI-03",
        "name": "Relative Attrition vs Baseline",
        "value": {
            "rate_diff": rate_diff,
            "rate_ratio": rate_ratio,
            "separation_flag": separation_flag,
            "baseline_rate": baseline_rate,
        },
        "n": n,
        "ci_low": ci_l,
        "ci_high": ci_h,
        "reliability": reliability,
        "source": "src.workforce_intel.kpis.kpi_03_relative_attrition",
        "as_of_filter": filter_dict,
    }


def kpi_04_overtime_association(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> Dict[str, Any]:
    """KPI-04: OverTime association within filter (crude OR, CI, OverTime share)."""
    filtered = apply_kpi_filter(df, filters)
    n = len(filtered)
    filter_dict = dict(filters) if filters else {}

    if n == 0:
        return {
            "kpi_id": "KPI-04",
            "name": "OverTime Association Within Filter",
            "value": None,
            "n": 0,
            "ci_low": None,
            "ci_high": None,
            "reliability": "no_data",
            "source": "src.workforce_intel.kpis.kpi_04_overtime_association",
            "as_of_filter": filter_dict,
        }

    n_ot = int((filtered["OverTime"] == "Yes").sum())
    n_no = int((filtered["OverTime"] == "No").sum())
    ot_share = float(n_ot / n)

    # 2x2 table cells:
    # a: Exposed Leavers (OT=Yes, Att=1)
    # b: Exposed Stayers (OT=Yes, Att=0)
    # c: Unexposed Leavers (OT=No, Att=1)
    # d: Unexposed Stayers (OT=No, Att=0)
    a = int(((filtered["OverTime"] == "Yes") & (filtered["Attrition"] == 1)).sum())
    b = int(((filtered["OverTime"] == "Yes") & (filtered["Attrition"] == 0)).sum())
    c = int(((filtered["OverTime"] == "No") & (filtered["Attrition"] == 1)).sum())
    d = int(((filtered["OverTime"] == "No") & (filtered["Attrition"] == 0)).sum())

    if n_ot == 0 or n_no == 0:
        return {
            "kpi_id": "KPI-04",
            "name": "OverTime Association Within Filter",
            "value": {
                "overtime_share": ot_share,
                "odds_ratio": None,
                "a_exposed_cases": a,
                "b_exposed_noncases": b,
                "c_unexposed_cases": c,
                "d_unexposed_noncases": d,
            },
            "n": n,
            "ci_low": None,
            "ci_high": None,
            "reliability": "exposure_fixed_by_filter",
            "source": "src.workforce_intel.kpis.kpi_04_overtime_association",
            "as_of_filter": filter_dict,
        }

    has_zero_cell = (a == 0 or b == 0 or c == 0 or d == 0)
    if has_zero_cell:
        return {
            "kpi_id": "KPI-04",
            "name": "OverTime Association Within Filter",
            "value": {
                "overtime_share": ot_share,
                "odds_ratio": None,
                "a_exposed_cases": a,
                "b_exposed_noncases": b,
                "c_unexposed_cases": c,
                "d_unexposed_noncases": d,
            },
            "n": n,
            "ci_low": None,
            "ci_high": None,
            "reliability": "unreliable_zero_cell",
            "source": "src.workforce_intel.kpis.kpi_04_overtime_association",
            "as_of_filter": filter_dict,
        }

    is_low_n_cell = min(a, b, c, d) < min_group_size or n < min_group_size
    or_val = float((a * d) / (b * c))
    se = float(math.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
    reliability = "low_n" if is_low_n_cell else "reliable"

    z = 1.959963984540054
    log_or = math.log(or_val)
    ci_l = float(math.exp(log_or - z * se))
    ci_h = float(math.exp(log_or + z * se))

    return {
        "kpi_id": "KPI-04",
        "name": "OverTime Association Within Filter",
        "value": {
            "overtime_share": ot_share,
            "odds_ratio": or_val,
            "a_exposed_cases": a,
            "b_exposed_noncases": b,
            "c_unexposed_cases": c,
            "d_unexposed_noncases": d,
        },
        "n": n,
        "ci_low": ci_l,
        "ci_high": ci_h,
        "reliability": reliability,
        "source": "src.workforce_intel.kpis.kpi_04_overtime_association",
        "as_of_filter": filter_dict,
    }


def kpi_05_promotion_gap_cohort(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> Dict[str, Any]:
    """KPI-05: Promotion-Gap Cohort Rate (6+ vs 0-5 yrs within PromotionAnalysisCohort)."""
    filtered = apply_kpi_filter(df, filters)
    filter_dict = dict(filters) if filters else {}

    # Restrict to PromotionAnalysisCohort
    cohort_df = filtered[filtered["PromotionAnalysisCohort"] == 1]
    n_cohort = len(cohort_df)

    if n_cohort == 0:
        return {
            "kpi_id": "KPI-05",
            "name": "Promotion-Gap Cohort Rate",
            "value": None,
            "n": 0,
            "ci_low": None,
            "ci_high": None,
            "reliability": "no_data",
            "source": "src.workforce_intel.kpis.kpi_05_promotion_gap_cohort",
            "as_of_filter": filter_dict,
            "cohort_definition": COHORT_DEFINITION_STR,
        }

    cohort_leavers = int(cohort_df["Attrition"].sum())
    cohort_rate = float(cohort_leavers / n_cohort)
    ci_l, ci_h = wilson_interval(cohort_leavers, n_cohort, confidence=0.95)

    # 2x2 comparison: 6+ yrs vs 0-5 yrs since last promotion
    # a: Exposed Leavers (YearsSinceLastPromotion >= 6 & Att=1)
    # b: Exposed Stayers (YearsSinceLastPromotion >= 6 & Att=0)
    # c: Unexposed Leavers (YearsSinceLastPromotion < 6 & Att=1)
    # d: Unexposed Stayers (YearsSinceLastPromotion < 6 & Att=0)
    a = int(((cohort_df["YearsSinceLastPromotion"] >= 6) & (cohort_df["Attrition"] == 1)).sum())
    b = int(((cohort_df["YearsSinceLastPromotion"] >= 6) & (cohort_df["Attrition"] == 0)).sum())
    c = int(((cohort_df["YearsSinceLastPromotion"] < 6) & (cohort_df["Attrition"] == 1)).sum())
    d = int(((cohort_df["YearsSinceLastPromotion"] < 6) & (cohort_df["Attrition"] == 0)).sum())

    n_exp = a + b
    n_unexp = c + d

    if n_exp == 0 or n_unexp == 0:
        or_val = None
        or_ci_l, or_ci_h = None, None
        reliability = "exposure_fixed_by_filter" if (n_cohort >= min_group_size) else "low_n"
    else:
        has_zero_cell = (a == 0 or b == 0 or c == 0 or d == 0)
        if has_zero_cell:
            or_val = None
            or_ci_l, or_ci_h = None, None
            reliability = "unreliable_zero_cell"
        else:
            is_low_n_cell = min(a, b, c, d) < min_group_size or n_cohort < min_group_size
            or_val = float((a * d) / (b * c))
            se = float(math.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
            reliability = "low_n" if is_low_n_cell else "reliable"
            z = 1.959963984540054
            log_or = math.log(or_val)
            or_ci_l = float(math.exp(log_or - z * se))
            or_ci_h = float(math.exp(log_or + z * se))

    return {
        "kpi_id": "KPI-05",
        "name": "Promotion-Gap Cohort Rate",
        "value": {
            "cohort_headcount": n_cohort,
            "cohort_leavers": cohort_leavers,
            "cohort_attrition_rate": cohort_rate,
            "crude_or_6plus_vs_0to5": or_val,
            "or_ci_low": or_ci_l,
            "or_ci_high": or_ci_h,
            "cohort_definition": COHORT_DEFINITION_STR,
        },
        "n": n_cohort,
        "ci_low": ci_l,
        "ci_high": ci_h,
        "reliability": reliability,
        "source": "src.workforce_intel.kpis.kpi_05_promotion_gap_cohort",
        "as_of_filter": filter_dict,
        "cohort_definition": COHORT_DEFINITION_STR,
    }


# =============================================================================
# STATIC KPIs (6 through 10)
# =============================================================================


def kpi_06_adjusted_overtime(
    tables_dir: Optional[Union[str, Path]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """KPI-06: Adjusted OverTime Association across adjustment frameworks."""
    _check_static_filters(filters, "KPI-06")
    rel_path = "reports/phase5/tables/overtime_comparison.csv"
    base_dir = Path(tables_dir) if tables_dir else PROJECT_ROOT
    table_path = base_dir / rel_path if not str(base_dir).endswith("tables") else Path(tables_dir) / "overtime_comparison.csv"

    verify_static_source_integrity(table_path, STATIC_TABLE_CHECKSUMS[rel_path])
    df_ot = pd.read_csv(table_path)

    # Extract estimates
    mh_row = df_ot[df_ot["method"] == "Stratified Mantel-Haenszel OR"].iloc[0]
    m1_row = df_ot[df_ot["method"] == "Adjusted Logistic Regression (M1)"].iloc[0]
    m2_row = df_ot[df_ot["method"] == "Adjusted Logistic Regression (M2)"].iloc[0]
    p4_row = df_ot[df_ot["method"] == "Phase 4 Regularised Contrast"].iloc[0]

    return {
        "kpi_id": "KPI-06",
        "name": "Adjusted OverTime Association",
        "value": {
            "mantel_haenszel_joblevel": {
                "odds_ratio": float(mh_row["odds_ratio"]),
                "ci_low": float(mh_row["ci_low"]),
                "ci_high": float(mh_row["ci_high"]),
                "adjustment": mh_row["adjustment_controls"],
            },
            "multivariable_m1": {
                "odds_ratio": float(m1_row["odds_ratio"]),
                "ci_low": float(m1_row["ci_low"]),
                "ci_high": float(m1_row["ci_high"]),
                "adjustment": m1_row["adjustment_controls"],
            },
            "multivariable_m2": {
                "odds_ratio": float(m2_row["odds_ratio"]),
                "ci_low": float(m2_row["ci_low"]),
                "ci_high": float(m2_row["ci_high"]),
                "adjustment": m2_row["adjustment_controls"],
            },
            "phase4_regularised_contrast": {
                "odds_ratio": float(p4_row["odds_ratio"]),
                "ci_low": float(p4_row["ci_low"]),
                "ci_high": float(p4_row["ci_high"]),
                "adjustment": p4_row["adjustment_controls"],
            },
        },
        "n": WORKFORCE_TOTAL_N,
        "ci_low": float(mh_row["ci_low"]),
        "ci_high": float(mh_row["ci_high"]),
        "reliability": "workforce-level, not filterable",
        "source": rel_path,
        "as_of_filter": {},
    }


def kpi_07_hotspot_universe_summary(
    tables_dir: Optional[Union[str, Path]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """KPI-07: Hotspot Universe Summary (universe size, sample floor, tiers, fitted prior)."""
    _check_static_filters(filters, "KPI-07")
    rel_path = "reports/phase5/tables/hotspot_universe.csv"
    base_dir = Path(tables_dir) if tables_dir else PROJECT_ROOT
    table_path = base_dir / rel_path if not str(base_dir).endswith("tables") else Path(tables_dir) / "hotspot_universe.csv"

    verify_static_source_integrity(table_path, STATIC_TABLE_CHECKSUMS[rel_path])
    df_univ = pd.read_csv(table_path)

    universe_size = len(df_univ)
    n_ge_30_count = int((df_univ["n"] >= MIN_GROUP_SIZE).sum())
    tier1_count = int(df_univ["is_tier1"].sum())
    tier2_count = int(df_univ["is_tier2"].sum())

    # Prior strength constants fitted in Phase 5 pipeline
    fitted_m_1way = 29.553471954595164
    fitted_m_2way = 8.653098550238532

    return {
        "kpi_id": "KPI-07",
        "name": "Hotspot Universe Summary",
        "value": {
            "universe_size": universe_size,
            "segments_n_ge_30": n_ge_30_count,
            "tier1_priority_hotspots": tier1_count,
            "tier2_secondary_hotspots": tier2_count,
            "fitted_prior_m_1way": fitted_m_1way,
            "fitted_prior_m_2way": fitted_m_2way,
        },
        "n": universe_size,
        "ci_low": None,
        "ci_high": None,
        "reliability": "workforce-level, not filterable",
        "source": rel_path,
        "as_of_filter": {},
    }


def kpi_08_hotspot_holdout_validation(
    tables_dir: Optional[Union[str, Path]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """KPI-08: Hotspot Holdout Validation (training screening and holdout replication)."""
    _check_static_filters(filters, "KPI-08")
    rel_path = "reports/phase5/tables/hotspot_split_validation.csv"
    base_dir = Path(tables_dir) if tables_dir else PROJECT_ROOT
    table_path = base_dir / rel_path if not str(base_dir).endswith("tables") else Path(tables_dir) / "hotspot_split_validation.csv"

    verify_static_source_integrity(table_path, STATIC_TABLE_CHECKSUMS[rel_path])
    df_val = pd.read_csv(table_path)

    screened_training_t1 = len(df_val)
    n30_sub = df_val[df_val["holdout_n_ge_30"]]
    holdout_n_ge_30 = len(n30_sub)
    holdout_above_point = int((n30_sub["holdout_rate"] > 0.1598639).sum())
    holdout_above_lower_bound = int(n30_sub["above_holdout_ci_low"].sum())

    return {
        "kpi_id": "KPI-08",
        "name": "Hotspot Holdout Validation",
        "value": {
            "screened_training_tier1": screened_training_t1,
            "holdout_sample_floor_met": holdout_n_ge_30,
            "holdout_above_baseline_point": holdout_above_point,
            "holdout_above_baseline_lower_bound": holdout_above_lower_bound,
            "sample_floor_proportion": f"{holdout_n_ge_30}/{screened_training_t1}",
            "point_replication_proportion": f"{holdout_above_point}/{holdout_n_ge_30}",
            "lower_bound_replication_proportion": f"{holdout_above_lower_bound}/{holdout_n_ge_30}",
            "governance_note": "full-data rows labeled optimistic: holdout was used to select",
        },
        "n": screened_training_t1,
        "ci_low": None,
        "ci_high": None,
        "reliability": "workforce-level, not filterable",
        "source": rel_path,
        "as_of_filter": {},
    }


def kpi_09_hotspot_coverage(
    tables_dir: Optional[Union[str, Path]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """KPI-09: Hotspot Greedy Coverage progression across non-redundant steps."""
    _check_static_filters(filters, "KPI-09")
    rel_path = "reports/phase5/tables/greedy_coverage.csv"
    base_dir = Path(tables_dir) if tables_dir else PROJECT_ROOT
    table_path = base_dir / rel_path if not str(base_dir).endswith("tables") else Path(tables_dir) / "greedy_coverage.csv"

    verify_static_source_integrity(table_path, STATIC_TABLE_CHECKSUMS[rel_path])
    df_cov = pd.read_csv(table_path)

    steps = []
    for _, r in df_cov.iterrows():
        steps.append({
            "step": int(r["step"]),
            "segment_id": r["segment_id"],
            "share_of_workforce": float(r["share_of_workforce"]),
            "share_of_leavers": float(r["share_of_leavers"]),
            "cum_employees": int(r["cum_employees"]),
            "cum_leavers": int(r["cum_leavers"]),
            "cum_rate": float(r["cum_rate"]),
            "marginal_rate": float(r["marginal_rate"]),
        })

    # Summary milestones at step 5 and final step
    step5 = steps[4] if len(steps) >= 5 else steps[-1]
    final_step = steps[-1]

    return {
        "kpi_id": "KPI-09",
        "name": "Hotspot Greedy Coverage",
        "value": {
            "total_steps": len(steps),
            "step_5_workforce_share": step5["share_of_workforce"],
            "step_5_leavers_share": step5["share_of_leavers"],
            "final_workforce_share": final_step["share_of_workforce"],
            "final_leavers_share": final_step["share_of_leavers"],
            "coverage_schedule": steps,
        },
        "n": WORKFORCE_TOTAL_N,
        "ci_low": None,
        "ci_high": None,
        "reliability": "workforce-level, not filterable",
        "source": rel_path,
        "as_of_filter": {},
    }


def kpi_10_model_triage_yield(
    tables_dir: Optional[Union[str, Path]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """KPI-10: Calibrated Model Triage Yield at top-10% and top-20% review thresholds."""
    _check_static_filters(filters, "KPI-10")
    rel_path = "reports/phase4/tables/holdout_metrics.csv"
    base_dir = Path(tables_dir) if tables_dir else PROJECT_ROOT
    table_path = base_dir / rel_path if not str(base_dir).endswith("tables") else Path(tables_dir) / "holdout_metrics.csv"

    verify_static_source_integrity(table_path, STATIC_TABLE_CHECKSUMS[rel_path])
    df_ho = pd.read_csv(table_path)

    # Calibrated Logistic Regression row
    cal_row = df_ho[df_ho["model"] == "Logistic Regression (Calibrated - Sigmoid)"].iloc[0]

    holdout_total_n = 294
    holdout_total_leavers = 47
    holdout_base_rate = 47 / 294  # ~0.159864

    top_10 = {
        "k": int(cal_row["k_10_point"]),
        "tp": int(cal_row["tp_10_point"]),
        "precision": float(cal_row["precision_at_10_point"]),
        "recall": float(cal_row["recall_at_10_point"]),
        "lift": float(cal_row["lift_at_10_point"]),
    }

    top_20 = {
        "k": int(cal_row["k_20_point"]),
        "tp": int(cal_row["tp_20_point"]),
        "precision": float(cal_row["precision_at_20_point"]),
        "recall": float(cal_row["recall_at_20_point"]),
        "lift": float(cal_row["lift_at_20_point"]),
    }

    return {
        "kpi_id": "KPI-10",
        "name": "Model Triage Yield",
        "value": {
            "holdout_base_rate": holdout_base_rate,
            "holdout_total_leavers": holdout_total_leavers,
            "holdout_total_headcount": holdout_total_n,
            "top_10_percent": top_10,
            "top_20_percent": top_20,
            "governance_label": "advisory decision support; no automated adverse actions",
        },
        "n": holdout_total_n,
        "ci_low": None,
        "ci_high": None,
        "reliability": "workforce-level, not filterable",
        "source": rel_path,
        "as_of_filter": {},
    }


def evaluate_all_kpis(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Any]] = None,
    tables_dir: Optional[Union[str, Path]] = None,
    min_group_size: int = MIN_GROUP_SIZE,
) -> List[Dict[str, Any]]:
    """Evaluate all 10 KPIs for a given dataset and optional exploratory filter.

    For LIVE KPIs (1-5), the subpopulation filter is applied.
    For STATIC KPIs (6-10), the static benchmark is evaluated and filters are rejected.
    """
    results = [
        kpi_01_attrition_rate(df, filters=filters, min_group_size=min_group_size),
        kpi_02_headcount_leavers(df, filters=filters, min_group_size=min_group_size),
        kpi_03_relative_attrition(df, filters=filters, min_group_size=min_group_size),
        kpi_04_overtime_association(df, filters=filters, min_group_size=min_group_size),
        kpi_05_promotion_gap_cohort(df, filters=filters, min_group_size=min_group_size),
        kpi_06_adjusted_overtime(tables_dir=tables_dir, filters=None),
        kpi_07_hotspot_universe_summary(tables_dir=tables_dir, filters=None),
        kpi_08_hotspot_holdout_validation(tables_dir=tables_dir, filters=None),
        kpi_09_hotspot_coverage(tables_dir=tables_dir, filters=None),
        kpi_10_model_triage_yield(tables_dir=tables_dir, filters=None),
    ]
    return results
