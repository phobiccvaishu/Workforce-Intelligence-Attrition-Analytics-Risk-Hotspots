"""Unit and invariant tests for Phase 3A Exploratory Data Analysis (EDA) engine."""

import math
import numpy as np
import pandas as pd
import pytest

from workforce_intel.config import (
    HOTSPOT_MIN_N,
    HOTSPOT_TOP_K,
    PHASE3_SUMMARY_JSON,
    PHASE3_TABLES_DIR,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
    TWO_WAY_SPECS,
)
from workforce_intel.eda import (
    exact_binomial_limits,
    funnel_table,
    hotspot_candidates,
    numeric_summary_by_outcome,
    overall_summary,
    rate_table,
    two_way_table,
    wilson_interval,
)
from workforce_intel.io import load_processed


# -----------------------------------------------------------------------------
# 1. Pure Unit Tests on Synthetic Frames
# -----------------------------------------------------------------------------

def test_rate_table_empty_frame():
    """Empty dataframe must return empty rate table with required schema."""
    df = pd.DataFrame(columns=["Dept", "Attrition"])
    rt = rate_table(df, "Dept")
    assert len(rt) == 0
    expected_cols = [
        "segment_dim",
        "segment",
        "n",
        "leavers",
        "retained",
        "attrition_rate_pct",
        "ci95_low_pct",
        "ci95_high_pct",
        "share_of_workforce_pct",
        "share_of_leavers_pct",
        "diff_vs_overall_pp",
        "ratio_vs_overall",
        "low_n_flag",
    ]
    assert list(rt.columns) == expected_cols


def test_rate_table_single_row():
    """Single row frame evaluates accurately."""
    df = pd.DataFrame({"Dept": ["Tech"], "Attrition": [1]})
    rt = rate_table(df, "Dept")
    assert len(rt) == 1
    row = rt.iloc[0]
    assert row["n"] == 1
    assert row["leavers"] == 1
    assert row["retained"] == 0
    assert row["attrition_rate_pct"] == 100.0
    assert bool(row["low_n_flag"]) is True


def test_rate_table_zero_and_hundred_percent_leavers():
    """Zero leavers and 100% leavers produce correct bounded intervals."""
    df_zero = pd.DataFrame({"Team": ["A", "A", "A"], "Attrition": [0, 0, 0]})
    rt_zero = rate_table(df_zero, "Team")
    assert rt_zero.iloc[0]["attrition_rate_pct"] == 0.0
    assert rt_zero.iloc[0]["ci95_low_pct"] == 0.0
    assert rt_zero.iloc[0]["ci95_high_pct"] > 0.0

    df_full = pd.DataFrame({"Team": ["B", "B"], "Attrition": [1, 1]})
    rt_full = rate_table(df_full, "Team")
    assert rt_full.iloc[0]["attrition_rate_pct"] == 100.0
    assert rt_full.iloc[0]["ci95_high_pct"] == 100.0
    assert rt_full.iloc[0]["ci95_low_pct"] < 100.0


def test_rate_table_unobserved_category_and_missing_values():
    """Unobserved categories appear as NaN rate with low_n_flag, and NaNs appear as 'Missing'."""
    cat_type = pd.CategoricalDtype(categories=["Observed", "Unobserved"], ordered=True)
    df = pd.DataFrame({
        "Status": pd.Series(["Observed", "Observed", None], dtype=cat_type),
        "Attrition": [0, 1, 0]
    })
    rt = rate_table(df, "Status")
    assert len(rt) == 3

    rt_dict = rt.set_index("segment").to_dict(orient="index")
    # Unobserved
    assert rt_dict["Unobserved"]["n"] == 0
    assert np.isnan(rt_dict["Unobserved"]["attrition_rate_pct"])
    assert rt_dict["Unobserved"]["low_n_flag"] is True

    # Missing
    assert rt_dict["Missing"]["n"] == 1
    assert rt_dict["Missing"]["attrition_rate_pct"] == 0.0
    assert rt_dict["Missing"]["low_n_flag"] is True


def test_rate_table_missing_column_raises():
    """Missing grouping column or target column must raise ValueError naming the column."""
    df = pd.DataFrame({"Dept": ["Tech"], "Attrition": [1]})
    with pytest.raises(ValueError, match="Grouping column 'NonExistent' not found"):
        rate_table(df, "NonExistent")

    with pytest.raises(ValueError, match="Target column 'Target_Absent' not found"):
        rate_table(df, "Dept", target_col="Target_Absent")


# -----------------------------------------------------------------------------
# 2. Pure Statistical Utilities Tests
# -----------------------------------------------------------------------------

def test_wilson_interval_properties():
    """Wilson interval obeys bounded mathematical properties."""
    low, high = wilson_interval(237, 1470, confidence=0.95)
    assert 0.14 <= low <= 0.15
    assert 0.17 <= high <= 0.19
    assert low < (237 / 1470) < high

    # Zero trials returns NaN
    nan_low, nan_high = wilson_interval(0, 0)
    assert math.isnan(nan_low) and math.isnan(nan_high)

    # Extreme bounds [0, 1]
    low_0, high_0 = wilson_interval(0, 10)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 1.0

    low_1, high_1 = wilson_interval(10, 10)
    assert high_1 == 1.0
    assert 0.0 < low_1 < 1.0


def test_exact_binomial_limits_properties():
    """Exact binomial funnel limits obey interval expansion across confidence tiers."""
    p0 = 0.1612
    n = 100
    low_95, up_95 = exact_binomial_limits(n, p0, level=0.95)
    low_998, up_998 = exact_binomial_limits(n, p0, level=0.998)

    assert 0.0 <= low_998 <= low_95 <= p0 <= up_95 <= up_998 <= 1.0


# -----------------------------------------------------------------------------
# 3. Dynamic Real-Data Invariants & Reconciliations
# -----------------------------------------------------------------------------

def test_real_data_one_way_reconciliations():
    """Every one-way rate table sums to total rows and total leavers."""
    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)
    total_n = len(df)
    total_leavers = int(df["Attrition"].sum())

    one_way_dims = [
        "Department",
        "JobRole",
        "JobLevel",
        "AgeGroup",
        "Gender",
        "MaritalStatus",
        "Education",
        "EducationField",
        "TenureBand",
        "EarlyTenure",
        "CareerStage",
        "RoleTenureBand",
        "ManagerTenureBand",
        "PromotionGapBand",
        "OverTime",
        "BusinessTravel",
        "DistanceBand",
        "WorkLifeBalance",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "RelationshipSatisfaction",
        "JobInvolvement",
        "StockOptionLevel",
        "TrainingTimesLastYear",
        "PerformanceRating",
    ]

    for dim in one_way_dims:
        rt = rate_table(df, dim)
        assert rt["n"].sum() == total_n, f"{dim} table row count mismatch"
        assert rt["leavers"].sum() == total_leavers, f"{dim} table leaver count mismatch"
        assert round(rt["share_of_workforce_pct"].sum(), 1) == 100.0, f"{dim} workforce share sum mismatch"
        assert round(rt["share_of_leavers_pct"].sum(), 1) == 100.0, f"{dim} leaver share sum mismatch"


def test_real_data_two_way_marginal_reconciliations():
    """Two-way cross-tabulations reconcile exactly to one-way marginals."""
    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)
    total_n = len(df)
    total_leavers = int(df["Attrition"].sum())

    for d1, d2 in TWO_WAY_SPECS:
        tw = two_way_table(df, d1, d2)
        assert tw["n"].sum() == total_n, f"Two-way {d1}x{d2} total n mismatch"
        assert tw["leavers"].sum() == total_leavers, f"Two-way {d1}x{d2} total leavers mismatch"

        # Marginal reconciliation for d1
        rt1 = rate_table(df, d1).set_index("segment")
        d1_grp = tw.groupby(d1)[["n", "leavers"]].sum()
        for cat in d1_grp.index:
            if str(cat) in rt1.index:
                assert d1_grp.loc[cat, "n"] == rt1.loc[str(cat), "n"]
                assert d1_grp.loc[cat, "leavers"] == rt1.loc[str(cat), "leavers"]


def test_real_data_hotspot_candidates_invariants():
    """Hotspot candidate scanning satisfies all qualification rules."""
    from workforce_intel.eda import hotspot_overlap_table

    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)
    ov = overall_summary(df)
    baseline_rate = float(ov["attrition_rate_pct"].iloc[0])

    two_ways = {f"{d1}_{d2}": two_way_table(df, d1, d2) for d1, d2 in TWO_WAY_SPECS}
    cands_df, total_scanned, cells_ge_30 = hotspot_candidates(
        two_ways, overall_rate_pct=baseline_rate, min_n=HOTSPOT_MIN_N, top_k=None
    )

    assert total_scanned == 58, f"Expected 58 scanned cells, got {total_scanned}"
    assert cells_ge_30 == 52, f"Expected 52 cells with n >= 30, got {cells_ge_30}"
    assert len(cands_df) == 16, f"Expected 16 eligible hotspot candidates, got {len(cands_df)}"

    for _, row in cands_df.iterrows():
        assert row["n"] >= HOTSPOT_MIN_N, f"Candidate {row['subgroup']} violates min n >= {HOTSPOT_MIN_N}"
        assert row["ci95_low_pct"] > baseline_rate, f"Candidate {row['subgroup']} Wilson lower bound <= baseline"
        assert row["diff_vs_overall_pp"] > 0.0

    # Overlap invariants
    overlap_df, union_n, union_leavers = hotspot_overlap_table(df, cands_df)
    assert len(overlap_df) == 120  # 16 * 15 / 2
    assert union_n == 599
    assert union_leavers == 171
    assert (overlap_df["note"] == "candidates overlap and are not independent").all()


def test_two_way_hand_computed_regression_cells():
    """Regression test verifying exact hand-computed cells across two-way tables."""
    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)

    def assert_cell(dim1, val1, dim2, val2, exp_n, exp_leavers):
        mask = (df[dim1].astype(str) == val1) & (df[dim2].astype(str) == val2)
        assert int(mask.sum()) == exp_n, f"{dim1}={val1} & {dim2}={val2} n mismatch"
        assert int(df.loc[mask, "Attrition"].sum()) == exp_leavers, f"{dim1}={val1} & {dim2}={val2} leavers mismatch"

    assert_cell("OverTime", "Yes", "AgeGroup", "26-35", 159, 62)
    assert_cell("OverTime", "Yes", "TenureBand", "2-3", 75, 29)
    assert_cell("OverTime", "Yes", "TenureBand", "4-5", 87, 22)
    assert_cell("OverTime", "Yes", "TenureBand", "6-10", 120, 31)
    assert_cell("OverTime", "Yes", "JobRole", "Sales Executive", 94, 31)
    assert_cell("OverTime", "Yes", "JobRole", "Research Scientist", 97, 33)
    assert_cell("OverTime", "No", "TenureBand", "0-1", 146, 37)
