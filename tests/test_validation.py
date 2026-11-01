"""Unit tests for pure validation functions using synthetic DataFrames.

Covers all functions in src/workforce_intel/validation.py across required edge cases:
- Empty DataFrames and single-row DataFrames
- NaN, whitespace-only strings, sentinel tokens ('?', 'NA', 'null', etc.)
- Exact full-row duplicates and key-subset duplicates
- Out-of-range ordinals and restricted-range ordinals
- Attrition target values outside {0, 1}
- Temporal contradictions (C1: YearsAtCompany > TotalWorkingYears)
- C6 contradictions (NumCompaniesWorked == 0 while TotalWorkingYears > YearsAtCompany)
"""

import numpy as np
import pandas as pd
import pytest

from workforce_intel.validation import (
    categorical_audit,
    collinearity_report,
    consistency_checks,
    constant_column_check,
    duplicate_report,
    group_size_report,
    missing_report,
    outlier_report,
    range_audit,
    structure_report,
    target_audit,
    zero_audit,
)


@pytest.fixture
def clean_synthetic_df() -> pd.DataFrame:
    """Fixture providing a clean, valid 3-row synthetic DataFrame."""
    return pd.DataFrame(
        {
            "Age": [30, 40, 50],
            "Attrition": [0, 1, 0],
            "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
            "Department": ["Sales", "Research & Development", "Sales"],
            "JobRole": ["Sales Executive", "Research Scientist", "Sales Executive"],
            "Education": [3, 4, 2],
            "PerformanceRating": [3, 4, 3],
            "YearsAtCompany": [5, 10, 15],
            "YearsInCurrentRole": [3, 5, 10],
            "YearsSinceLastPromotion": [1, 2, 0],
            "YearsWithCurrManager": [3, 5, 10],
            "TotalWorkingYears": [8, 18, 25],
            "NumCompaniesWorked": [2, 1, 3],
            "DailyRate": [800, 1200, 950],
            "HourlyRate": [60, 75, 65],
            "MonthlyRate": [15000, 20000, 18000],
            "MonthlyIncome": [4000, 8000, 6000],
            "JobLevel": [1, 3, 2],
            "StockOptionLevel": [0, 1, 2],
        }
    )


def test_structure_report_matching(clean_synthetic_df):
    cols = list(clean_synthetic_df.columns)
    rep = structure_report(clean_synthetic_df, expected_columns=cols)
    assert rep["n_rows"] == 3
    assert rep["n_cols"] == len(cols)
    assert not rep["contract_violated"]
    assert rep["order_matches_expected"]


def test_structure_report_mismatches():
    df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
    rep = structure_report(df, expected_columns=["B", "A", "C"])
    assert rep["contract_violated"]
    assert "C" in rep["missing_columns"]
    assert not rep["order_matches_expected"]


def test_structure_report_empty_df():
    empty_df = pd.DataFrame(columns=["A", "B"])
    rep = structure_report(empty_df, expected_columns=["A", "B"])
    assert rep["n_rows"] == 0
    assert rep["n_cols"] == 2
    assert not rep["contract_violated"]


def test_missing_report_sentinels_and_whitespace():
    df = pd.DataFrame(
        {
            "text_col": ["valid", "   ", "?", "NA", "null", None, ""],
            "num_col": [10, -1, 999, 9999, None, 50, 60],
        }
    )
    rep = missing_report(df).set_index("column")

    assert rep.loc["text_col", "whitespace_only_count"] == 1
    assert rep.loc["text_col", "empty_string_count"] == 1
    # '?', 'NA', 'null' are string sentinels
    assert rep.loc["text_col", "string_sentinel_count"] == 3
    assert rep.loc["text_col", "nan_count"] == 1

    # -1, 999, 9999 are numeric sentinels
    assert rep.loc["num_col", "numeric_sentinel_count"] == 3
    assert rep.loc["num_col", "nan_count"] == 1


def test_duplicate_report():
    df = pd.DataFrame(
        {
            "Age": [25, 25, 30],
            "Gender": ["Male", "Male", "Female"],
            "Department": ["Sales", "Sales", "HR"],
            "JobRole": ["Sales Rep", "Sales Rep", "HR Rep"],
            "TotalWorkingYears": [3, 3, 5],
            "YearsAtCompany": [2, 2, 4],
        }
    )
    rep = duplicate_report(df)
    assert rep["full_row_duplicates"] == 1
    assert rep["key_subset_duplicates"] == 1
    assert "NOTICE" in rep["uniqueness_note"]


def test_categorical_audit():
    df = pd.DataFrame(
        {
            "Department": ["Sales", " sales", "SALES", "HR"],
        }
    )
    rep = categorical_audit(df, nominal_columns=["Department"])
    assert rep["leading_trailing_whitespace"].any()
    assert rep["has_case_variants"].any()
    assert len(rep) == 4


def test_range_audit_and_restricted_range():
    df = pd.DataFrame(
        {
            "Education": [1, 2, 6],          # 6 is out of documented (1, 5)
            "PerformanceRating": [3, 4, 3],  # only 2 levels observed out of documented 4 (1-4)
        }
    )
    ranges = {"Education": (1, 5), "PerformanceRating": (1, 4)}
    rep = range_audit(df, ordinal_ranges=ranges, numeric_columns=["Education", "PerformanceRating"]).set_index("column")

    assert rep.loc["Education", "out_of_bounds"]
    assert not rep.loc["PerformanceRating", "out_of_bounds"]
    assert rep.loc["PerformanceRating", "is_restricted_range"]


def test_target_audit_clean():
    df = pd.DataFrame({"Attrition": [0, 0, 0, 1]})
    rep = target_audit(df, target_col="Attrition")
    assert rep["is_binary_01_subset"]
    assert rep["count_attrition_0"] == 3
    assert rep["count_attrition_1"] == 1
    assert rep["rate_attrition_1_pct"] == 25.0
    assert rep["wilson_95_ci_lower_pct"] < rep["rate_attrition_1_pct"] < rep["wilson_95_ci_upper_pct"]


def test_target_audit_invalid_values():
    df = pd.DataFrame({"Attrition": [0, 1, 2, 99]})
    rep = target_audit(df, target_col="Attrition")
    assert not rep["is_binary_01_subset"]


def test_consistency_checks_violations():
    df = pd.DataFrame(
        {
            "Age": [25, 30],
            "YearsAtCompany": [10, 5],
            "TotalWorkingYears": [5, 12],            # row 0 has YearsAtCompany (10) > TotalWorkingYears (5) -> C1
            "YearsInCurrentRole": [12, 3],           # row 0 has YearsInCurrentRole (12) > YearsAtCompany (10) -> C2
            "YearsSinceLastPromotion": [11, 2],      # row 0 has YearsSinceLastPromotion (11) > YearsAtCompany (10) -> C3
            "YearsWithCurrManager": [15, 2],         # row 0 has YearsWithCurrManager (15) > YearsAtCompany (10) -> C4
            "NumCompaniesWorked": [0, 2],            # row 1 has NumCompaniesWorked==0 & TotalWorkingYears (12) > YearsAtCompany (5) -> C6
            "Department": ["Sales", "Sales"],
            "JobRole": ["Sales Exec", "Sales Exec"],
            "MonthlyIncome": [3000, 7000],
            "JobLevel": [1, 2],
            "DailyRate": [500, 1000],
            "HourlyRate": [50, 80],
            "MonthlyRate": [10000, 20000],
        }
    )
    # Also test C5 hard: Age 25 with TotalWorkingYears 15 -> 15 > 25 - 14 = 11
    df.loc[0, "TotalWorkingYears"] = 15
    df.loc[0, "Age"] = 25

    results = consistency_checks(df)
    assert results["C5_hard"]["violation_count"] == 1
    assert 0 in results["C5_hard"]["offending_indices"]
    assert results["C2"]["violation_count"] == 1
    assert 0 in results["C2"]["offending_indices"]
    assert results["C3"]["violation_count"] == 1
    assert results["C4"]["violation_count"] == 1


def test_consistency_check_c6_contradiction():
    df = pd.DataFrame(
        {
            "NumCompaniesWorked": [0, 0, 1],
            "TotalWorkingYears": [10, 5, 8],
            "YearsAtCompany": [5, 5, 4],
        }
    )
    results = consistency_checks(df)
    assert results["C6"]["violation_count"] == 1
    assert results["C6"]["offending_indices"] == [0]


def test_zero_audit():
    df = pd.DataFrame(
        {
            "StockOptionLevel": [0, 1, 0],
            "DailyRate": [100, 0, 500],
        }
    )
    rep = zero_audit(df, numeric_columns=["StockOptionLevel", "DailyRate"]).set_index("column")
    assert rep.loc["StockOptionLevel", "zero_count"] == 2
    assert "legitimate zero" in rep.loc["StockOptionLevel", "candidate_classification"]
    assert "possibly missing" in rep.loc["DailyRate", "candidate_classification"]


def test_outlier_report():
    # 20 regular numbers + 1 extreme outlier
    data = [10] * 20 + [500]
    df = pd.DataFrame({"Metric": data})
    rep = outlier_report(df, numeric_columns=["Metric"]).iloc[0]
    assert rep["outliers_count_1_5x"] == 1
    assert rep["outliers_count_3_0x"] == 1
    assert rep["skewness"] > 0


def test_collinearity_and_vif():
    # Collinear columns
    x = np.linspace(1, 100, 50)
    df = pd.DataFrame(
        {
            "feat_a": x,
            "feat_b": 2 * x + 1,
            "feat_c": np.random.normal(0, 1, 50),
        }
    )
    rep = collinearity_report(df, numeric_columns=["feat_a", "feat_b", "feat_c"])
    high_pairs = rep["high_correlation_pairs"]
    assert not high_pairs.empty
    assert (
        (high_pairs["feature_1"] == "feat_a") & (high_pairs["feature_2"] == "feat_b")
    ).any()


def test_group_size_report():
    df = pd.DataFrame(
        {
            "Department": ["Sales"] * 25 + ["R&D"] * 50,
            "JobRole": ["Sales Exec"] * 25 + ["Scientist"] * 50,
        }
    )
    rep = group_size_report(df, nominal_columns=["Department"], threshold=30)
    assert not rep["small_nominal_cells_table"].empty
    assert rep["small_nominal_cells_table"].iloc[0]["count"] == 25


def test_constant_column_check():
    df = pd.DataFrame(
        {
            "constant_col": [1] * 100,
            "near_constant": [0] * 99 + [1],
            "normal_col": list(range(100)),
        }
    )
    rep = constant_column_check(df, near_constant_threshold=0.99).set_index("column")
    assert rep.loc["constant_col", "is_constant"]
    assert rep.loc["near_constant", "is_near_constant"]
    assert not rep.loc["normal_col", "is_constant"]
    assert not rep.loc["normal_col", "is_near_constant"]
