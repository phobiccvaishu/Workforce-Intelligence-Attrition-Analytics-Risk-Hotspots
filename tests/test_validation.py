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

import json
from pathlib import Path
import re
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from workforce_intel.config import (
    AGE_LABELS,
    CAREER_STAGE_LABELS,
    DERIVED_COLUMNS,
    DISTANCE_LABELS,
    EXPECTED_COLUMNS,
    IQR_MULTIPLIER_OUTLIER,
    MANAGER_TENURE_LABELS,
    MIN_TENURE_FOR_PROMOTION_ANALYSIS,
    PROMOTION_GAP_LABELS,
    RAW_DATA_FILE,
    ROLE_TENURE_LABELS,
    SMALL_SAMPLE_THRESHOLD,
    TENURE_LABELS,
)
from workforce_intel.io import load_raw

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


def test_consistency_check_c7_multi_department_role():
    """Verify C7 detects JobRoles spanning more than one Department."""
    df = pd.DataFrame(
        {
            "Department": ["Sales", "Research & Development", "Sales"],
            "JobRole": ["Manager", "Manager", "Sales Executive"],
        }
    )
    results = consistency_checks(df)
    assert "Manager" in results["C7"]["multi_department_roles"]
    assert results["C7"]["multi_department_roles"]["Manager"] == 2
    assert "Sales Executive" not in results["C7"]["multi_department_roles"]
    assert results["C7"]["populated_cells"] == 3



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



def test_report_provenance_synthetic_dataset():
    """Verify that build_report formats purely from computed data and has no hardcoded values."""
    from workforce_intel.reporting import build_report

    # Create synthetic dataset with exactly 60 rows
    n = 60
    # 15 left (1), 45 stayed (0)
    attrition = [1] * 15 + [0] * 45
    # Two synthetic departments
    depts = ["Engineering"] * 40 + ["Marketing"] * 20
    roles = ["Software Engineer"] * 30 + ["Lead Engineer"] * 10 + ["Marketing Specialist"] * 20

    df_synth = pd.DataFrame(
        {
            "Age": [32] * n,
            "Attrition": attrition,
            "BusinessTravel": ["Non-Travel"] * n,
            "DailyRate": [700] * n,
            "Department": depts,
            "DistanceFromHome": [5] * n,
            "Education": [3] * n,
            "EducationField": ["Technical"] * n,
            "EnvironmentSatisfaction": [3] * n,
            "Gender": ["Female"] * n,
            "HourlyRate": [70] * n,
            "JobInvolvement": [3] * n,
            "JobLevel": [2] * n,
            "JobRole": roles,
            "JobSatisfaction": [3] * n,
            "MaritalStatus": ["Married"] * n,
            "MonthlyIncome": [5500] * n,
            "MonthlyRate": [12000] * n,
            "NumCompaniesWorked": [1] * n,
            "OverTime": ["No"] * n,
            "PercentSalaryHike": [14] * n,
            "PerformanceRating": [3] * n,
            "RelationshipSatisfaction": [3] * n,
            "StockOptionLevel": [1] * n,
            "TotalWorkingYears": [8] * n,
            "TrainingTimesLastYear": [2] * n,
            "WorkLifeBalance": [3] * n,
            "YearsAtCompany": [4] * n,
            "YearsInCurrentRole": [2] * n,
            "YearsSinceLastPromotion": [1] * n,
            "YearsWithCurrManager": [2] * n,
        }
    )

    # Compute all audits on synthetic data
    cols = list(df_synth.columns)
    struct = structure_report(df_synth, expected_columns=cols)
    missing_df = missing_report(df_synth)
    dup = duplicate_report(df_synth)
    cat_df = categorical_audit(df_synth)
    range_df = range_audit(df_synth)
    target = target_audit(df_synth)
    consistency = consistency_checks(df_synth)
    zero_df = zero_audit(df_synth)
    outlier_df = outlier_report(df_synth)
    collin = collinearity_report(df_synth)
    group_sz = group_size_report(df_synth)
    const_df = constant_column_check(df_synth)

    summary_payload = {
        "dataset_file": "synthetic_dataset.csv",
        "sha256": "abcdef1234567890",
        "execution_timestamp": "2026-11-01T00:00:00Z",
        "environment": {"pandas_version": "test", "numpy_version": "test", "scipy_version": "test"},
    }

    report = build_report(
        summary_payload=summary_payload,
        struct=struct,
        missing_df=missing_df,
        dup=dup,
        cat_df=cat_df,
        range_df=range_df,
        target=target,
        consistency=consistency,
        zero_df=zero_df,
        outlier_df=outlier_df,
        collin=collin,
        group_sz=group_sz,
        const_df=const_df,
        checkpoints_summary=None,
    )

    # Assert synthetic computed values appear
    assert "60" in report, "Synthetic total rows 60 must appear in report"
    assert "15" in report, "Synthetic class 1 count 15 must appear in report"
    assert "45" in report, "Synthetic class 0 count 45 must appear in report"
    assert "25.0%" in report, "Synthetic attrition rate 25.0% must appear in report"

    # Assert real-dataset statistics do NOT incorrectly appear in synthetic report
    forbidden_real_data_strings = [
        "1,470", "1470", "237", "1,233", "1233",
        "197", "581", "114", "16581", "16,581", "11 of 27"
    ]
    for forbidden in forbidden_real_data_strings:
        assert forbidden not in report, (
            f"Hardcoded real-data statistic '{forbidden}' was found in synthetic report!"
        )


def test_all_markdown_tables_column_consistency():
    """Parse every markdown table in docs/*.md and reports/phase1/*.md.
    
    Assert that every row in each table has the exact same number of cells as its header.
    Catches table breaking bugs such as unescaped pipes (|rho|).
    """
    from pathlib import Path

    md_files = list(Path("docs").glob("*.md")) + list(Path("reports/phase1").glob("*.md")) + list(Path("reports/phase2").glob("*.md")) + list(Path("reports/phase3").glob("*.md"))
    assert len(md_files) > 0, "No markdown files found to validate"

    errors = []
    for md_file in md_files:
        lines = md_file.read_text(encoding="utf-8").splitlines()
        in_table = False
        header_cols = 0

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
                cells = [c.strip() for c in stripped[1:-1].split("|")]
                if not in_table:
                    if line_num < len(lines):
                        next_line = lines[line_num].strip()
                        if (
                            next_line.startswith("|")
                            and next_line.endswith("|")
                            and set(next_line.replace("|", "").replace(":", "").replace("-", "").strip()) == set()
                        ):
                            in_table = True
                            header_cols = len(cells)
                            continue
                else:
                    if set(stripped.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                        if len(cells) != header_cols:
                            errors.append(f"{md_file}:{line_num} Delimiter cell count {len(cells)} != header {header_cols}")
                        continue
                    if len(cells) != header_cols:
                        errors.append(f"{md_file}:{line_num} Row cell count {len(cells)} != header {header_cols}: '{stripped}'")
            else:
                in_table = False
                header_cols = 0

    assert not errors, "Markdown table column inconsistencies found:\n" + "\n".join(errors)


def _build_expected_decision_log_tokens() -> Dict[str, List[str]]:
    """Dynamically build expected numeric token sequence for every DL row from computed artifacts."""
    with open("reports/phase1/validation_summary.json", encoding="utf-8") as f:
        s1 = json.load(f)

    raw_df, _ = load_raw(RAW_DATA_FILE)

    c6_table = pd.read_csv("reports/phase1/tables/consistency_c6_contradictions.csv")
    range_table = pd.read_csv("reports/phase1/tables/range_and_scale_audit.csv").set_index("column")
    rate_matrix = pd.read_csv("reports/phase1/tables/rate_coherence_spearman.csv", index_col=0)
    high_corr_table = pd.read_csv("reports/phase1/tables/high_correlation_pairs.csv")
    outlier_table = pd.read_csv("reports/phase1/tables/outlier_audit.csv").set_index("column")
    small_table = pd.read_csv("reports/phase1/tables/small_sample_nominal_levels.csv")
    zero_table = pd.read_csv("reports/phase1/tables/zero_value_audit.csv").set_index("column")

    reconcil_table = pd.read_csv("reports/phase2/tables/reconciliation.csv").set_index("Metric")
    bucket_table = pd.read_csv("reports/phase2/tables/bucket_sizes.csv").set_index(["column", "level"])
    feat_table = pd.read_csv("reports/phase2/tables/derived_feature_audit.csv").set_index("derived_column")
    nan_table = pd.read_csv("reports/phase2/tables/nan_audit.csv").set_index("column")

    t_metrics = s1["target_metrics"]
    total_rows = str(reconcil_table.loc["Processed Dataset Rows", "Value"])

    def get_spearman(col_a: str, col_b: str) -> float:
        row = high_corr_table[
            ((high_corr_table["feature_1"] == col_a) & (high_corr_table["feature_2"] == col_b)) |
            ((high_corr_table["feature_1"] == col_b) & (high_corr_table["feature_2"] == col_a))
        ]
        return float(row["spearman_rho"].iloc[0])

    rho_income_level = f"{get_spearman('MonthlyIncome', 'JobLevel'):.4f}"
    rho_role = f"{get_spearman('YearsAtCompany', 'YearsInCurrentRole'):.3f}"
    rho_mgr = f"{get_spearman('YearsAtCompany', 'YearsWithCurrManager'):.3f}"

    off_diag = rate_matrix.values[~np.eye(rate_matrix.shape[0], dtype=bool)]
    max_abs_rho = f"{np.max(np.abs(off_diag)):.4f}"

    sentinel_matches = raw_df[raw_df["DailyRate"].isin({-1, 999, 9999})]["DailyRate"]
    sentinel_idx = str(sentinel_matches.index[0])
    sentinel_val = str(sentinel_matches.iloc[0])

    prior_max = re.search(r"\.\.\s*(\d+)", feat_table.loc["PriorExperienceYears", "domain_summary"]).group(1)

    count_rating_3 = str(bucket_table.loc[("OutstandingRating", "0"), "count"])
    count_rating_4 = str(bucket_table.loc[("OutstandingRating", "1"), "count"])

    return {
        "DL-01": [str(x) for x in t_metrics["distinct_values"]] + ["0", str(t_metrics["count_attrition_0"]), "1", str(t_metrics["count_attrition_1"])],
        "DL-02": [str(len(c6_table)), "0"],
        "DL-03": [str(x) for x in s1["checkpoints_validation"]["performance_rating_levels"]["computed"]] + ["3", count_rating_3, "4", count_rating_4, str(int(range_table.loc["PerformanceRating", "documented_min"])), str(int(range_table.loc["PerformanceRating", "documented_max"]))],
        "DL-04": [max_abs_rho],
        "DL-05": [rho_income_level],
        "DL-06": [rho_role, rho_mgr, str(s1["checkpoints_validation"]["years_at_company_zero"]["computed"]), "0"],
        "DL-07": [str(int(outlier_table.loc["MonthlyIncome", "outliers_count_1_5x"])), str(IQR_MULTIPLIER_OUTLIER), str(int(outlier_table.loc["MonthlyIncome", "upper_bound_1_5x"]))],
        "DL-08": [str(small_table[small_table["level"] == "Human Resources"]["count"].iloc[0]), str(SMALL_SAMPLE_THRESHOLD)],
        "DL-09": ["0", sentinel_idx, sentinel_val],
        "DL-10": [total_rows, "0", str(int(total_rows) - 1), "1"],
        "DL-11": [str(len(EXPECTED_COLUMNS)), "0", "0", total_rows],
        "DL-12": ["17", "25", "25", "35", "35", "45", "45", "60"] + [str(bucket_table.loc[("AgeGroup", l), "count"]) for l in AGE_LABELS] + [total_rows],
        "DL-13": ["1", "1", "1", "3", "3", "5", "5", "10", "10", "100"] + [str(bucket_table.loc[("TenureBand", l), "count"]) for l in TENURE_LABELS],
        "DL-14": ["1", "2", str(bucket_table.loc[("EarlyTenure", "1"), "count"]), f"{bucket_table.loc[('EarlyTenure', '1'), 'percentage']:.2f}"],
        "DL-15": [str(bucket_table.loc[("CareerStage", l), "count"]) for l in CAREER_STAGE_LABELS],
        "DL-16": [str(bucket_table.loc[("RoleTenureBand", l), "count"]) for l in ROLE_TENURE_LABELS],
        "DL-17": [str(bucket_table.loc[("ManagerTenureBand", l), "count"]) for l in MANAGER_TENURE_LABELS],
        "DL-18": [str(bucket_table.loc[("PromotionGapBand", l), "count"]) for l in PROMOTION_GAP_LABELS],
        "DL-19": [str(bucket_table.loc[("DistanceBand", l), "count"]) for l in DISTANCE_LABELS],
        "DL-20": ["0", prior_max, str(bucket_table.loc[("HasPriorExperience", "0"), "count"])],
        "DL-21": ["0", str(bucket_table.loc[("HasPriorExperience", "1"), "count"]), f"{bucket_table.loc[('HasPriorExperience', '1'), 'percentage']:.2f}"],
        "DL-22": [str(nan_table.loc["RoleTenureShare", "nan_count"]), "0", str(nan_table.loc["RoleTenureShare", "nan_count"])],
        "DL-23": ["0", str(bucket_table.loc[("PromotionGapEqualsTenure", "1"), "count"]), f"{bucket_table.loc[('PromotionGapEqualsTenure', '1'), 'percentage']:.2f}"],
        "DL-24": [str(MIN_TENURE_FOR_PROMOTION_ANALYSIS), str(bucket_table.loc[("PromotionAnalysisCohort", "1"), "count"]), f"{bucket_table.loc[('PromotionAnalysisCohort', '1'), 'percentage']:.2f}"],
        "DL-25": [str(max(s1["checkpoints_validation"]["performance_rating_levels"]["computed"])), str(bucket_table.loc[("OutstandingRating", "1"), "count"]), f"{bucket_table.loc[('OutstandingRating', '1'), 'percentage']:.2f}"],
        "DL-26": ["0", str(zero_table.loc["YearsInCurrentRole", "zero_count"])],
        "DL-27": [total_rows, "9"],
        "DL-28": [f"{t_metrics['rate_attrition_1_pct']:.2f}", "95", f"{t_metrics['wilson_95_ci_lower_pct']:.2f}", f"{t_metrics['wilson_95_ci_upper_pct']:.2f}"],
        "DL-29": ["12", f"{t_metrics['rate_attrition_1_pct']:.2f}", "3", "9"],
        "DL-30": ["15", "58", "30"],
        "DL-31": [str(bucket_table.loc[("PromotionAnalysisCohort", "1"), "count"]), total_rows, "4"],
    }


def _check_decision_log_tokens(content: str) -> None:
    """Parse numeric tokens from decision log and assert exact equality with generated evidence."""
    actual_tokens = {}
    for line in content.splitlines():
        if "DL-" in line:
            parts = [p.strip() for p in line[1:-1].split("|")]
            dl_id = parts[0].strip("`")
            evidence_cell = parts[2]
            nums = re.findall(r"\d+(?:,\d+)*(?:\.\d+)?", evidence_cell.split("Table:")[0])
            actual_tokens[dl_id] = [n.replace(",", "") for n in nums]

    expected_tokens = _build_expected_decision_log_tokens()

    assert len(actual_tokens) == len(expected_tokens), (
        f"Row count mismatch: {len(actual_tokens)} rows in log vs {len(expected_tokens)} expected"
    )

    for dl_id, exp in expected_tokens.items():
        assert dl_id in actual_tokens, f"Decision row {dl_id} missing from decision log"
        act = actual_tokens[dl_id]
        assert act == exp, f"Evidence tokens mismatch for {dl_id}: actual={act} vs expected={exp}"


def test_documentation_evidence_matches_validation_summary():
    """Verify that every count, metric, status, and evidence in docs/ traces to validation outputs."""
    summary_file = Path("reports/phase1/validation_summary.json")
    if not summary_file.exists():
        pytest.skip("validation_summary.json not yet generated")

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Verify target metrics internal consistency (exact literals verified in test_reference_checkpoints.py)
    t_metrics = summary["target_metrics"]
    assert t_metrics["total_records"] == t_metrics["count_attrition_1"] + t_metrics["count_attrition_0"]
    assert round(t_metrics["rate_attrition_1_pct"] + t_metrics["rate_attrition_0_pct"], 1) == 100.0

    # Verify decision log evidence and statuses
    decision_log_path = Path("docs/decision_log.md")
    assert decision_log_path.exists(), "docs/decision_log.md must exist"
    content = decision_log_path.read_text(encoding="utf-8")

    # Status verifications for ALL DL rows (DL-01 to DL-26)
    expected_statuses = {
        "DL-01": "APPROVED",
        "DL-02": "APPROVED",
        "DL-03": "APPROVED",
        "DL-04": "APPROVED",
        "DL-05": "PROPOSED",
        "DL-06": "PROPOSED",
        "DL-07": "APPROVED",
        "DL-08": "APPROVED",
        "DL-09": "APPROVED",
        "DL-10": "APPROVED",
        "DL-11": "APPROVED",
        "DL-12": "APPROVED",
        "DL-13": "APPROVED",
        "DL-14": "APPROVED",
        "DL-15": "APPROVED",
        "DL-16": "APPROVED",
        "DL-17": "APPROVED",
        "DL-18": "APPROVED",
        "DL-19": "APPROVED",
        "DL-20": "APPROVED",
        "DL-21": "APPROVED",
        "DL-22": "APPROVED",
        "DL-23": "APPROVED",
        "DL-24": "APPROVED",
        "DL-25": "APPROVED",
        "DL-26": "DEFERRED",
        "DL-27": "PROPOSED",
        "DL-28": "PROPOSED",
        "DL-29": "PROPOSED",
        "DL-30": "PROPOSED",
        "DL-31": "PROPOSED",
    }
    for dl_id, expected_status in expected_statuses.items():
        pattern = rf"`{dl_id}`.*?`{expected_status}`"
        assert re.search(pattern, content), f"Status for {dl_id} must be `{expected_status}`"

    assert "$" not in content, "No dollar signs permitted in decision log"
    assert "|rho|" not in content, "No unescaped '|rho|' allowed in decision log"

    # Exact token equality verification across EVERY DL row (DL-01 to DL-26)
    _check_decision_log_tokens(content)

    # Verify data dictionary observed-data statements
    data_dict_path = Path("docs/data_dictionary.md")
    assert data_dict_path.exists()
    dict_content = data_dict_path.read_text(encoding="utf-8")
    assert "$" not in dict_content, "No dollar signs permitted in data dictionary"

    range_df = pd.read_csv("reports/phase1/tables/range_and_scale_audit.csv").set_index("column")
    for line in dict_content.splitlines():
        if line.startswith("| `") and ("[" in line):
            parts = [p.strip() for p in line[1:-1].split("|")]
            col = parts[0].strip("`")
            obs_range_str = parts[3]
            if col in range_df.index:
                obs_min = range_df.loc[col, "observed_min"]
                obs_max = range_df.loc[col, "observed_max"]
                m = re.search(r"\[([\d,]+)\s*\.\.\s*([\d,]+)\]", obs_range_str)
                if m:
                    val_min = int(m.group(1).replace(",", ""))
                    val_max = int(m.group(2).replace(",", ""))
                    assert val_min == int(obs_min), f"{col} min mismatch in data dictionary: {val_min} != {obs_min}"
                    assert val_max == int(obs_max), f"{col} max mismatch in data dictionary: {val_max} != {obs_max}"


def test_decision_log_evidence_mutation_raises():
    """Mutation test: changing any evidence number in the decision log must raise an AssertionError."""
    decision_log_path = Path("docs/decision_log.md")
    content = decision_log_path.read_text(encoding="utf-8")

    # Mutation 1: Change count 237 to 238 in DL-01
    assert "count 1: 237" in content
    mutated_content_1 = content.replace("count 1: 237", "count 1: 238", 1)
    with pytest.raises(AssertionError, match="Evidence tokens mismatch for DL-01"):
        _check_decision_log_tokens(mutated_content_1)

    # Mutation 2: Change promotion gap count in DL-23
    assert "matches 135 employees" in content
    mutated_content_2 = content.replace("matches 135 employees", "matches 136 employees", 1)
    with pytest.raises(AssertionError, match="Evidence tokens mismatch for DL-23"):
        _check_decision_log_tokens(mutated_content_2)
