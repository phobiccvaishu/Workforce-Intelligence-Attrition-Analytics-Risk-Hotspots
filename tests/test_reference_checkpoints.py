# =============================================================================
# OWNER-SUPPLIED INDEPENDENT PRE-CHECKS (Real-data literals permitted here only)
# =============================================================================
"""Reference checkpoints supplied by the project owner for independent validation.

Per project testing policy, real-data numerical literals (e.g. counts, rates,
benchmark figures) are strictly segregated to this file to prevent test overfitting
and ensure all operational tests remain dynamic and decoupled.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from workforce_intel.config import (
    PHASE1_SUMMARY_JSON,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
)
from workforce_intel.io import load_processed


def test_target_metrics_pre_checks():
    """Verify target metrics against owner-supplied benchmark numbers."""
    assert PHASE1_SUMMARY_JSON.exists()
    with open(PHASE1_SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary = json.load(f)

    t_metrics = summary["target_metrics"]
    assert t_metrics["total_records"] == 1470
    assert t_metrics["count_attrition_1"] == 237
    assert t_metrics["count_attrition_0"] == 1233
    assert t_metrics["rate_attrition_1_pct"] == 16.12
    assert t_metrics["rate_attrition_0_pct"] == 83.88


def test_phase2_section_8_checkpoints():
    """Verify Phase 2 Section 8 contract checkpoints."""
    processed_df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)

    # 1. Total shape
    assert len(processed_df) == 1470
    assert len(processed_df.columns) == 47

    # 2. Bands
    assert processed_df["AgeGroup"].value_counts()[["18-25", "26-35", "36-45", "46-60"]].tolist() == [123, 606, 468, 273]
    assert processed_df["TenureBand"].value_counts()[["0-1", "2-3", "4-5", "6-10", "11+"]].tolist() == [215, 255, 306, 448, 246]
    assert processed_df["CareerStage"].value_counts()[["0-5", "6-10", "11-20", "21+"]].tolist() == [316, 607, 340, 207]
    assert processed_df["RoleTenureBand"].value_counts()[["0-1", "2-3", "4-7", "8+"]].tolist() == [301, 507, 399, 263]
    assert processed_df["ManagerTenureBand"].value_counts()[["0-1", "2-3", "4-7", "8+"]].tolist() == [339, 486, 374, 271]
    assert processed_df["PromotionGapBand"].value_counts()[["0", "1-2", "3-5", "6+"]].tolist() == [581, 516, 158, 215]
    assert processed_df["DistanceBand"].value_counts()[["1-5", "6-10", "11-20", "21-29"]].tolist() == [632, 394, 240, 204]

    # 3. Indicators & Derived metrics
    assert int((processed_df["EarlyTenure"] == 1).sum()) == 215
    assert processed_df["PriorExperienceYears"].min() == 0
    assert processed_df["PriorExperienceYears"].max() == 33
    assert int((processed_df["PriorExperienceYears"] == 0).sum()) == 474
    assert int((processed_df["HasPriorExperience"] == 1).sum()) == 996
    assert int(processed_df["RoleTenureShare"].isna().sum()) == 44
    assert int(processed_df["PromotionGapShare"].isna().sum()) == 44
    assert int((processed_df["PromotionGapEqualsTenure"] == 1).sum()) == 135
    assert int((processed_df["PromotionAnalysisCohort"] == 1).sum()) == 890
    assert int((processed_df["OutstandingRating"] == 1).sum()) == 226


def test_phase3a_section_7_checkpoints():
    """Verify Phase 3A Section 7 owner-supplied EDA reference checkpoints."""
    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)
    y = df["Attrition"]

    # 1. Overall
    assert len(df) == 1470
    assert int(y.sum()) == 237
    assert round(float(y.mean()) * 100, 1) == 16.1

    # 2. Department
    dept_grp = df.groupby("Department")["Attrition"]
    assert dept_grp.get_group("Research & Development").agg(["sum", "count"]).tolist() == [133, 961]
    assert round(dept_grp.get_group("Research & Development").mean() * 100, 1) == 13.8
    assert dept_grp.get_group("Sales").agg(["sum", "count"]).tolist() == [92, 446]
    assert round(dept_grp.get_group("Sales").mean() * 100, 1) == 20.6
    assert dept_grp.get_group("Human Resources").agg(["sum", "count"]).tolist() == [12, 63]
    assert round(dept_grp.get_group("Human Resources").mean() * 100, 1) == 19.0

    # 3. JobRole
    role_grp = df.groupby("JobRole")["Attrition"]
    roles_expected = {
        "Sales Representative": (33, 83, 39.8),
        "Laboratory Technician": (62, 259, 23.9),
        "Human Resources": (12, 52, 23.1),
        "Sales Executive": (57, 326, 17.5),
        "Research Scientist": (47, 292, 16.1),
        "Healthcare Representative": (9, 131, 6.9),
        "Manufacturing Director": (10, 145, 6.9),
        "Manager": (5, 102, 4.9),
        "Research Director": (2, 80, 2.5),
    }
    for role, (exp_leavers, exp_n, exp_rate) in roles_expected.items():
        assert role_grp.get_group(role).agg(["sum", "count"]).tolist() == [exp_leavers, exp_n]
        assert round(role_grp.get_group(role).mean() * 100, 1) == exp_rate

    # 4. AgeGroup
    age_grp = df.groupby("AgeGroup")["Attrition"]
    ages_expected = {
        "18-25": (44, 123, 35.8),
        "26-35": (116, 606, 19.1),
        "36-45": (43, 468, 9.2),
        "46-60": (34, 273, 12.5),
    }
    for band, (exp_leavers, exp_n, exp_rate) in ages_expected.items():
        assert age_grp.get_group(band).agg(["sum", "count"]).tolist() == [exp_leavers, exp_n]
        assert round(age_grp.get_group(band).mean() * 100, 1) == exp_rate

    # 5. TenureBand
    tenure_grp = df.groupby("TenureBand")["Attrition"]
    tenures_expected = {
        "0-1": (75, 215, 34.9),
        "2-3": (47, 255, 18.4),
        "4-5": (40, 306, 13.1),
        "6-10": (55, 448, 12.3),
        "11+": (20, 246, 8.1),
    }
    for band, (exp_leavers, exp_n, exp_rate) in tenures_expected.items():
        assert tenure_grp.get_group(band).agg(["sum", "count"]).tolist() == [exp_leavers, exp_n]
        assert round(tenure_grp.get_group(band).mean() * 100, 1) == exp_rate

    # 6. EarlyTenure sensitivity
    le2 = df["YearsAtCompany"] <= 2
    gt2 = df["YearsAtCompany"] > 2
    assert int(le2.sum()) == 342
    assert int(y[le2].sum()) == 102
    assert round(float(y[le2].mean()) * 100, 1) == 29.8
    assert int(gt2.sum()) == 1128
    assert int(y[gt2].sum()) == 135
    assert round(float(y[gt2].mean()) * 100, 1) == 12.0

    # 7. OverTime
    ot_grp = df.groupby("OverTime")["Attrition"]
    assert ot_grp.get_group("Yes").agg(["sum", "count"]).tolist() == [127, 416]
    assert round(ot_grp.get_group("Yes").mean() * 100, 1) == 30.5
    assert ot_grp.get_group("No").agg(["sum", "count"]).tolist() == [110, 1054]
    assert round(ot_grp.get_group("No").mean() * 100, 1) == 10.4

    # 8. BusinessTravel
    bt_grp = df.groupby("BusinessTravel")["Attrition"]
    assert bt_grp.get_group("Non-Travel").agg(["sum", "count"]).tolist() == [12, 150]
    assert round(bt_grp.get_group("Non-Travel").mean() * 100, 1) == 8.0
    assert bt_grp.get_group("Travel_Rarely").agg(["sum", "count"]).tolist() == [156, 1043]
    assert round(bt_grp.get_group("Travel_Rarely").mean() * 100, 1) == 15.0
    assert bt_grp.get_group("Travel_Frequently").agg(["sum", "count"]).tolist() == [69, 277]
    assert round(bt_grp.get_group("Travel_Frequently").mean() * 100, 1) == 24.9

    # 9. DistanceBand
    dist_grp = df.groupby("DistanceBand")["Attrition"]
    dists_expected = {
        "1-5": (87, 632, 13.8),
        "6-10": (57, 394, 14.5),
        "11-20": (48, 240, 20.0),
        "21-29": (45, 204, 22.1),
    }
    for band, (exp_leavers, exp_n, exp_rate) in dists_expected.items():
        assert dist_grp.get_group(band).agg(["sum", "count"]).tolist() == [exp_leavers, exp_n]
        assert round(dist_grp.get_group(band).mean() * 100, 1) == exp_rate

    # 10. PromotionGapBand pooled
    p_pooled_grp = df.groupby("PromotionGapBand")["Attrition"]
    pooled_expected = {
        "0": (110, 581, 18.9),
        "1-2": (76, 516, 14.7),
        "3-5": (16, 158, 10.1),
        "6+": (35, 215, 16.3),
    }
    for band, (exp_leavers, exp_n, exp_rate) in pooled_expected.items():
        assert p_pooled_grp.get_group(band).agg(["sum", "count"]).tolist() == [exp_leavers, exp_n]
        assert round(p_pooled_grp.get_group(band).mean() * 100, 1) == exp_rate

    # 11. PromotionAnalysisCohort (n = 890)
    pac_df = df[df["PromotionAnalysisCohort"] == 1]
    assert len(pac_df) == 890
    pac_grp = pac_df.groupby("PromotionGapBand")["Attrition"]
    pac_expected = {
        "0": (23, 266, 8.6),
        "1-2": (23, 257, 8.9),
        "3-5": (15, 152, 9.9),
        "6+": (35, 215, 16.3),
    }
    for band, (exp_leavers, exp_n, exp_rate) in pac_expected.items():
        assert pac_grp.get_group(band).agg(["sum", "count"]).tolist() == [exp_leavers, exp_n]
        assert round(pac_grp.get_group(band).mean() * 100, 1) == exp_rate

    # 12. Two-way interactions
    def verify_cell(cond, exp_n, exp_rate):
        assert int(cond.sum()) == exp_n
        assert round(float(y[cond].mean()) * 100, 1) == exp_rate

    verify_cell((df["Department"] == "Sales") & (df["OverTime"] == "Yes"), 128, 37.5)
    verify_cell((df["Department"] == "Research & Development") & (df["OverTime"] == "Yes"), 271, 27.3)
    verify_cell((df["Department"] == "Research & Development") & (df["OverTime"] == "No"), 690, 8.6)
    verify_cell((df["JobRole"] == "Sales Representative") & (df["OverTime"] == "Yes"), 24, 66.7)
    verify_cell((df["JobRole"] == "Laboratory Technician") & (df["OverTime"] == "Yes"), 62, 50.0)
    verify_cell((df["AgeGroup"] == "18-25") & (df["OverTime"] == "Yes"), 39, 64.1)
    verify_cell((df["TenureBand"] == "0-1") & (df["OverTime"] == "Yes"), 69, 55.1)
    verify_cell((df["JobLevel"] == 1) & (df["OverTime"] == "Yes"), 156, 52.6)
    verify_cell((df["OverTime"] == "Yes") & (df["BusinessTravel"] == "Travel_Frequently"), 86, 41.9)

    # 13. JobLevel 1 overall
    assert int((df["JobLevel"] == 1).sum()) == 543
    assert round(float(y[df["JobLevel"] == 1].mean()) * 100, 1) == 26.3

    # 14. Leaver shares vs workforce shares
    rd_mask = df["Department"] == "Research & Development"
    assert round(float(y[rd_mask].sum() / y.sum()) * 100, 1) == 56.1
    assert round(float(rd_mask.mean()) * 100, 1) == 65.4

    ot_mask = df["OverTime"] == "Yes"
    assert round(float(y[ot_mask].sum() / y.sum()) * 100, 1) == 53.6
    assert round(float(ot_mask.mean()) * 100, 1) == 28.3
