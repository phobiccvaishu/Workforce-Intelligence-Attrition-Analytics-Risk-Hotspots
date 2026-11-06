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

def test_phase3b_section_6_checkpoints():
    """Verify Section 6 reference checkpoints computed directly from processed data."""
    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)
    n_tot = len(df)
    leavers_tot = int(df["Attrition"].sum())
    overall_rate = (leavers_tot / n_tot) * 100.0

    # Wilson score interval calculation directly in pandas
    from scipy.stats import norm
    z = norm.ppf(0.975)

    def wilson_ci(k, n):
        p = k / n
        denom = 1 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        delta = (z * ((p * (1 - p) / n + (z**2) / (4 * (n**2))) ** 0.5)) / denom
        return (center - delta) * 100.0, (center + delta) * 100.0

    dims = [
        "Department", "JobRole", "JobLevel", "AgeGroup", "Gender", "MaritalStatus",
        "Education", "EducationField", "TenureBand", "EarlyTenure", "CareerStage",
        "RoleTenureBand", "ManagerTenureBand", "PromotionGapBand", "PromotionGapEqualsTenure",
        "HasPriorExperience", "NumCompaniesWorked", "OverTime", "BusinessTravel",
        "DistanceBand", "WorkLifeBalance", "JobSatisfaction", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "JobInvolvement", "StockOptionLevel",
        "TrainingTimesLastYear", "PerformanceRating",
    ]

    above_flags = {}
    below_flags = {}
    no_sep_dims = []
    segments_ge_30 = 0
    segments_lt_30 = []

    for dim in dims:
        grp = df.groupby(dim)["Attrition"].agg(["count", "sum"])
        dim_above = []
        dim_below = []
        for val, (cnt, lvs) in grp.iterrows():
            if cnt >= 30:
                segments_ge_30 += 1
            else:
                segments_lt_30.append((dim, str(val), int(cnt)))
                continue
            lb, ub = wilson_ci(lvs, cnt)
            if lb > overall_rate:
                dim_above.append(str(val))
            elif ub < overall_rate:
                dim_below.append(str(val))
        if dim_above:
            above_flags[dim] = dim_above
        if dim_below:
            below_flags[dim] = dim_below
        if not dim_above and not dim_below:
            no_sep_dims.append(dim)

    # Check: evaluated segments with n >= 30 across 28 dimensions = 115; exactly one below 30
    assert segments_ge_30 == 115, f"Expected 115 evaluated segments (n >= 30), got {segments_ge_30}"
    assert segments_lt_30 == [("EducationField", "Human Resources", 27)], f"Unexpected low-n segments: {segments_lt_30}" 

    # Check 1: Flag counts
    total_above = sum(len(v) for v in above_flags.values())
    total_below = sum(len(v) for v in below_flags.values())
    assert total_above == 26, f"Expected 26 above, got {total_above}: {above_flags}"
    assert total_below == 26, f"Expected 26 below, got {total_below}: {below_flags}"
    assert sorted(no_sep_dims) == ["Education", "Gender", "HasPriorExperience", "PerformanceRating"]

    # Check 2: Specific flagged segments
    assert above_flags["Department"] == ["Sales"]
    assert "Department" not in below_flags

    assert sorted(above_flags["JobRole"]) == ["Laboratory Technician", "Sales Representative"]
    assert sorted(below_flags["JobRole"]) == ["Healthcare Representative", "Manager", "Manufacturing Director", "Research Director"]

    assert above_flags["JobLevel"] == ["1"]
    assert sorted(below_flags["JobLevel"]) == ["2", "4", "5"]

    assert sorted(above_flags["AgeGroup"]) == ["18-25", "26-35"]
    assert below_flags["AgeGroup"] == ["36-45"]

    assert above_flags["MaritalStatus"] == ["Single"]
    assert sorted(below_flags["MaritalStatus"]) == ["Divorced", "Married"]

    assert sorted(above_flags["EducationField"]) == ["Marketing", "Technical Degree"]

    assert above_flags["OverTime"] == ["Yes"]
    assert below_flags["OverTime"] == ["No"]

    # OverTime x JobLevel independent checkpoints: separated = levels 1 and 2; overlapping = levels 3 and 4; not assessed = level 5
    ot_jl_sep = []
    ot_jl_ovp = []
    ot_jl_low = []
    for jl_val in [1, 2, 3, 4, 5]:
        s_yes = df[(df["JobLevel"] == jl_val) & (df["OverTime"] == "Yes")]
        s_no = df[(df["JobLevel"] == jl_val) & (df["OverTime"] == "No")]
        n_y, l_y = len(s_yes), int(s_yes["Attrition"].sum())
        n_n, l_n = len(s_no), int(s_no["Attrition"].sum())
        if n_y < 30 or n_n < 30:
            ot_jl_low.append(jl_val)
        else:
            lb_y, ub_y = wilson_ci(l_y, n_y)
            lb_n, ub_n = wilson_ci(l_n, n_n)
            if lb_y > ub_n:
                ot_jl_sep.append(jl_val)
            elif (l_y / n_y) > (l_n / n_n):
                ot_jl_ovp.append(jl_val)

    assert ot_jl_sep == [1, 2], f"Expected separated levels [1, 2], got {ot_jl_sep}"
    assert ot_jl_ovp == [3, 4], f"Expected overlapping levels [3, 4], got {ot_jl_ovp}"
    assert ot_jl_low == [5], f"Expected not assessed levels [5], got {ot_jl_low}"

    assert above_flags["BusinessTravel"] == ["Travel_Frequently"]
    assert below_flags["BusinessTravel"] == ["Non-Travel"]

    assert above_flags["DistanceBand"] == ["21-29"]
    assert "DistanceBand" not in below_flags

    assert "PromotionGapBand" not in above_flags
    assert below_flags["PromotionGapBand"] == ["3-5"]

    assert above_flags["WorkLifeBalance"] == ["1"]
    assert int(df[df["WorkLifeBalance"] == 1]["Attrition"].count()) == 80

    assert above_flags["StockOptionLevel"] == ["0"]
    assert sorted(below_flags["StockOptionLevel"]) == ["1", "2"]

    assert above_flags["TrainingTimesLastYear"] == ["0"]
    assert int(df[df["TrainingTimesLastYear"] == 0]["Attrition"].count()) == 54

    assert above_flags["EarlyTenure"] == ["1"]
    early_t_cnt = int(df[df["EarlyTenure"] == 1]["Attrition"].count())
    early_t_lvs = int(df[df["EarlyTenure"] == 1]["Attrition"].sum())
    assert early_t_cnt == 215
    assert round((early_t_lvs / early_t_cnt) * 100, 1) == 34.9

    # Check 3: Cohort promotion gap baseline (96 / 890 = 10.79%)
    cohort_df = df[df["PromotionAnalysisCohort"] == 1]
    cohort_n = len(cohort_df)
    cohort_lvs = int(cohort_df["Attrition"].sum())
    assert cohort_n == 890
    assert cohort_lvs == 96
    cohort_rate = (cohort_lvs / cohort_n) * 100.0
    assert round(cohort_rate, 1) == 10.8

    c_grp = cohort_df.groupby("PromotionGapBand")["Attrition"].agg(["count", "sum"])
    for val, (cnt, lvs) in c_grp.iterrows():
        lb, ub = wilson_ci(lvs, cnt)
        if val == "6+":
            assert lb > cohort_rate, f"Cohort 6+ lower bound {lb} must exceed cohort rate {cohort_rate}"
            assert round((lvs / cnt) * 100, 1) == 16.3
            assert round(lb, 1) == 11.9
            assert round(ub, 1) == 21.8
        else:
            assert lb <= cohort_rate <= ub, f"Cohort band {val} should not separate from cohort rate"

    # Check 4: Hotspot candidate union
    # 16 candidate cells cover 599 employees (40.7%) and 171 leavers (72.2%), union rate 28.5%
    # Outside: 871 employees, 66 leavers, 7.6%
    hotspots = pd.read_csv("reports/phase3/tables/hotspot_candidates.csv")
    assert len(hotspots) == 16

    union_mask = pd.Series(False, index=df.index)
    for _, row in hotspots.iterrows():
        d1, d2 = row["interaction"].split(" x ")
        subgroup = row["subgroup"]
        p1, p2 = subgroup.split(f" & {d2}=")
        v1 = p1.replace(f"{d1}=", "")
        v2 = p2
        c_mask = pd.Series(True, index=df.index)
        for c, v in [(d1, v1), (d2, v2)]:
            if c in ["JobLevel", "Education", "JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance", "PerformanceRating", "StockOptionLevel"]:
                c_mask = c_mask & (df[c] == int(v))
            else:
                c_mask = c_mask & (df[c].astype(str) == v)
        union_mask = union_mask | c_mask

    u_n = int(union_mask.sum())
    u_lvs = int(df.loc[union_mask, "Attrition"].sum())
    assert u_n == 599
    assert u_lvs == 171
    assert round((u_n / n_tot) * 100, 1) == 40.7
    assert round((u_lvs / leavers_tot) * 100, 1) == 72.2
    assert round((u_lvs / u_n) * 100, 1) == 28.5

    out_mask = ~union_mask
    out_n = int(out_mask.sum())
    out_lvs = int(df.loc[out_mask, "Attrition"].sum())
    assert out_n == 871
    assert out_lvs == 66
    assert round((out_lvs / out_n) * 100, 1) == 7.6


def test_phase5_section_6_reference_checkpoints():
    """Verify Phase 5 Section 6 and owner-supplied reference checkpoints computed independently."""
    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)

    # 1. OverTime crude 2x2
    # 127 leavers / 289 stayers (Yes) vs 110 / 944 (No); crude OR about 3.77
    ot_yes = df[df["OverTime"] == "Yes"]
    ot_no = df[df["OverTime"] == "No"]
    a = int(ot_yes["Attrition"].sum())
    b = len(ot_yes) - a
    c = int(ot_no["Attrition"].sum())
    d = len(ot_no) - c

    assert a == 127
    assert b == 289
    assert c == 110
    assert d == 944
    crude_or = (a * d) / (b * c)
    assert crude_or == pytest.approx(3.7712, abs=0.02)

    # 2. Stratum ORs by JobLevel
    # L1 82/74 | 61/326 -> 5.92
    # L2 26/120 | 26/362 -> 3.02
    # L3 13/50 | 19/136 -> 1.86
    # L4 3/30 | 2/71 -> 3.55
    # L5 3/15 | 2/49 -> 4.90
    expected_jl_counts = {
        1: (82, 74, 61, 326, 5.9220),
        2: (26, 120, 26, 362, 3.0167),
        3: (13, 50, 19, 136, 1.8611),
        4: (3, 30, 2, 71, 3.5500),
        5: (3, 15, 2, 49, 4.9000),
    }
    for jl, (exp_a, exp_b, exp_c, exp_d, exp_or) in expected_jl_counts.items():
        s = df[df["JobLevel"] == jl]
        sa = int(((s["OverTime"] == "Yes") & (s["Attrition"] == 1)).sum())
        sb = int(((s["OverTime"] == "Yes") & (s["Attrition"] == 0)).sum())
        sc = int(((s["OverTime"] == "No") & (s["Attrition"] == 1)).sum())
        sd = int(((s["OverTime"] == "No") & (s["Attrition"] == 0)).sum())
        assert (sa, sb, sc, sd) == (exp_a, exp_b, exp_c, exp_d)
        s_or = (sa * sd) / (sb * sc)
        assert s_or == pytest.approx(exp_or, abs=0.02)

    # Read committed Phase 5 tables
    mh_summary = pd.read_csv("reports/phase5/tables/mantel_haenszel.csv")

    # OverTime by JobLevel: MH OR 4.05 [3.01, 5.46], Breslow-Day (Tarone) p 0.084
    mh_jl = mh_summary[mh_summary["stratifier"] == "JobLevel"].iloc[0]
    assert mh_jl["mh_or"] == pytest.approx(4.05, abs=0.02)
    assert mh_jl["mh_ci_low"] == pytest.approx(3.01, abs=0.05)
    assert mh_jl["mh_ci_high"] == pytest.approx(5.46, abs=0.05)
    assert mh_jl["breslow_day_p"] == pytest.approx(0.084, abs=0.02)

    # OverTime by Department: MH OR 3.81 [2.85, 5.08], p 0.73
    mh_dept = mh_summary[mh_summary["stratifier"] == "Department"].iloc[0]
    assert mh_dept["mh_or"] == pytest.approx(3.81, abs=0.02)
    assert mh_dept["mh_ci_low"] == pytest.approx(2.85, abs=0.05)
    assert mh_dept["mh_ci_high"] == pytest.approx(5.08, abs=0.05)
    assert mh_dept["breslow_day_p"] == pytest.approx(0.73, abs=0.02)

    # OverTime by TenureBand: MH OR 3.85 [2.86, 5.18], p 0.28
    mh_tb = mh_summary[(mh_summary["stratifier"] == "TenureBand") & (mh_summary["analysis_id"] == "MH-03")].iloc[0]
    assert mh_tb["mh_or"] == pytest.approx(3.85, abs=0.02)
    assert mh_tb["mh_ci_low"] == pytest.approx(2.86, abs=0.05)
    assert mh_tb["mh_ci_high"] == pytest.approx(5.18, abs=0.05)
    assert mh_tb["breslow_day_p"] == pytest.approx(0.28, abs=0.02)

    # Cohort promotion gap 6+ vs 0-5 by TenureBand: crude 1.96, MH 2.38 [1.44, 3.92], p 0.61
    mh_c = mh_summary[mh_summary["analysis_id"] == "MH-04"].iloc[0]
    assert mh_c["crude_or"] == pytest.approx(1.96, abs=0.02)
    assert mh_c["mh_or"] == pytest.approx(2.38, abs=0.02)
    assert mh_c["mh_ci_low"] == pytest.approx(1.44, abs=0.05)
    assert mh_c["mh_ci_high"] == pytest.approx(3.92, abs=0.05)
    assert mh_c["breslow_day_p"] == pytest.approx(0.61, abs=0.02)


def test_phase6_kpi_reference_checkpoints():
    """Verify Section 6 hand-derived reference checkpoints for Phase 6 KPI module."""
    from workforce_intel.kpis import (
        evaluate_all_kpis,
        kpi_01_attrition_rate,
        kpi_02_headcount_leavers,
        kpi_03_relative_attrition,
        kpi_04_overtime_association,
        kpi_05_promotion_gap_cohort,
        kpi_06_adjusted_overtime,
        kpi_07_hotspot_universe_summary,
        kpi_08_hotspot_holdout_validation,
        kpi_09_hotspot_coverage,
        kpi_10_model_triage_yield,
    )

    df = pd.read_csv("data/processed/workforce_attrition_processed.csv")

    # Overall 237/1470 = 16.12% (tolerance 0.0005)
    k1 = kpi_01_attrition_rate(df)
    assert k1["n"] == 1470
    assert k1["value"] == pytest.approx(237 / 1470, abs=0.0005)
    assert k1["ci_low"] == pytest.approx(0.1433, abs=0.001)
    assert k1["ci_high"] == pytest.approx(0.1809, abs=0.001)
    assert k1["reliability"] == "reliable"

    # Cohort 96/890 = 10.79% (tolerance 0.0005)
    k5 = kpi_05_promotion_gap_cohort(df)
    assert k5["n"] == 890
    assert k5["value"]["cohort_headcount"] == 890
    assert k5["value"]["cohort_leavers"] == 96
    assert k5["value"]["cohort_attrition_rate"] == pytest.approx(96 / 890, abs=0.0005)
    assert k5["value"]["crude_or_6plus_vs_0to5"] == pytest.approx(1.96, abs=0.02)

    # OverTime crude OR 3.77 (tolerance 0.02)
    k4 = kpi_04_overtime_association(df)
    assert k4["n"] == 1470
    assert k4["value"]["odds_ratio"] == pytest.approx(3.77, abs=0.02)
    assert k4["value"]["overtime_share"] == pytest.approx(416 / 1470, abs=0.001)

    # MH by JobLevel 4.05 [3.01, 5.46]; M1 OverTime 4.81, M2 5.27
    k6 = kpi_06_adjusted_overtime()
    v6 = k6["value"]
    assert v6["mantel_haenszel_joblevel"]["odds_ratio"] == pytest.approx(4.05, abs=0.02)
    assert v6["mantel_haenszel_joblevel"]["ci_low"] == pytest.approx(3.01, abs=0.05)
    assert v6["mantel_haenszel_joblevel"]["ci_high"] == pytest.approx(5.46, abs=0.05)
    assert v6["multivariable_m1"]["odds_ratio"] == pytest.approx(4.81, abs=0.02)
    assert v6["multivariable_m2"]["odds_ratio"] == pytest.approx(5.27, abs=0.02)

    # Hotspot 104 / 98 / 25 / 2
    k7 = kpi_07_hotspot_universe_summary()
    v7 = k7["value"]
    assert v7["universe_size"] == 104
    assert v7["segments_n_ge_30"] == 98
    assert v7["tier1_priority_hotspots"] == 25
    assert v7["tier2_secondary_hotspots"] == 2

    # Split-sample 25 screened, 14 with holdout n >= 30, 7 above by lower bound
    k8 = kpi_08_hotspot_holdout_validation()
    v8 = k8["value"]
    assert v8["screened_training_tier1"] == 25
    assert v8["holdout_sample_floor_met"] == 14
    assert v8["holdout_above_baseline_lower_bound"] == 7
    assert v8["sample_floor_proportion"] == "14/25"
    assert v8["lower_bound_replication_proportion"] == "7/14"

    # Triage top-10% 18 of 29 and top-20% 27 of 59 (47 holdout leavers)
    k10 = kpi_10_model_triage_yield()
    v10 = k10["value"]
    assert v10["holdout_total_headcount"] == 294
    assert v10["holdout_total_leavers"] == 47
    assert v10["holdout_base_rate"] == pytest.approx(47 / 294, abs=0.0005)
    assert v10["top_10_percent"]["k"] == 29
    assert v10["top_10_percent"]["tp"] == 18
    assert v10["top_20_percent"]["k"] == 59
    assert v10["top_20_percent"]["tp"] == 27



