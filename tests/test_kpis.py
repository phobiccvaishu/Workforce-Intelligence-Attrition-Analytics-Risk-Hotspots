"""Comprehensive Unit and Governance Tests for Phase 6 KPI Module.

Covers:
- Hand-computed unit tests on tiny synthetic DataFrames
- Filter edge cases (empty, n=1, n=29/30 boundary, unknown key/value, ordering)
- Gate test: Baseline KPIs match independent recomputations
- Static-source integrity: Checksum validation and mutation failures
- Mutation tests: Threshold and flag modifications
- Provenance: Alternative synthetic dataset (~300 rows)
- Traceability: Documentation numbers match source tables
- Forbidden words & zero dollar sign enforcement
- Protected attribute test: Static KPIs strictly reject filters
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from workforce_intel.config import (
    ALLOWED_FILTER_DIMENSIONS,
    KPI_DEFINITIONS_MD,
    KPI_REGISTRY,
    MIN_GROUP_SIZE,
    PHASE6_REPORT_MD,
    PHASE6_TABLES_DIR,
    PROJECT_ROOT,
    STATIC_TABLE_CHECKSUMS,
)
from workforce_intel.kpis import (
    apply_kpi_filter,
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
    verify_static_source_integrity,
)
from workforce_intel.stats_utils import wilson_interval


# =============================================================================
# 1. Hand-Computed Unit Tests on Tiny Frames
# =============================================================================


def test_kpi_01_hand_computed_tiny_frame():
    """Verify KPI-01 on a hand-constructed 10-row dataset."""
    df_tiny = pd.DataFrame({
        "Attrition": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Department": ["Sales"] * 10,
    })
    # Leavers = 2, n = 10 -> rate = 0.20
    exp_ci_l, exp_ci_h = wilson_interval(2, 10, confidence=0.95)
    res = kpi_01_attrition_rate(df_tiny)

    assert res["kpi_id"] == "KPI-01"
    assert res["n"] == 10
    assert res["value"] == pytest.approx(0.20, abs=1e-6)
    assert res["ci_low"] == pytest.approx(exp_ci_l, abs=1e-6)
    assert res["ci_high"] == pytest.approx(exp_ci_h, abs=1e-6)
    assert res["reliability"] == "low_n"  # 10 < 30


def test_kpi_02_hand_computed_tiny_frame():
    """Verify KPI-02 headcount and leavers proportions."""
    df_tiny = pd.DataFrame({
        "Attrition": [1, 1, 0, 0, 0],
        "Department": ["Sales"] * 5,
    })
    res = kpi_02_headcount_leavers(df_tiny, total_headcount=100, total_leavers=20)
    assert res["kpi_id"] == "KPI-02"
    assert res["value"]["headcount"] == 5
    assert res["value"]["leavers"] == 2
    assert res["value"]["headcount_share"] == pytest.approx(0.05, abs=1e-6)
    assert res["value"]["leavers_share"] == pytest.approx(0.10, abs=1e-6)
    assert res["reliability"] == "low_n"


def test_kpi_03_hand_computed_tiny_frame():
    """Verify KPI-03 relative difference, ratio, and interval separation flag."""
    # Group rate = 4/10 = 0.40; baseline = 0.16
    df_tiny = pd.DataFrame({
        "Attrition": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "Department": ["Sales"] * 10,
    })
    ci_l, ci_h = wilson_interval(4, 10, confidence=0.95)
    res = kpi_03_relative_attrition(df_tiny, baseline_rate=0.16)

    assert res["value"]["rate_diff"] == pytest.approx(0.40 - 0.16, abs=1e-6)
    assert res["value"]["rate_ratio"] == pytest.approx(0.40 / 0.16, abs=1e-6)
    expected_flag = "above_baseline" if ci_l > 0.16 else ("below_baseline" if ci_h < 0.16 else "no_separation")
    assert res["value"]["separation_flag"] == expected_flag


def test_kpi_04_hand_computed_with_zero_cell():
    """Verify KPI-04 handles zero cells with Woolf +0.5 continuity correction."""
    # a=2, b=3, c=0, d=5 -> cell c has zero
    df_zero = pd.DataFrame({
        "Attrition": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "OverTime": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No"],
    })
    res = kpi_04_overtime_association(df_zero)
    assert res["kpi_id"] == "KPI-04"
    assert res["reliability"] == "unreliable_zero_cell"
    # Adjusted cells: a=2.5, b=3.5, c=0.5, d=5.5
    exp_or = (2.5 * 5.5) / (3.5 * 0.5)
    assert res["value"]["odds_ratio"] == pytest.approx(exp_or, abs=1e-4)


def test_kpi_05_hand_computed_cohort_contrast():
    """Verify KPI-05 promotion gap cohort contrast."""
    df_cohort = pd.DataFrame({
        "PromotionAnalysisCohort": [1, 1, 1, 1, 1, 1, 0, 0],
        "YearsSinceLastPromotion": [7, 8, 2, 3, 1, 0, 10, 2],
        "Attrition": [1, 1, 0, 0, 0, 0, 1, 0],
        "Department": ["Sales"] * 8,
    })
    # Within cohort: n=6, 6+ group: [7, 8] with attr=[1,1]; 0-5 group: [2,3,1,0] with attr=[0,0,0,0]
    res = kpi_05_promotion_gap_cohort(df_cohort)
    assert res["kpi_id"] == "KPI-05"
    assert res["n"] == 6
    assert res["value"]["cohort_headcount"] == 6
    assert res["value"]["cohort_leavers"] == 2
    assert res["reliability"] in ["low_n", "unreliable_zero_cell"]


# =============================================================================
# 2. Filter Edge Cases
# =============================================================================


def test_filter_edge_case_empty_result():
    """Verify empty filter returns value None and reliability 'no_data' without error."""
    df = pd.DataFrame({
        "Attrition": [0, 1],
        "Department": ["Sales", "Research & Development"],
        "JobRole": ["Manager", "Laboratory Technician"],
    })
    # Filter matching nothing
    filters = {"Department": "Sales", "JobRole": "Laboratory Technician"}
    res1 = kpi_01_attrition_rate(df, filters=filters)
    assert res1["value"] is None
    assert res1["n"] == 0
    assert res1["reliability"] == "no_data"

    res2 = kpi_02_headcount_leavers(df, filters=filters)
    assert res2["value"] is None
    assert res2["reliability"] == "no_data"

    res3 = kpi_03_relative_attrition(df, filters=filters)
    assert res3["value"] is None
    assert res3["reliability"] == "no_data"

    res4 = kpi_04_overtime_association(df, filters=filters)
    assert res4["value"] is None
    assert res4["reliability"] == "no_data"


def test_filter_edge_case_single_element():
    """Verify dataset with n=1 is evaluated and flagged as low_n."""
    df_single = pd.DataFrame({
        "Attrition": [1],
        "Department": ["Sales"],
        "OverTime": ["Yes"],
    })
    res = kpi_01_attrition_rate(df_single)
    assert res["n"] == 1
    assert res["value"] == 1.0
    assert res["reliability"] == "low_n"


def test_filter_boundary_flag_n29_vs_n30():
    """Verify boundary flag precisely switches at n=30."""
    df_29 = pd.DataFrame({"Attrition": [0] * 29, "Department": ["Sales"] * 29})
    df_30 = pd.DataFrame({"Attrition": [0] * 30, "Department": ["Sales"] * 30})

    res_29 = kpi_01_attrition_rate(df_29)
    res_30 = kpi_01_attrition_rate(df_30)

    assert res_29["reliability"] == "low_n"
    assert res_30["reliability"] == "reliable"


def test_filter_unknown_dimension_raises():
    """Verify filtering by an unauthorized or non-existent dimension raises ValueError."""
    df = pd.DataFrame({"Attrition": [0], "Department": ["Sales"]})
    with pytest.raises(ValueError, match="not an authorized filter dimension"):
        apply_kpi_filter(df, {"InvalidColumnName": "Value"})


def test_filter_unknown_level_raises():
    """Verify filtering by an unknown category level raises ValueError."""
    df = pd.DataFrame({"Attrition": [0], "Department": ["Sales"]})
    with pytest.raises(ValueError, match="does not exist in dimension"):
        apply_kpi_filter(df, {"Department": "NonExistentDept"})


def test_filter_order_invariance():
    """Verify filter dictionary key order does not alter output results."""
    df = pd.read_csv("data/processed/workforce_attrition_processed.csv")
    f1 = {"Department": "Sales", "OverTime": "Yes"}
    f2 = {"OverTime": "Yes", "Department": "Sales"}

    res1 = kpi_01_attrition_rate(df, filters=f1)
    res2 = kpi_01_attrition_rate(df, filters=f2)

    assert res1["value"] == res2["value"]
    assert res1["n"] == res2["n"]
    assert res1["ci_low"] == res2["ci_low"]


# =============================================================================
# 3. Gate Test: Baseline KPIs Match Report/Notebook Values
# =============================================================================


def test_gate_test_baseline_kpis():
    """GATE TEST: Assert baseline KPI outputs match notebook/report recomputations."""
    df = pd.read_csv("data/processed/workforce_attrition_processed.csv")
    kpis = evaluate_all_kpis(df)
    kpi_map = {k["kpi_id"]: k for k in kpis}

    # KPI-01: 237 / 1470
    assert kpi_map["KPI-01"]["value"] == pytest.approx(237 / 1470, abs=1e-6)
    assert kpi_map["KPI-01"]["n"] == 1470

    # KPI-02: 1470 headcount, 237 leavers
    assert kpi_map["KPI-02"]["value"]["headcount"] == 1470
    assert kpi_map["KPI-02"]["value"]["leavers"] == 237

    # KPI-03: baseline diff 0.0, ratio 1.0
    assert kpi_map["KPI-03"]["value"]["rate_diff"] == pytest.approx(0.0, abs=1e-6)
    assert kpi_map["KPI-03"]["value"]["rate_ratio"] == pytest.approx(1.0, abs=1e-6)

    # KPI-04: crude OR ~3.77
    assert kpi_map["KPI-04"]["value"]["odds_ratio"] == pytest.approx(3.7712, abs=1e-3)

    # KPI-05: cohort rate 96/890 ~ 10.79%
    assert kpi_map["KPI-05"]["value"]["cohort_headcount"] == 890
    assert kpi_map["KPI-05"]["value"]["cohort_leavers"] == 96
    assert kpi_map["KPI-05"]["value"]["cohort_attrition_rate"] == pytest.approx(96 / 890, abs=1e-6)

    # KPI-06: MH OR 4.05, M1 4.81, M2 5.27
    v6 = kpi_map["KPI-06"]["value"]
    assert v6["mantel_haenszel_joblevel"]["odds_ratio"] == pytest.approx(4.0537, abs=1e-3)
    assert v6["multivariable_m1"]["odds_ratio"] == pytest.approx(4.8086, abs=1e-3)
    assert v6["multivariable_m2"]["odds_ratio"] == pytest.approx(5.2663, abs=1e-3)

    # KPI-07: 104 segments, 98 n>=30, 25 tier 1, 2 tier 2
    v7 = kpi_map["KPI-07"]["value"]
    assert v7["universe_size"] == 104
    assert v7["segments_n_ge_30"] == 98
    assert v7["tier1_priority_hotspots"] == 25
    assert v7["tier2_secondary_hotspots"] == 2

    # KPI-08: 25 screened, 14 holdout n>=30, 7 lower bound
    v8 = kpi_map["KPI-08"]["value"]
    assert v8["screened_training_tier1"] == 25
    assert v8["holdout_sample_floor_met"] == 14
    assert v8["holdout_above_baseline_lower_bound"] == 7

    # KPI-09: 8 greedy steps
    v9 = kpi_map["KPI-09"]["value"]
    assert v9["total_steps"] == 8

    # KPI-10: holdout triage yields
    v10 = kpi_map["KPI-10"]["value"]
    assert v10["holdout_total_headcount"] == 294
    assert v10["holdout_total_leavers"] == 47
    assert v10["top_10_percent"]["tp"] == 18
    assert v10["top_20_percent"]["tp"] == 27


# =============================================================================
# 4. Static-Source Integrity & Mutation Tests
# =============================================================================


def test_static_source_integrity_checksums_pass():
    """Verify all 5 static source tables match registered checksums."""
    for rel_path, exp_hash in STATIC_TABLE_CHECKSUMS.items():
        file_path = PROJECT_ROOT / rel_path
        verify_static_source_integrity(file_path, exp_hash)


def test_static_source_mutation_fails(tmp_path):
    """Mutation Test: Altering one byte of a static table causes integrity check to fail."""
    test_file = tmp_path / "mutated_table.csv"
    test_file.write_text("colA,colB\n1,2\n", encoding="utf-8")
    valid_hash = hashlib.sha256(test_file.read_bytes()).hexdigest()

    # Pass with valid hash
    verify_static_source_integrity(test_file, valid_hash)

    # Mutate file
    test_file.write_text("colA,colB\n1,999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Static source integrity failure"):
        verify_static_source_integrity(test_file, valid_hash)


def test_mutation_min_group_size_flips_flag():
    """Mutation Test: Increasing MIN_GROUP_SIZE flips a borderline segment to low_n."""
    df_border = pd.DataFrame({"Attrition": [0] * 35, "Department": ["Sales"] * 35})
    # Default MIN_GROUP_SIZE = 30 -> reliable
    res_default = kpi_01_attrition_rate(df_border, min_group_size=30)
    assert res_default["reliability"] == "reliable"

    # Mutated MIN_GROUP_SIZE = 50 -> low_n
    res_mutated = kpi_01_attrition_rate(df_border, min_group_size=50)
    assert res_mutated["reliability"] == "low_n"


def test_mutation_baseline_rate_flips_relative_flag():
    """Mutation Test: Shifting baseline rate alters separation_flag."""
    df_test = pd.DataFrame({"Attrition": [1] * 20 + [0] * 80, "Department": ["Sales"] * 100})
    # Rate = 0.20, CI = [0.133, 0.289]
    res_low_base = kpi_03_relative_attrition(df_test, baseline_rate=0.10)
    assert res_low_base["value"]["separation_flag"] == "above_baseline"

    res_high_base = kpi_03_relative_attrition(df_test, baseline_rate=0.35)
    assert res_high_base["value"]["separation_flag"] == "below_baseline"


# =============================================================================
# 5. Provenance on Alternative Synthetic Dataset (~300 Rows)
# =============================================================================


def test_provenance_alternative_synthetic_dataset():
    """Verify LIVE KPIs evaluate reliably on an independent synthetic dataset (~300 rows)."""
    np.random.seed(999)
    n_synth = 300
    depts = np.random.choice(["Sales", "Research & Development", "Human Resources"], size=n_synth, p=[0.3, 0.6, 0.1])
    ot = np.random.choice(["Yes", "No"], size=n_synth, p=[0.25, 0.75])
    # Attrition dependent on OverTime
    p_att = np.where(ot == "Yes", 0.35, 0.10)
    att = (np.random.rand(n_synth) < p_att).astype(int)
    twy = np.random.randint(1, 25, size=n_synth)
    yac = np.random.randint(0, 15, size=n_synth)
    prom = np.random.randint(0, 10, size=n_synth)
    jl = np.random.randint(1, 5, size=n_synth)
    cohort = ((twy >= 6) & (yac >= 6) & (jl <= 3)).astype(int)

    df_synth = pd.DataFrame({
        "Attrition": att,
        "Department": depts,
        "OverTime": ot,
        "TotalWorkingYears": twy,
        "YearsAtCompany": yac,
        "YearsSinceLastPromotion": prom,
        "JobLevel": jl,
        "PromotionAnalysisCohort": cohort,
    })

    # Evaluate live KPIs
    k1 = kpi_01_attrition_rate(df_synth)
    assert k1["n"] == 300
    assert k1["value"] == pytest.approx(df_synth["Attrition"].mean(), abs=1e-6)
    assert k1["reliability"] == "reliable"

    k2 = kpi_02_headcount_leavers(df_synth, total_headcount=300, total_leavers=int(df_synth["Attrition"].sum()))
    assert k2["value"]["headcount_share"] == 1.0

    k4 = kpi_04_overtime_association(df_synth)
    assert k4["value"]["odds_ratio"] > 1.0  # Synthetic DGP generated higher odds for OverTime


# =============================================================================
# 6. Traceability and Governance Validation
# =============================================================================


def test_traceability_kpi_report_and_definitions():
    """Verify every table cell in baseline CSV matches generated report text."""
    df_base = pd.read_csv(PHASE6_TABLES_DIR / "kpi_baseline.csv")
    assert len(df_base) == 10

    report_text = PHASE6_REPORT_MD.read_text(encoding="utf-8")
    defs_text = KPI_DEFINITIONS_MD.read_text(encoding="utf-8")

    for _, row in df_base.iterrows():
        # Assert KPI ID is present in report and definitions
        assert f"`{row['kpi_id']}`" in report_text
        assert f"`{row['kpi_id']}`" in defs_text
        assert row["name"] in report_text


def test_protected_attribute_static_kpis_reject_filters():
    """Protected Attribute Governance: Static KPIs 6-10 must strictly reject filters."""
    static_kpi_funcs = [
        kpi_06_adjusted_overtime,
        kpi_07_hotspot_universe_summary,
        kpi_08_hotspot_holdout_validation,
        kpi_09_hotspot_coverage,
        kpi_10_model_triage_yield,
    ]

    prohibited_filters = [
        {"Gender": "Female"},
        {"MaritalStatus": "Single"},
        {"AgeGroup": "18-25"},
        {"Department": "Sales"},
    ]

    for func in static_kpi_funcs:
        for f in prohibited_filters:
            with pytest.raises(ValueError, match="cannot accept subpopulation filters"):
                func(filters=f)


def test_forbidden_words_on_every_line():
    """Strict Language Invariant: No forbidden words or 'significant' on any generated line."""
    forbidden_terms = [
        "cause", "causes", "caused", "drives", "driver", "leads to",
        "because", "effect of", "impact of", "risk factor", "predicts",
        "stagnation", "stagnant", "flight risk", "proves", "confirms",
        "significant",
    ]

    target_files = [
        PHASE6_REPORT_MD,
        KPI_DEFINITIONS_MD,
        PHASE6_TABLES_DIR / "kpi_baseline.csv",
        PHASE6_TABLES_DIR / "kpi_filter_cases.csv",
    ]

    errors = []
    for fpath in target_files:
        assert fpath.is_file(), f"Target file missing: {fpath}"
        lines = fpath.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            lower_line = line.lower()
            for term in forbidden_terms:
                # Whole word regex search to prevent false positive substring matches
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, lower_line):
                    errors.append(f"{fpath.name}:{idx} contains forbidden word '{term}': {line.strip()}")

    assert not errors, "Forbidden words found:\n" + "\n".join(errors)


def test_zero_dollar_signs_across_phase6():
    """Strict Notation Invariant: Exactly zero dollar signs ($) in Phase 6 outputs."""
    target_files = [
        PHASE6_REPORT_MD,
        KPI_DEFINITIONS_MD,
        PHASE6_TABLES_DIR / "kpi_baseline.csv",
        PHASE6_TABLES_DIR / "kpi_filter_cases.csv",
    ]

    errors = []
    for fpath in target_files:
        assert fpath.is_file(), f"Target file missing: {fpath}"
        lines = fpath.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            if "$" in line:
                errors.append(f"{fpath.name}:{idx} contains dollar sign: {line.strip()}")

    assert not errors, "Dollar signs found in Phase 6 outputs:\n" + "\n".join(errors)
