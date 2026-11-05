"""Provenance test for Phase 3A report generator on synthetic data."""

from datetime import timezone
from datetime import datetime
import pandas as pd
import pytest

from workforce_intel.eda import (
    concentration_table,
    funnel_table,
    hotspot_candidates,
    overall_summary,
    rate_table,
    two_way_table,
)
from workforce_intel.reporting import build_phase3a_report


def test_phase3a_report_provenance_synthetic_dataset():
    """Verify that build_phase3a_report generates text strictly from supplied tables."""
    # Synthetic dataset with 100 rows, 33 leavers (33.00%), distinct department names
    synth_df = pd.DataFrame({
        "Department": ["CyberSecurity"] * 60 + ["CloudEngineering"] * 40,
        "JobRole": ["SecEngineer"] * 60 + ["CloudArchitect"] * 40,
        "JobLevel": [1] * 50 + [2] * 50,
        "AgeGroup": ["26-35"] * 100,
        "Gender": ["Female"] * 50 + ["Male"] * 50,
        "MaritalStatus": ["Single"] * 100,
        "TenureBand": ["2-3"] * 100,
        "EarlyTenure": [0] * 100,
        "OverTime": ["Yes"] * 40 + ["No"] * 60,
        "BusinessTravel": ["Travel_Rarely"] * 100,
        "DistanceBand": ["1-5"] * 100,
        "PromotionAnalysisCohort": [1] * 100,
        "PromotionGapBand": ["1-2"] * 100,
        "Attrition": [1] * 33 + [0] * 67,
    })

    tables_dict = {
        "overall_baseline": overall_summary(synth_df),
        "department_attrition": rate_table(synth_df, "Department"),
        "jobrole_attrition": rate_table(synth_df, "JobRole"),
        "joblevel_attrition": rate_table(synth_df, "JobLevel"),
        "agegroup_attrition": rate_table(synth_df, "AgeGroup"),
        "gender_attrition": rate_table(synth_df, "Gender"),
        "maritalstatus_attrition": rate_table(synth_df, "MaritalStatus"),
        "tenureband_attrition": rate_table(synth_df, "TenureBand"),
        "earlytenure_attrition": rate_table(synth_df, "EarlyTenure"),
        "overtime_attrition": rate_table(synth_df, "OverTime"),
        "businesstravel_attrition": rate_table(synth_df, "BusinessTravel"),
        "distanceband_attrition": rate_table(synth_df, "DistanceBand"),
        "promotiongap_cohort_attrition": rate_table(synth_df, "PromotionGapBand"),
        "funnel_limits": funnel_table(synth_df, ["Department"], p0=0.33),
    }

    two_ways = {"ot_dept": two_way_table(synth_df, "OverTime", "Department")}
    cands, scanned = hotspot_candidates(two_ways, overall_rate_pct=33.0)
    tables_dict["hotspot_candidates"] = cands

    summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_file": "synthetic_workforce.csv",
        "dataset_sha256": "0" * 64,
        "environment": {"python_version": "3.14.0", "pandas_version": "3.0.3", "numpy_version": "2.4.4"},
        "dimensions": {"rows": 100, "columns": len(synth_df.columns)},
        "overall_metrics": {
            "total_employees": 100,
            "leavers": 33,
            "retained": 67,
            "attrition_rate_pct": 33.0,
            "ci95_low_pct": 24.5,
            "ci95_high_pct": 42.8,
            "imbalance_ratio": 2.03,
        },
        "hotspot_candidates_scan": {
            "total_cells_scanned": scanned,
            "min_n_threshold": 30,
            "candidate_count": len(cands),
        },
    }

    report_text = build_phase3a_report(tables_dict, summary)

    # Assert synthetic numbers and names appear
    assert "100" in report_text
    assert "CyberSecurity" in report_text
    assert "CloudEngineering" in report_text
    assert "33.0%" in report_text

    # Assert real-dataset figures do NOT appear as hardcoded strings
    assert "1,470" not in report_text
    assert "16.12%" not in report_text
    assert "Research & Development" not in report_text
    assert "Sales Executive" not in report_text
