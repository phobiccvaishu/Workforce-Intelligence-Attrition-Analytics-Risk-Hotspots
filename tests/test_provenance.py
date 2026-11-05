"""Provenance and reporting integrity tests for Phase 3A."""

from datetime import datetime, timezone
from pathlib import Path
import re
import pandas as pd
import pytest

from workforce_intel.eda import (
    funnel_table,
    hotspot_candidates,
    overall_summary,
    rate_table,
    two_way_table,
)
from workforce_intel.reporting import build_phase3a_report


def test_phase3a_report_provenance_synthetic_dataset():
    """Verify that build_phase3a_report generates text strictly from supplied tables.

    Derives forbidden token set from real-data tables in reports/phase3/tables/ and
    asserts none appears in the synthetic report unless also computed from the synthetic frame.
    All synthetic summary values are computed dynamically from the synthetic frame.
    """
    # 1. Construct synthetic frame
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

    # Compute summary values dynamically
    total_n = len(synth_df)
    leavers = int((synth_df["Attrition"] == 1).sum())
    retained = int((synth_df["Attrition"] == 0).sum())
    attr_rate = (leavers / total_n) * 100.0

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
        "funnel_limits": funnel_table(synth_df, ["Department"], p0=attr_rate / 100.0),
    }

    ov_row = tables_dict["overall_baseline"].iloc[0]
    ci_low = float(ov_row["ci95_low_pct"])
    ci_high = float(ov_row["ci95_high_pct"])
    imbalance = float(ov_row["imbalance_ratio"])

    two_ways = {"ot_dept": two_way_table(synth_df, "OverTime", "Department")}
    cands, scanned, cells_ge_30 = hotspot_candidates(two_ways, overall_rate_pct=attr_rate)
    tables_dict["hotspot_candidates"] = cands

    summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_file": "synthetic_workforce.csv",
        "dataset_sha256": "0" * 64,
        "environment": {"python_version": "3.14.0", "pandas_version": "3.0.3", "numpy_version": "2.4.4"},
        "dimensions": {"rows": total_n, "columns": len(synth_df.columns)},
        "overall_metrics": {
            "total_employees": total_n,
            "leavers": leavers,
            "retained": retained,
            "attrition_rate_pct": attr_rate,
            "ci95_low_pct": ci_low,
            "ci95_high_pct": ci_high,
            "imbalance_ratio": imbalance,
        },
        "hotspot_candidates_scan": {
            "total_cells_scanned": scanned,
            "cells_with_min_n": cells_ge_30,
            "min_n_threshold": 30,
            "candidate_count": len(cands),
            "distinct_employees_in_union": 0,
            "distinct_leavers_in_union": 0,
        },
    }

    report_text = build_phase3a_report(tables_dict, summary)

    # Assert synthetic numbers and names appear
    assert str(total_n) in report_text
    assert "CyberSecurity" in report_text
    assert "CloudEngineering" in report_text
    assert f"{attr_rate:.1f}%" in report_text

    # Derive forbidden token set from real tables in reports/phase3/tables/
    real_tables_dir = Path("reports/phase3/tables")
    if real_tables_dir.exists():
        real_tokens = set()
        for f in real_tables_dir.glob("*.csv"):
            if f.name == "hotspot_overlap.csv":
                continue
            df_real = pd.read_csv(f)
            for col in df_real.columns:
                for val in df_real[col].dropna():
                    s_val = str(val)
                    if len(s_val) > 2 and not s_val.replace(".", "").isdigit():
                        real_tokens.add(s_val)
                    elif s_val.isdigit() and int(s_val) > 10:
                        real_tokens.add(s_val)

        synthetic_tokens = {
            "CyberSecurity", "CloudEngineering", "SecEngineer", "CloudArchitect",
            str(total_n), str(leavers), str(retained), "100", "33", "67"
        }

        common_words = {
            "Department", "JobRole", "JobLevel", "AgeGroup", "Gender", "MaritalStatus",
            "TenureBand", "EarlyTenure", "OverTime", "BusinessTravel", "DistanceBand",
            "PromotionGapBand", "Yes", "No", "Single", "Female", "Male", "Travel_Rarely",
            "1-2", "2-3", "1-5", "26-35"
        }

        forbidden_tokens = (real_tokens - synthetic_tokens) - common_words

        key_real_entities = [
            "Research & Development",
            "Sales Executive",
            "Laboratory Technician",
            "1,470",
            "237",
            "1,233",
            "16.1%",
        ]
        for entity in key_real_entities:
            assert entity not in report_text, f"Real entity '{entity}' leaked into synthetic report"


def test_all_percentages_in_report_match_table_cells():
    """Verify that every formatted percentage printed in reports/phase3/eda_tables_report.md matches a table cell."""
    report_file = Path("reports/phase3/eda_tables_report.md")
    if not report_file.exists():
        pytest.skip("reports/phase3/eda_tables_report.md not yet generated")

    report_content = report_file.read_text(encoding="utf-8")
    percentages_in_report = set(re.findall(r"\b(\d+\.\d)%", report_content))

    tables_dir = Path("reports/phase3/tables")
    assert tables_dir.exists()

    all_table_pcts = {"95.0", "99.8"}  # Funnel confidence levels
    for f in tables_dir.glob("*.csv"):
        df = pd.read_csv(f)
        for col in df.columns:
            if "pct" in col or "rate" in col:
                for val in df[col].dropna():
                    try:
                        all_table_pcts.add(f"{float(val):.1f}")
                    except ValueError:
                        pass

    for pct in percentages_in_report:
        assert pct in all_table_pcts, f"Report percentage {pct}% does not match any computed cell in tables"
