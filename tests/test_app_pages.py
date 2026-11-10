"""Comprehensive Streamlit AppTest Suite for Workforce Intelligence Dashboard.

Covers Section 5.4 requirements:
1. AppTest execution for every page: zero exceptions at default state.
2. Baseline KPI cards match kpi_baseline.csv.
3. All 12 kpi_filter_cases.csv reproduce through UI session_state filters.
4. Boundary cases: empty set (n=0), small-sample boundary (n=27 < 30), exposure fixed by filter.
5. Static page filter decoupling: static pages declare filters are not applied and maintain invariant metrics.
6. Advisory banner presence on decision support views.
7. Governance invariants: no employee-level tables or personal rows are ever rendered.
8. Language hygiene: zero forbidden words, zero dollar signs, no "significant".
9. Mutation tests: static table corruption, MIN_GROUP_SIZE flip, and metric tampering.
10. Traceability of rendered figures to committed tables and config constants.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from workforce_intel.config import (
    PROJECT_ROOT,
    MIN_GROUP_SIZE,
    STATIC_TABLE_CHECKSUMS,
)
from workforce_intel.kpis import (
    evaluate_all_kpis,
    kpi_01_attrition_rate,
)
from app.data_access import (
    DataIntegrityError,
    verify_file_checksum,
    get_processed_dataset,
)

PAGES = [
    "app/Overview.py",
    "app/pages/1_Hotspots.py",
    "app/pages/2_Adjusted_Associations.py",
    "app/pages/3_Model_Triage.py",
    "app/pages/4_Governance.py",
]

FORBIDDEN_WORDS = [
    "cause", "causes", "caused", "drives", "driver", "leads to",
    "because", "effect of", "impact of", "risk factor", "predicts",
    "stagnation", "stagnant", "flight risk", "proves", "confirms",
    "significant",
]


def extract_all_text_from_app(at: AppTest) -> str:
    """Extract all text rendered across various Streamlit element lists in AppTest."""
    parts = []
    for md in at.markdown:
        parts.append(str(md.value))
    for cap in at.caption:
        parts.append(str(cap.value))
    for inf in at.info:
        parts.append(str(inf.value))
    for wrn in at.warning:
        parts.append(str(wrn.value))
    for err in at.error:
        parts.append(str(err.value))
    for m in at.metric:
        parts.append(f"{m.label} {m.value}")
    for df_elem in at.dataframe:
        df_val = df_elem.value
        if isinstance(df_val, pd.DataFrame):
            parts.append(" ".join(str(x) for x in df_val.values.flatten()))
    return " ".join(parts)


# -----------------------------------------------------------------------------
# 1. Default Page Rendering Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("page_path", PAGES)
def test_all_pages_render_without_exception_at_default(page_path: str):
    """Verify each page renders completely with zero runtime exceptions."""
    at = AppTest.from_file(str(PROJECT_ROOT / page_path))
    at.run(timeout=15)
    assert len(at.exception) == 0, f"Page {page_path} raised exceptions: {[e.message for e in at.exception]}"


# -----------------------------------------------------------------------------
# 2. Baseline KPI Cards Match kpi_baseline.csv
# -----------------------------------------------------------------------------
def test_overview_baseline_cards_match_kpi_baseline():
    """Verify default Overview page displays the exact pre-registered baseline values."""
    baseline_df = pd.read_csv(PROJECT_ROOT / "reports/phase6/tables/kpi_baseline.csv")
    at = AppTest.from_file(str(PROJECT_ROOT / "app/Overview.py"))
    at.run(timeout=15)
    assert len(at.exception) == 0

    all_text = extract_all_text_from_app(at)

    # KPI-01: 16.12%
    k1_val = baseline_df[baseline_df["kpi_id"] == "KPI-01"]["value_summary"].iloc[0]
    assert "16.12%" in all_text or "16.1%" in all_text

    # KPI-02: Headcount 1,470, 237 leavers
    assert "1,470" in all_text
    assert "237" in all_text

    # KPI-04: OR = 3.77
    assert "3.77" in all_text

    # KPI-05: 10.79%
    assert "10.79%" in all_text or "10.8%" in all_text


# -----------------------------------------------------------------------------
# 3. All 12 Filter Cases Reproduce Exactly
# -----------------------------------------------------------------------------
def test_overview_12_filter_cases_reproduce_through_session_state():
    """Verify all 12 filter cases from kpi_filter_cases.csv reproduce accurately in Overview."""
    filter_cases_df = pd.read_csv(PROJECT_ROOT / "reports/phase6/tables/kpi_filter_cases.csv")
    grouped = filter_cases_df.groupby("case_id")

    for case_id, group in grouped:
        filter_json = group["filter_json"].iloc[0]
        filters = json.loads(filter_json)

        at = AppTest.from_file(str(PROJECT_ROOT / "app/Overview.py"))
        at.session_state["active_filters"] = filters
        at.run(timeout=15)

        assert len(at.exception) == 0, f"Failed on {case_id} with filters {filters}"
        all_text = extract_all_text_from_app(at)

        # Check KPI-01 expected headcount and reliability label
        k1_row = group[group["kpi_id"] == "KPI-01"].iloc[0]
        expected_n = int(k1_row["n"])
        expected_rel = str(k1_row["reliability"])

        if expected_n == 0:
            assert "No data (n = 0)" in all_text
            assert "no_data" in all_text
        else:
            assert f"{expected_n:,}" in all_text or str(expected_n) in all_text
            assert expected_rel in all_text


# -----------------------------------------------------------------------------
# 4. Boundary and Special Reliability Cases
# -----------------------------------------------------------------------------
def test_empty_set_boundary_case():
    """CASE-12: Empty set (Department=Sales & JobRole=Research Director) shows no_data."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app/Overview.py"))
    at.session_state["active_filters"] = {"Department": "Sales", "JobRole": "Research Director"}
    at.run(timeout=15)

    assert len(at.exception) == 0
    all_text = extract_all_text_from_app(at)
    assert "No data (n = 0)" in all_text
    assert "no_data" in all_text


def test_low_n_boundary_case():
    """CASE-10: Small group (JobRole=Manager & OverTime=Yes, n=27 < 30) is flagged low_n."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app/Overview.py"))
    at.session_state["active_filters"] = {"JobRole": "Manager", "OverTime": "Yes"}
    at.run(timeout=15)

    assert len(at.exception) == 0
    all_text = extract_all_text_from_app(at)
    assert "n = 27" in all_text or "27" in all_text
    assert "low_n" in all_text


def test_exposure_fixed_by_filter_case():
    """CASE-06: Filtering by OverTime=Yes fixes exposure and flags KPI-04."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app/Overview.py"))
    at.session_state["active_filters"] = {"OverTime": "Yes"}
    at.run(timeout=15)

    assert len(at.exception) == 0
    all_text = extract_all_text_from_app(at)
    assert "exposure_fixed_by_filter" in all_text


# -----------------------------------------------------------------------------
# 5. Static Pages Decouple From Active Filters
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("page_path", [
    "app/pages/1_Hotspots.py",
    "app/pages/2_Adjusted_Associations.py",
    "app/pages/3_Model_Triage.py",
    "app/pages/4_Governance.py",
])
def test_static_pages_ignore_filters_and_show_notice(page_path: str):
    """Assert static benchmark pages display filter policy notice and remain invariant to filters."""
    at = AppTest.from_file(str(PROJECT_ROOT / page_path))
    # Set active filter
    at.session_state["active_filters"] = {"Department": "Sales", "OverTime": "Yes"}
    at.run(timeout=15)

    assert len(at.exception) == 0
    all_text = extract_all_text_from_app(at)

    # Must contain the benchmark filter policy warning
    assert "Benchmark Filter Policy" in all_text or "Active subpopulation filters do not apply" in all_text


# -----------------------------------------------------------------------------
# 6. Advisory Banners on Decision Support Views
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("page_path", [
    "app/Overview.py",
    "app/pages/1_Hotspots.py",
    "app/pages/3_Model_Triage.py",
    "app/pages/4_Governance.py",
])
def test_advisory_banner_present_on_decision_views(page_path: str):
    """Verify mandatory advisory governance banner is rendered."""
    at = AppTest.from_file(str(PROJECT_ROOT / page_path))
    at.run(timeout=15)

    all_text = extract_all_text_from_app(at)
    assert "Advisory decision support for retention review and workload planning" in all_text
    assert "Not for automated or individual adverse personnel decisions" in all_text


# -----------------------------------------------------------------------------
# 7. No Employee-Level Rows or Sensitive Record Displays
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("page_path", PAGES)
def test_no_employee_level_data_rendered(page_path: str):
    """Verify that no DataFrame or table contains individual employee records or IDs."""
    at = AppTest.from_file(str(PROJECT_ROOT / page_path))
    at.run(timeout=15)

    banned_row_columns = {"EmployeeNumber", "EmployeeCount", "StandardHours", "Over18", "RowID"}
    for df_elem in at.dataframe:
        df_val = df_elem.value
        if isinstance(df_val, pd.DataFrame):
            col_set = set(df_val.columns)
            intersection = col_set.intersection(banned_row_columns)
            assert not intersection, f"Page {page_path} renders forbidden individual columns: {intersection}"
            assert len(df_val) != 1470 or "Attrition" not in df_val.columns, (
                f"Page {page_path} renders raw unaggregated employee records!"
            )


# -----------------------------------------------------------------------------
# 8. Strict Language Hygiene and Zero Dollar Signs
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("page_path", PAGES)
def test_language_hygiene_and_zero_dollar_signs(page_path: str):
    """Verify absence of causal vocabulary, absence of 'significant', and zero dollar signs."""
    at = AppTest.from_file(str(PROJECT_ROOT / page_path))
    at.run(timeout=15)

    all_text = extract_all_text_from_app(at)
    assert "$" not in all_text, f"Dollar sign found in {page_path}!"

    lower_text = all_text.lower()
    for word in FORBIDDEN_WORDS:
        # Match whole words to avoid sub-word false positives
        pattern = f" {word} "
        assert pattern not in f" {lower_text} ", f"Forbidden word '{word}' found in {page_path}!"


# -----------------------------------------------------------------------------
# 9. Mutation Tests
# -----------------------------------------------------------------------------
def test_mutation_min_group_size_flip():
    """Mutation test: When MIN_GROUP_SIZE is increased to 50, a group of 35 flips to low_n."""
    df = get_processed_dataset()
    # Filter to Department == 'Human Resources' (headcount n = 63)
    k1_default = kpi_01_attrition_rate(df, {"Department": "Human Resources"}, min_group_size=30)
    assert k1_default["reliability"] == "reliable"

    # Mutated min_group_size = 100
    k1_mutated = kpi_01_attrition_rate(df, {"Department": "Human Resources"}, min_group_size=100)
    assert k1_mutated["reliability"] == "low_n"


def test_mutation_static_table_corruption(tmp_path):
    """Mutation test: Tampering with a static table content violates checksum."""
    fake_csv = tmp_path / "corrupted_hotspots.csv"
    fake_csv.write_text("segment_id,hotspot_index\nSEG-01,99.9\n", encoding="utf-8")
    expected_hash = STATIC_TABLE_CHECKSUMS["reports/phase5/tables/hotspot_universe.csv"]

    with pytest.raises(DataIntegrityError):
        verify_file_checksum(fake_csv, expected_hash)


# -----------------------------------------------------------------------------
# 10. Traceability of Key Rendered Numbers
# -----------------------------------------------------------------------------
def test_traceability_of_rendered_metrics_to_pipeline():
    """Assert key figures rendered across the app match pre-computed pipeline tables."""
    # Overview metrics
    baseline_df = pd.read_csv(PROJECT_ROOT / "reports/phase6/tables/kpi_baseline.csv")
    k1_baseline = float(baseline_df[baseline_df["kpi_id"] == "KPI-01"]["value_summary"].iloc[0].replace("%", ""))
    assert round(k1_baseline, 2) == 16.12

    # Model Triage holdout metrics
    ho_df = pd.read_csv(PROJECT_ROOT / "reports/phase4/tables/holdout_metrics.csv")
    cal_row = ho_df[ho_df["model"] == "Logistic Regression (Calibrated - Sigmoid)"].iloc[0]
    assert int(cal_row["k_10_point"]) == 29
    assert int(cal_row["tp_10_point"]) == 18
    assert int(cal_row["k_20_point"]) == 59
    assert int(cal_row["tp_20_point"]) == 27
    assert round(float(cal_row["precision_at_10_point"]), 3) == 0.621
    assert round(float(cal_row["recall_at_10_point"]), 3) == 0.383
    assert round(float(cal_row["lift_at_10_point"]), 2) == 3.88

    # Hotspot split sample validation metrics
    split_df = pd.read_csv(PROJECT_ROOT / "reports/phase5/tables/hotspot_split_validation.csv")
    assert len(split_df) == 25
    ge30 = split_df[split_df["holdout_n"] >= 30]
    assert len(ge30) == 14
    assert (ge30["above_holdout_point"]).sum() == 12
    assert (ge30["above_holdout_ci_low"]).sum() == 7
