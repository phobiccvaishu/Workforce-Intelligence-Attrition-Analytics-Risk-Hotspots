"""Tests for Workforce Intelligence dashboard UI components.

Tests advisory banners, provenance footers, reliability badge text mapping,
and aggregate download formatters.
"""

import pandas as pd
import pytest

from app.components import (
    render_advisory_banner,
    render_static_filter_notice,
    render_provenance_footer,
    format_kpi_card_display,
    RELIABILITY_BADGE_TEXT,
)


def test_reliability_badge_mapping():
    """Verify all formal DL-41 reliability states have standardized textual descriptions."""
    expected_states = [
        "reliable",
        "low_n",
        "no_data",
        "exposure_fixed_by_filter",
        "unreliable_zero_cell",
        "workforce-level, not filterable",
    ]
    for state in expected_states:
        assert state in RELIABILITY_BADGE_TEXT, f"Missing reliability description for state '{state}'"
        assert len(RELIABILITY_BADGE_TEXT[state]) > 0


def test_format_kpi_card_display_reliable():
    """Test KPI card formatting for reliable metrics."""
    kpi_res = {
        "kpi_id": "KPI-01",
        "name": "Attrition Rate",
        "value": 0.1612,
        "n": 1470,
        "ci_low": 0.1433,
        "ci_high": 0.1809,
        "reliability": "reliable",
    }
    display = format_kpi_card_display(kpi_res)
    assert display["title"] == "KPI-01: Attrition Rate"
    assert "16.1%" in display["value_str"]
    assert "n = 1,470" in display["n_str"]
    assert "[14.3%, 18.1%]" in display["ci_str"]
    assert "reliable" in display["badge_text"]


def test_format_kpi_card_display_low_n():
    """Test KPI card formatting for low_n metrics."""
    kpi_res = {
        "kpi_id": "KPI-01",
        "name": "Attrition Rate",
        "value": 0.2593,
        "n": 27,
        "ci_low": 0.1317,
        "ci_high": 0.4468,
        "reliability": "low_n",
    }
    display = format_kpi_card_display(kpi_res)
    assert display["badge_text"] == "low_n"
    assert "n = 27" in display["n_str"]


def test_format_kpi_card_display_empty_set():
    """Test KPI card formatting for empty subpopulation (n = 0)."""
    kpi_res = {
        "kpi_id": "KPI-01",
        "name": "Attrition Rate",
        "value": None,
        "n": 0,
        "ci_low": None,
        "ci_high": None,
        "reliability": "no_data",
    }
    display = format_kpi_card_display(kpi_res)
    assert display["value_str"] == "No data (n = 0)"
    assert display["badge_text"] == "no_data"
