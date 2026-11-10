"""Tests for Workforce Intelligence dashboard data access layer.

Verifies cryptographic checksums for all pre-registered benchmark tables and figures,
ensures proper caching, and tests mutation detection when table contents or hashes are tampered with.
"""

from pathlib import Path
import pytest
import pandas as pd

from app.data_access import (
    DataIntegrityError,
    verify_file_checksum,
    get_processed_dataset,
    get_hotspot_universe,
    get_hotspot_split_validation,
    get_greedy_coverage,
    get_overtime_comparison,
    get_holdout_metrics,
    get_bh_family_audit,
    get_logistic_models,
    get_demographic_fairness,
    get_consistency_audit,
    get_model_comparison,
    get_calibration_bins,
    get_paired_model_differences,
    get_baseline_kpis,
    get_decision_status,
    get_verified_figure_path,
)
from workforce_intel.config import PROJECT_ROOT, STATIC_TABLE_CHECKSUMS


def test_processed_dataset_integrity():
    """Verify processed dataset loads exactly 1,470 records with expected schema."""
    df = get_processed_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1470
    assert "Attrition" in df.columns
    assert "OverTime" in df.columns


def test_static_table_checksums_match_disk():
    """Verify that every static file registered in STATIC_TABLE_CHECKSUMS matches its disk SHA-256."""
    for rel_path, expected_hash in STATIC_TABLE_CHECKSUMS.items():
        file_path = PROJECT_ROOT / rel_path
        assert file_path.is_file(), f"Registered file missing from disk: {rel_path}"
        # verify_file_checksum raises DataIntegrityError on mismatch
        verify_file_checksum(file_path, expected_hash)


def test_static_table_loaders_return_valid_dataframes():
    """Verify that static table loader functions return non-empty DataFrames."""
    assert len(get_hotspot_universe()) == 104
    assert len(get_hotspot_split_validation()) == 25
    assert len(get_greedy_coverage()) == 8
    assert len(get_overtime_comparison()) >= 4
    assert len(get_holdout_metrics()) >= 3
    assert len(get_bh_family_audit()) == 28
    assert len(get_logistic_models()) >= 10
    assert len(get_demographic_fairness()) >= 9
    assert len(get_consistency_audit()) == 57
    assert len(get_model_comparison()) >= 4
    assert len(get_calibration_bins()) >= 10
    assert len(get_paired_model_differences()) >= 5
    assert len(get_baseline_kpis()) == 10
    assert len(get_decision_status()) >= 41


def test_verified_figure_paths():
    """Verify that all Phase 4 figures can be verified and accessed."""
    fig_names = [
        "reports/phase4/figures/fig15_precision_recall_curves.png",
        "reports/phase4/figures/fig16_calibration_curves.png",
        "reports/phase4/figures/fig17_risk_tier_attrition.png",
        "reports/phase4/figures/fig18_feature_odds_ratios.png",
        "reports/phase4/figures/fig19_hotspot_validation.png",
        "reports/phase4/figures/fig20_demographic_fairness.png",
    ]
    for fig_path in fig_names:
        p = get_verified_figure_path(fig_path)
        assert p.is_file()


def test_mutation_checksum_mismatch_raises_error(tmp_path):
    """Mutation test: Tampering with file content causes DataIntegrityError."""
    fake_file = tmp_path / "tampered_table.csv"
    fake_file.write_text("col1,col2\n1,2", encoding="utf-8")
    wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"

    with pytest.raises(DataIntegrityError) as exc_info:
        verify_file_checksum(fake_file, wrong_hash)

    assert "Integrity check failed" in str(exc_info.value)


def test_reports_contain_no_unauthorized_ip_addresses():
    """Verify that reports/ contains no external or local network IP addresses."""
    import re
    ip_pattern = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    reports_dir = PROJECT_ROOT / "reports"
    violations = {}

    for file_path in reports_dir.rglob("*"):
        if file_path.is_file() and not file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            matches = ip_pattern.findall(content)
            unauthorized = [m for m in matches if m != "127.0.0.1"]
            if unauthorized:
                violations[str(file_path.relative_to(PROJECT_ROOT))] = unauthorized

    assert not violations, f"Unauthorized IP addresses detected under reports/: {violations}"
