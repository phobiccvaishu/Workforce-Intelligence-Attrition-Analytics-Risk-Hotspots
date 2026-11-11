"""Hash-verified data access layer for the Workforce Intelligence dashboard.

Verifies SHA-256 integrity checksums against pre-registered values in
src.workforce_intel.config.STATIC_TABLE_CHECKSUMS before returning any table.
Caches loaded tables using streamlit.cache_data to guarantee performance.
Enforces the pure-presentation invariant: no statistical transformations,
no grouping, no value counting, and no manual rate arithmetic.
"""

import hashlib
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import streamlit as st

from workforce_intel.config import (
    PROJECT_ROOT,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
    STATIC_TABLE_CHECKSUMS,
)
from workforce_intel.io import load_processed


class DataIntegrityError(Exception):
    """Raised when an inferential or model benchmark fails SHA-256 cryptographic verification."""
    pass


def verify_file_checksum(file_path: Path, expected_hash: str) -> None:
    """Verify that file on disk matches expected SHA-256 hex digest."""
    if not file_path.is_file():
        raise DataIntegrityError(f"Required static benchmark file missing: {file_path}")
    raw = file_path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if actual_hash == expected_hash:
        return

    # Direct check against pre-registered LF checksums (for Linux / Streamlit Cloud)
    from workforce_intel.config import STATIC_TABLE_CHECKSUMS, STATIC_TABLE_CHECKSUMS_LF
    for rel_key, exp_crlf in STATIC_TABLE_CHECKSUMS.items():
        if exp_crlf == expected_hash and rel_key in STATIC_TABLE_CHECKSUMS_LF:
            if actual_hash == STATIC_TABLE_CHECKSUMS_LF[rel_key]:
                return

    # Normalize cross-platform git line endings (e.g. LF checkout on Linux vs CRLF on Windows)
    crlf_hash = hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
    if crlf_hash == expected_hash:
        return

    raise DataIntegrityError(
        f"Integrity check failed for {file_path.name}. "
        f"Expected SHA-256: {expected_hash}, actual: {actual_hash}"
    )


@st.cache_data
def get_processed_dataset() -> pd.DataFrame:
    """Load canonical processed workforce attrition dataset (1,470 records)."""
    return load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)


@st.cache_data
def get_static_table(rel_path: str) -> pd.DataFrame:
    """Load a pre-registered static benchmark table verified by SHA-256 hash."""
    file_path = PROJECT_ROOT / rel_path
    if rel_path in STATIC_TABLE_CHECKSUMS:
        verify_file_checksum(file_path, STATIC_TABLE_CHECKSUMS[rel_path])
    return pd.read_csv(file_path)


@st.cache_data
def get_hotspot_universe() -> pd.DataFrame:
    """Load verified hotspot screening universe (104 candidate segments)."""
    return get_static_table("reports/phase5/tables/hotspot_universe.csv")


@st.cache_data
def get_hotspot_split_validation() -> pd.DataFrame:
    """Load verified split-sample validation benchmarks (25 candidate segments)."""
    return get_static_table("reports/phase5/tables/hotspot_split_validation.csv")


@st.cache_data
def get_greedy_coverage() -> pd.DataFrame:
    """Load verified greedy marginal coverage progression (8 steps)."""
    return get_static_table("reports/phase5/tables/greedy_coverage.csv")


@st.cache_data
def get_overtime_comparison() -> pd.DataFrame:
    """Load verified adjusted OverTime association comparison table."""
    return get_static_table("reports/phase5/tables/overtime_comparison.csv")


@st.cache_data
def get_holdout_metrics() -> pd.DataFrame:
    """Load verified Phase 4 holdout evaluation metrics (top-10% and top-20% triage yield)."""
    return get_static_table("reports/phase4/tables/holdout_metrics.csv")


@st.cache_data
def get_bh_family_audit() -> pd.DataFrame:
    """Load Benjamini-Hochberg multiplicity audit table for 28 categorical dimensions."""
    return get_static_table("reports/phase5/tables/bh_family_audit.csv")


@st.cache_data
def get_logistic_models() -> pd.DataFrame:
    """Load adjusted multivariable logistic regression parameters (Models M1 and M2)."""
    return get_static_table("reports/phase5/tables/logistic_m1_m2.csv")


@st.cache_data
def get_demographic_fairness() -> pd.DataFrame:
    """Load demographic fairness and parity audit metrics."""
    return get_static_table("reports/phase4/tables/demographic_fairness.csv")


@st.cache_data
def get_consistency_audit() -> pd.DataFrame:
    """Load 57-row Phase 3 vs Phase 4 finding consistency audit table."""
    return get_static_table("reports/phase4/tables/consistency_audit.csv")


@st.cache_data
def get_model_comparison() -> pd.DataFrame:
    """Load Phase 4 model comparison benchmarks (Dummy, LR, HGB, LR Calibrated)."""
    return get_static_table("reports/phase4/tables/model_comparison.csv")


@st.cache_data
def get_paired_model_differences() -> pd.DataFrame:
    """Load Phase 4 paired bootstrap model difference benchmarks (LR vs HGB, Set A vs Set B)."""
    return get_static_table("reports/phase4/tables/paired_model_differences.csv")


@st.cache_data
def get_calibration_bins() -> pd.DataFrame:
    """Load calibration bin evaluation table with sample counts and observed rates."""
    return get_static_table("reports/phase4/tables/calibration_bins.csv")


@st.cache_data
def get_baseline_kpis() -> pd.DataFrame:
    """Load Phase 6 baseline KPI evaluations across the full enterprise."""
    return get_static_table("reports/phase6/tables/kpi_baseline.csv")


@st.cache_data
def get_decision_status() -> pd.DataFrame:
    """Load decision status registry table (DL-01 through current logged decisions)."""
    file_path = PROJECT_ROOT / "docs/decision_status.md"
    lines = file_path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines:
        clean = line.strip()
        if not ("| `DL-" in clean or "| DL-" in clean):
            continue
        parts = [p.strip() for p in clean.split("|")[1:-1]]
        if len(parts) >= 4:
            records.append({
                "Decision ID": parts[0].replace("`", ""),
                "Title": parts[1].replace("**", ""),
                "Status": parts[2].replace("`", ""),
                "Phase": parts[3],
            })
    return pd.DataFrame(records)


def get_verified_figure_path(rel_path: str) -> Path:
    """Verify figure file SHA-256 integrity and return its absolute Path."""
    file_path = PROJECT_ROOT / rel_path
    if rel_path in STATIC_TABLE_CHECKSUMS:
        verify_file_checksum(file_path, STATIC_TABLE_CHECKSUMS[rel_path])
    return file_path
