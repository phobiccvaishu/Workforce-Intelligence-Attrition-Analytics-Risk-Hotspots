"""Data contract, idempotency, and provenance tests for Phase 2 processed dataset.

Adheres strictly to Phase 2 Section 5.9 and user requirements:
- Real-data contract: processed originals == raw originals
- Column ordering and schema.json consistency
- Byte-identical determinism of generated processed CSV vs tracked file
- Section 8 reference checkpoints verified
- Provenance test on synthetic dataset
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from workforce_intel.cleaning import build_processed
from workforce_intel.config import (
    CONTINUOUS_COLUMNS,
    DERIVED_COLUMNS,
    DISCRETE_COUNT_COLUMNS,
    EXPECTED_COLUMNS,
    NOMINAL_COLUMNS,
    ORDINAL_COLUMNS_DOCUMENTED_RANGES,
    PROCESSED_COLUMNS,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
    RAW_DATA_FILE,
    SURROGATE_KEY_COLUMN,
    TARGET_COLUMN,
)
from workforce_intel.io import load_processed, load_raw
from workforce_intel.reporting import build_phase2_report
from workforce_intel.validation import reconciliation_audit


def test_processed_contract_real_dataset():
    """Verify that processed dataset exactly matches the 47-column contract."""
    raw_df, _ = load_raw(RAW_DATA_FILE)
    processed_df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)

    # 1. Dimensions
    assert len(processed_df) == 1470
    assert len(processed_df.columns) == 47
    assert list(processed_df.columns) == PROCESSED_COLUMNS

    # 2. RowID
    assert processed_df[SURROGATE_KEY_COLUMN].tolist() == list(range(1470))

    # 3. Cell-for-cell identity on 31 original columns
    for col in EXPECTED_COLUMNS:
        match = (raw_df[col] == processed_df[col]) | (
            raw_df[col].isna() & processed_df[col].isna()
        )
        assert match.all(), f"Mismatch detected in original column '{col}' between raw and processed"

    # 4. Zero denominator NaNs
    zero_tenure_count = int((raw_df["YearsAtCompany"] == 0).sum())
    assert zero_tenure_count == 44
    assert processed_df["RoleTenureShare"].isna().sum() == 44
    assert processed_df["PromotionGapShare"].isna().sum() == 44

    # 5. Non-null ranges for ratio shares
    valid_role_shares = processed_df.loc[processed_df["YearsAtCompany"] > 0, "RoleTenureShare"]
    assert valid_role_shares.min() >= 0.0 and valid_role_shares.max() <= 1.0

    valid_prom_shares = processed_df.loc[processed_df["YearsAtCompany"] > 0, "PromotionGapShare"]
    assert valid_prom_shares.min() >= 0.0 and valid_prom_shares.max() <= 1.0


def test_schema_json_metadata():
    """Verify that schema.json contains all required metadata fields and flags."""
    assert PROCESSED_SCHEMA_FILE.exists()
    with open(PROCESSED_SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)

    cols_meta = schema["columns"]
    assert len(cols_meta) == 47

    # Check protected attributes
    for col in ["Age", "Gender", "MaritalStatus"]:
        assert cols_meta[col]["protected_attribute"] is True, f"{col} must be flagged as protected attribute"

    # Check analytical use flags
    for col in ["DailyRate", "HourlyRate", "MonthlyRate"]:
        assert cols_meta[col]["analytical_use"] is False, f"{col} must have analytical_use=False"

    for col in ["MonthlyIncome", "TotalWorkingYears", "Attrition", "AgeGroup"]:
        assert cols_meta[col]["analytical_use"] is True, f"{col} must have analytical_use=True"

    # Check nullable flags
    assert cols_meta["RoleTenureShare"]["nullable"] is True
    assert cols_meta["PromotionGapShare"]["nullable"] is True
    assert cols_meta["AgeGroup"]["nullable"] is False


def test_processed_csv_byte_identical_reproducibility():
    """Verify that regenerating processed CSV from data/raw produces byte-identical hash."""
    raw_df, _ = load_raw(RAW_DATA_FILE)
    regenerated_df = build_processed(raw_df)

    float_cols = ["RoleTenureShare", "PromotionGapShare"]
    save_copy = regenerated_df.copy()
    for fc in float_cols:
        save_copy[fc] = save_copy[fc].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "")

    csv_bytes = save_copy.to_csv(index=False, encoding="utf-8", lineterminator="\n").encode("utf-8")
    regen_hash = hashlib.sha256(csv_bytes).hexdigest()

    disk_bytes = PROCESSED_DATA_FILE.read_bytes()
    disk_hash = hashlib.sha256(disk_bytes).hexdigest()

    assert regen_hash == disk_hash, (
        f"Regenerated processed CSV hash ({regen_hash}) != tracked file hash ({disk_hash})"
    )


def test_schema_roles_match_config_groups():
    """Assert that schema.json role matches config.py classification for all 31 original columns."""
    assert PROCESSED_SCHEMA_FILE.exists()
    with open(PROCESSED_SCHEMA_FILE, "r", encoding="utf-8") as f:
        schema = json.load(f)

    cols_meta = schema["columns"]

    for col in CONTINUOUS_COLUMNS:
        assert cols_meta[col]["role"] == "Continuous", f"{col} role must be 'Continuous' (got {cols_meta[col]['role']})"

    for col in DISCRETE_COUNT_COLUMNS:
        assert cols_meta[col]["role"] == "Discrete Count", f"{col} role must be 'Discrete Count' (got {cols_meta[col]['role']})"

    for col in NOMINAL_COLUMNS:
        assert cols_meta[col]["role"] == "Nominal", f"{col} role must be 'Nominal' (got {cols_meta[col]['role']})"

    for col in ORDINAL_COLUMNS_DOCUMENTED_RANGES:
        assert cols_meta[col]["role"] == "Ordinal", f"{col} role must be 'Ordinal' (got {cols_meta[col]['role']})"

    assert cols_meta[TARGET_COLUMN]["role"] == "Target"


def test_phase2_report_provenance_synthetic_dataset():
    """Verify build_phase2_report on synthetic data does NOT emit real-data numbers."""
    synthetic_raw = pd.DataFrame(
        {
            "Age": [22, 30, 40],
            "Attrition": [0, 1, 0],
            "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            "DailyRate": [100, 200, 300],
            "Department": ["Sales", "Sales", "Research & Development"],
            "DistanceFromHome": [2, 8, 15],
            "Education": [1, 2, 3],
            "EducationField": ["Medical", "Life Sciences", "Marketing"],
            "EnvironmentSatisfaction": [1, 2, 3],
            "Gender": ["Male", "Female", "Male"],
            "HourlyRate": [30, 40, 50],
            "JobInvolvement": [1, 2, 3],
            "JobLevel": [1, 2, 3],
            "JobRole": ["Sales Representative", "Sales Representative", "Research Scientist"],
            "JobSatisfaction": [1, 2, 3],
            "MaritalStatus": ["Single", "Married", "Divorced"],
            "MonthlyIncome": [2000, 4000, 6000],
            "MonthlyRate": [5000, 10000, 15000],
            "NumCompaniesWorked": [1, 0, 2],
            "OverTime": ["No", "Yes", "No"],
            "PercentSalaryHike": [11, 15, 20],
            "PerformanceRating": [3, 4, 3],
            "RelationshipSatisfaction": [1, 2, 3],
            "StockOptionLevel": [0, 1, 2],
            "TotalWorkingYears": [4, 6, 12],
            "TrainingTimesLastYear": [1, 2, 3],
            "WorkLifeBalance": [1, 2, 3],
            "YearsAtCompany": [2, 4, 8],
            "YearsInCurrentRole": [1, 2, 4],
            "YearsSinceLastPromotion": [0, 1, 2],
            "YearsWithCurrManager": [1, 2, 4],
        }
    )
    processed_synthetic = build_processed(synthetic_raw)
    recon = reconciliation_audit(synthetic_raw, processed_synthetic)

    summary_payload = {
        "execution_timestamp": "2026-11-02T00:00:00Z",
        "raw_sha256": "fake_raw_hash",
        "processed_sha256": "fake_proc_hash",
    }
    report = build_phase2_report(summary_payload, recon)

    assert "fake_raw_hash" in report
    forbidden_real_data_strings = [
        "1,470", "1470", "237", "1,233", "1233",
        "197", "581", "114", "16581", "16,581",
        "123-606-468-273", "215-255-306-448-246"
    ]
    for forbidden in forbidden_real_data_strings:
        assert forbidden not in report, (
            f"Hardcoded real-data string '{forbidden}' found in synthetic Phase 2 report!"
        )
