"""Data contract tests on the preserved raw dataset.

Asserts structural constraints only (schema, ordering, dtypes, checksums, ordinal boundaries).
Does not assert statistical findings or distributions.
"""

import hashlib
from pathlib import Path
import re

import pandas as pd
from pandas.api import types as ptypes
import pytest

from workforce_intel.config import (
    CHECKSUMS_FILE,
    EXPECTED_COLUMNS,
    NOMINAL_COLUMNS,
    ORDINAL_COLUMNS_DOCUMENTED_RANGES,
    RAW_DATA_FILE,
    TARGET_COLUMN,
)
from workforce_intel.io import load_raw


@pytest.fixture(scope="module")
def raw_dataset() -> pd.DataFrame:
    """Load raw dataset via the package loader."""
    df, _ = load_raw(RAW_DATA_FILE)
    return df


def test_checksum_matches_recorded():
    """Verify that the SHA-256 hash of data/raw copy matches CHECKSUMS.txt."""
    assert RAW_DATA_FILE.exists(), f"Raw data file missing at {RAW_DATA_FILE}"
    assert CHECKSUMS_FILE.exists(), f"Checksums file missing at {CHECKSUMS_FILE}"

    # Compute actual SHA-256
    hasher = hashlib.sha256()
    with open(RAW_DATA_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest().lower()

    # Parse recorded hash from CHECKSUMS.txt
    content = CHECKSUMS_FILE.read_text(encoding="utf-8")
    match = re.search(r"sha256:\s*([a-fA-F0-9]{64})", content)
    assert match is not None, "SHA-256 entry not found in CHECKSUMS.txt"
    recorded_hash = match.group(1).lower()

    assert actual_hash == recorded_hash, (
        f"Hash mismatch: actual {actual_hash} vs recorded {recorded_hash}"
    )


def test_column_names_and_exact_order(raw_dataset):
    """Verify exactly 31 columns matching expected names and exact order."""
    actual_cols = list(raw_dataset.columns)
    assert len(actual_cols) == 31, f"Expected 31 columns, found {len(actual_cols)}"
    assert actual_cols == EXPECTED_COLUMNS, "Column order or names do not match contract."


def test_target_values_subset_binary(raw_dataset):
    """Verify Attrition target values are strictly in {0, 1}."""
    unique_vals = set(raw_dataset[TARGET_COLUMN].dropna().unique())
    assert unique_vals.issubset({0, 1}), f"Attrition contains invalid values: {unique_vals}"


def test_data_types_expected(raw_dataset):
    """Verify nominal columns are text/string and numeric columns are numeric types."""
    for col in NOMINAL_COLUMNS:
        assert ptypes.is_string_dtype(raw_dataset[col]) or raw_dataset[col].dtype == object, (
            f"Expected string type for nominal column '{col}', got {raw_dataset[col].dtype}"
        )

    for col in ORDINAL_COLUMNS_DOCUMENTED_RANGES:
        assert ptypes.is_numeric_dtype(raw_dataset[col]), (
            f"Expected numeric type for ordinal column '{col}', got {raw_dataset[col].dtype}"
        )


def test_ordinal_bounds(raw_dataset):
    """Verify ordinal values do not exceed documented min/max scales."""
    for col, (doc_min, doc_max) in ORDINAL_COLUMNS_DOCUMENTED_RANGES.items():
        obs_min = raw_dataset[col].min()
        obs_max = raw_dataset[col].max()
        assert obs_min >= doc_min, f"Column {col} min {obs_min} below documented min {doc_min}"
        assert obs_max <= doc_max, f"Column {col} max {obs_max} above documented max {doc_max}"
