"""I/O module for Workforce Intelligence.

Provides safe raw dataset loading and write-protection guards.
Adheres to Section 5.2 and 5.4:
- Explicit UTF-8 read handling BOM and CRLF
- No silent coercion or custom na_values tricks
- Automatic delimiter detection
- Strict refusal to write any files inside data/raw/
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from workforce_intel.config import (
    PROCESSED_COLUMNS,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


def detect_delimiter(file_path: Path) -> str:
    """Detect the delimiter of a text-based tabular data file using csv.Sniffer."""
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        if not sample:
            raise ValueError(f"File {file_path} is empty (0 bytes).")
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except Exception:
            # Fall back to inspecting the first line
            first_line = sample.splitlines()[0]
            for delim in [",", "\t", ";", "|"]:
                if delim in first_line:
                    return delim
            return ","


def load_raw(path: Union[str, Path]) -> Tuple[pd.DataFrame, str]:
    """Load a raw dataset without silent coercion.

    Args:
        path: Path to the raw CSV/tabular file.

    Returns:
        Tuple of (DataFrame untouched, detected delimiter string).

    Raises:
        FileNotFoundError: If the target file does not exist, with candidates searched.
        ValueError: If the file is empty, header-only with no records, or unparseable.
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        # Search for candidates in parent or common directories
        candidates = []
        search_dirs = [file_path.parent, Path.cwd(), Path.cwd() / "data" / "raw"]
        for sdir in search_dirs:
            if sdir.exists():
                candidates.extend(list(sdir.glob("*.csv")) + list(sdir.glob("*.xlsx")))
        unique_candidates = sorted(list(set(str(c) for c in candidates)))
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Candidate tabular files located: {unique_candidates}"
        )

    if not file_path.is_file():
        raise ValueError(f"Path '{file_path}' is not a regular file.")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Dataset at '{file_path}' is empty (0 bytes).")

    delimiter = detect_delimiter(file_path)

    try:
        # Explicit utf-8-sig handles both UTF-8 with BOM and standard UTF-8 seamlessly,
        # and CRLF line endings are automatically handled by the underlying engine.
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            encoding="utf-8-sig",
            engine="python",
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file at '{file_path}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset at '{file_path}' has no data rows (empty or header-only).")

    logger.info(
        "Successfully loaded dataset from %s (rows: %d, cols: %d, delimiter: %r)",
        file_path,
        len(df),
        len(df.columns),
        delimiter,
    )
    return df, delimiter


def assert_safe_write_path(path: Union[str, Path]) -> Path:
    """Ensure that the target path does not resolve inside data/raw/.

    Args:
        path: Target write path.

    Returns:
        Resolved target Path.

    Raises:
        PermissionError: If the write destination is within data/raw/.
    """
    target = Path(path).resolve()
    raw_dir = RAW_DATA_DIR.resolve()

    try:
        target.relative_to(raw_dir)
        is_inside_raw = True
    except ValueError:
        is_inside_raw = False

    if is_inside_raw:
        raise PermissionError(
            f"Write operation refused: '{target}' is inside protected raw directory '{raw_dir}'."
        )

    return target


def load_processed(
    csv_path: Optional[Union[str, Path]] = None,
    schema_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the Phase 2 processed dataset and restore strict dtypes from schema.json.

    Args:
        csv_path: Path to processed CSV file (defaults to PROCESSED_DATA_FILE).
        schema_path: Path to schema JSON file (defaults to PROCESSED_SCHEMA_FILE).

    Returns:
        DataFrame with exact column contract, types, and ordered categoricals.

    Raises:
        FileNotFoundError: If the processed CSV or schema file is missing.
        ValueError: If column contract or schemas do not match.
    """
    target_csv = Path(csv_path).resolve() if csv_path else PROCESSED_DATA_FILE.resolve()
    target_schema = Path(schema_path).resolve() if schema_path else PROCESSED_SCHEMA_FILE.resolve()

    if not target_csv.exists():
        raise FileNotFoundError(f"Processed dataset not found at '{target_csv}'.")

    if not target_schema.exists():
        raise FileNotFoundError(f"Processed schema not found at '{target_schema}'.")

    with open(target_schema, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    # Read CSV
    df = pd.read_csv(target_csv, encoding="utf-8")

    # Verify column contract
    if list(df.columns) != PROCESSED_COLUMNS:
        raise ValueError(
            f"Loaded processed dataset columns do not match contract.\n"
            f"Expected: {PROCESSED_COLUMNS}\n"
            f"Observed: {list(df.columns)}"
        )

    # Restore types and ordered categoricals from schema
    columns_meta = schema_data.get("columns", {})
    for col_name, meta in columns_meta.items():
        if col_name not in df.columns:
            continue
        dtype_str = meta.get("dtype")
        categories = meta.get("categories")
        is_ordered = meta.get("ordered", False)

        if categories is not None and is_ordered:
            cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
            df[col_name] = df[col_name].astype(cat_type)
        elif dtype_str == "int64":
            df[col_name] = df[col_name].astype("int64")
        elif dtype_str == "float64":
            df[col_name] = df[col_name].astype("float64")
        elif dtype_str in ["object", "str", "string"]:
            df[col_name] = df[col_name].astype(str)

    logger.info(
        "Successfully loaded processed dataset from %s (rows: %d, cols: %d)",
        target_csv,
        len(df),
        len(df.columns),
    )
    return df
