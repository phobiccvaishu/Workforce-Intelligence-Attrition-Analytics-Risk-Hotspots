"""Phase 2 Preprocessing Pipeline Runner.

One-command, idempotent, deterministic execution.
Executes raw dataset verification, builds processed dataset with 15 derived features,
saves schema.json, writes reports, tables, and documentation updates.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from workforce_intel.cleaning import build_processed
from workforce_intel.config import (
    AGE_LABELS,
    CAREER_STAGE_LABELS,
    CHECKSUMS_FILE,
    CONTINUOUS_COLUMNS,
    DERIVED_COLUMNS,
    DISCRETE_COUNT_COLUMNS,
    DISTANCE_LABELS,
    DOCS_DIR,
    EXPECTED_COLUMNS,
    MANAGER_TENURE_LABELS,
    MIN_TENURE_FOR_PROMOTION_ANALYSIS,
    NOMINAL_COLUMNS,
    NOT_USED_ANALYTICAL_COLUMNS,
    NULLABLE_DERIVED_COLUMNS,
    ORDINAL_COLUMNS_DOCUMENTED_RANGES,
    PHASE1_TABLES_DIR,
    PHASE2_LOG_FILE,
    PHASE2_REPORT_MD,
    PHASE2_REPORTS_DIR,
    PHASE2_SUMMARY_JSON,
    PHASE2_TABLES_DIR,
    PROCESSED_COLUMNS,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
    PROMOTION_GAP_LABELS,
    PROTECTED_ATTRIBUTE_COLUMNS,
    RAW_DATA_FILE,
    RAW_DATA_FILENAME,
    ROLE_TENURE_LABELS,
    SURROGATE_KEY_COLUMN,
    TARGET_COLUMN,
    TENURE_LABELS,
    get_environment_metadata,
)
from workforce_intel.io import assert_safe_write_path, load_raw
from workforce_intel.reporting import build_phase2_report
from workforce_intel.validation import reconciliation_audit


def setup_logger() -> logging.Logger:
    """Configure logger for Phase 2 preprocessing."""
    PHASE2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("workforce_intel.phase2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(PHASE2_LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_schema_json(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate the comprehensive machine-readable schema metadata per Section 5.5."""
    columns_meta = {}

    for col in PROCESSED_COLUMNS:
        series = df[col]
        dtype_str = str(series.dtype)
        is_derived = col in DERIVED_COLUMNS or col == SURROGATE_KEY_COLUMN
        source = "derived" if is_derived else "original"

        # Determine Role
        if col == SURROGATE_KEY_COLUMN:
            role = "Surrogate Identifier"
            derivation = "0-based integer sequence (0 .. n-1) matching Phase 1 audit row numbers"
            categories = None
            is_ordered = False
        elif col == TARGET_COLUMN:
            role = "Target"
            derivation = "Preserved original binary departure flag (1=left, 0=stayed)"
            categories = None
            is_ordered = False
        elif col in NOMINAL_COLUMNS:
            role = "Nominal"
            derivation = "Preserved from source dataset"
            categories = sorted(series.unique().tolist())
            is_ordered = False
        elif col in ORDINAL_COLUMNS_DOCUMENTED_RANGES:
            role = "Ordinal"
            derivation = "Preserved from source dataset (assumed scale)"
            categories = None
            is_ordered = False
        elif col in DERIVED_COLUMNS:
            if hasattr(series, "cat"):
                role = "Categorical Band"
                categories = list(series.cat.categories)
                is_ordered = True
            elif col in ["RoleTenureShare", "PromotionGapShare"]:
                role = "Ratio Feature"
                categories = None
                is_ordered = False
            else:
                role = "Binary Flag" if set(series.dropna().unique()).issubset({0, 1}) else "Discrete Count"
                categories = None
                is_ordered = False

            # Derivation text
            derivations_map = {
                "AgeGroup": "Right-inclusive cuts on Age: (17, 25], (25, 35], (35, 45], (45, 60]",
                "TenureBand": "Right-inclusive cuts on YearsAtCompany: [-1, 1], (1, 3], (3, 5], (5, 10], (10, 100]",
                "EarlyTenure": "Binary indicator: YearsAtCompany <= 1 (fewer than 2 completed years)",
                "CareerStage": "Right-inclusive cuts on TotalWorkingYears: [-1, 5], (5, 10], (10, 20], (20, 100]",
                "RoleTenureBand": "Right-inclusive cuts on YearsInCurrentRole: [-1, 1], (1, 3], (3, 7], (7, 100]",
                "ManagerTenureBand": "Right-inclusive cuts on YearsWithCurrManager: [-1, 1], (1, 3], (3, 7], (7, 100]",
                "PromotionGapBand": "Right-inclusive cuts on YearsSinceLastPromotion: [-1, 0], (0, 2], (2, 5], (5, 100]",
                "DistanceBand": "Right-inclusive cuts on DistanceFromHome: [0, 5], (5, 10], (10, 20], (20, 29] (unit undocumented)",
                "PriorExperienceYears": "TotalWorkingYears - YearsAtCompany (pre-company professional experience)",
                "HasPriorExperience": "Binary indicator: PriorExperienceYears > 0",
                "RoleTenureShare": "YearsInCurrentRole / YearsAtCompany (NaN when YearsAtCompany == 0; zero-denominator rule)",
                "PromotionGapShare": "YearsSinceLastPromotion / YearsAtCompany (NaN when YearsAtCompany == 0; zero-denominator rule)",
                "PromotionGapEqualsTenure": "Descriptive flag: YearsAtCompany > 0 & YearsSinceLastPromotion == YearsAtCompany",
                "PromotionAnalysisCohort": "Cohort flag: YearsAtCompany >= 5 (minimum tenure for promotion analysis)",
                "OutstandingRating": "Binary indicator: PerformanceRating == 4 ('Outstanding')",
            }
            derivation = derivations_map.get(col, "Phase 2 derived feature")
        elif col in CONTINUOUS_COLUMNS:
            role = "Continuous"
            derivation = "Preserved from source dataset"
            categories = None
            is_ordered = False
        elif col in DISCRETE_COUNT_COLUMNS:
            role = "Discrete Count"
            derivation = "Preserved from source dataset"
            categories = None
            is_ordered = False
        else:
            role = "Discrete Count" if dtype_str.startswith("int") else "Continuous"
            derivation = "Preserved from source dataset"
            categories = None
            is_ordered = False

        analytical_use = col not in NOT_USED_ANALYTICAL_COLUMNS
        protected_attr = col in PROTECTED_ATTRIBUTE_COLUMNS
        nullable = col in NULLABLE_DERIVED_COLUMNS

        columns_meta[col] = {
            "name": col,
            "dtype": dtype_str,
            "source": source,
            "role": role,
            "derivation": derivation,
            "categories": categories,
            "ordered": is_ordered,
            "analytical_use": analytical_use,
            "protected_attribute": protected_attr,
            "nullable": nullable,
        }

    return {
        "dataset_name": "workforce_attrition_processed.csv",
        "total_columns": len(PROCESSED_COLUMNS),
        "total_rows": len(df),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "columns": columns_meta,
    }


def update_data_dictionary_phase2(schema_data: Dict[str, Any]) -> None:
    """Append Section 4 Derived Features Specification to docs/data_dictionary.md."""
    dict_path = DOCS_DIR / "data_dictionary.md"
    content = dict_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    section_header = "## 4. Phase 2 Derived Features Specification"
    if section_header in content:
        content = content[: content.find(section_header)].strip() + "\n\n"

    rows = [
        "| Field | Role | Type | Categories / Bounds | Derivation Formula / Rule | Nullable | Analytical Use |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for col in [SURROGATE_KEY_COLUMN] + DERIVED_COLUMNS:
        meta = schema_data["columns"][col]
        cats = meta["categories"]
        if cats:
            domain = f"{len(cats)} levels: {', '.join(cats)}"
        elif meta["nullable"]:
            domain = "[0.0 .. 1.0] (44 NaNs)"
        elif meta["role"] == "Binary Flag":
            domain = "{0, 1}"
        else:
            domain = "Integer Range"
        
        rows.append(
            f"| `{meta['name']}` | {meta['role']} | `{meta['dtype']}` | {domain} | {meta['derivation']} | `{meta['nullable']}` | `{meta['analytical_use']}` |"
        )

    derived_section = f"""{section_header}

The following 15 features (plus surrogate index `RowID`) were derived in Phase 2 in strict accordance with the Section 5.4 contract. No rows were dropped, original values were untouched, and all cut-points represent documented design proposals.

{chr(10).join(rows)}
"""
    content = content + "\n" + derived_section
    assert_safe_write_path(dict_path)
    dict_path.write_text(content, encoding="utf-8")


def update_decision_log_phase2(reconciliation: Dict[str, Any]) -> None:
    """Append DL-10 through DL-26 to docs/decision_log.md."""
    dl_path = DOCS_DIR / "decision_log.md"
    content = dl_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    marker = "| `DL-10` |"
    if marker in content:
        content = content[: content.find(marker)].strip() + "\n"

    bucket_df = reconciliation["bucket_sizes_table"].set_index(["column", "level"])
    zero_df = pd.read_csv(PHASE1_TABLES_DIR / "zero_value_audit.csv").set_index("column")
    feat_df = reconciliation["derived_feature_audit"].set_index("derived_column")

    # Computed counts for evidence
    age_counts = "-".join(str(bucket_df.loc[("AgeGroup", l), "count"]) for l in AGE_LABELS)
    tenure_counts = "-".join(str(bucket_df.loc[("TenureBand", l), "count"]) for l in TENURE_LABELS)
    career_counts = "-".join(str(bucket_df.loc[("CareerStage", l), "count"]) for l in CAREER_STAGE_LABELS)
    role_counts = "-".join(str(bucket_df.loc[("RoleTenureBand", l), "count"]) for l in ROLE_TENURE_LABELS)
    mgr_counts = "-".join(str(bucket_df.loc[("ManagerTenureBand", l), "count"]) for l in MANAGER_TENURE_LABELS)
    prom_counts = "-".join(str(bucket_df.loc[("PromotionGapBand", l), "count"]) for l in PROMOTION_GAP_LABELS)
    dist_counts = "-".join(str(bucket_df.loc[("DistanceBand", l), "count"]) for l in DISTANCE_LABELS)

    early_tenure_1 = int(bucket_df.loc[("EarlyTenure", "1"), "count"])
    early_tenure_pct = f"{bucket_df.loc[('EarlyTenure', '1'), 'percentage']:.2f}"

    prior_zeros = int(bucket_df.loc[("HasPriorExperience", "0"), "count"])
    prior_exp_1 = int(bucket_df.loc[("HasPriorExperience", "1"), "count"])
    prior_exp_pct = f"{bucket_df.loc[('HasPriorExperience', '1'), 'percentage']:.2f}"

    prom_eq_1 = int(bucket_df.loc[("PromotionGapEqualsTenure", "1"), "count"])
    prom_eq_pct = f"{bucket_df.loc[('PromotionGapEqualsTenure', '1'), 'percentage']:.2f}"

    cohort_1 = int(bucket_df.loc[("PromotionAnalysisCohort", "1"), "count"])
    cohort_pct = f"{bucket_df.loc[('PromotionAnalysisCohort', '1'), 'percentage']:.2f}"

    outstanding_1 = int(bucket_df.loc[("OutstandingRating", "1"), "count"])
    outstanding_pct = f"{bucket_df.loc[('OutstandingRating', '1'), 'percentage']:.2f}"

    role_zero_count = int(zero_df.loc["YearsInCurrentRole", "zero_count"])
    nan_table = reconciliation["nan_audit_table"].set_index("column")
    nan_count = int(nan_table.loc["RoleTenureShare", "nan_count"])

    new_decisions = [
        {
            "id": "DL-10",
            "issue": "Surrogate RowID Column Injection",
            "evidence": "Dataset lacks employee ID. 1,470 continuous surrogate integers [0 .. 1469] injected matching Phase 1 table row indices. Table: reports/phase2/tables/reconciliation.csv",
            "proposed_handling": "Add RowID as first column. Document that CSV line number equals RowID + 2.",
            "alternatives": "Use default pandas index; omit surrogate identifier.",
            "status": "APPROVED",
            "phase": "Phase 2",
        },
        {
            "id": "DL-11",
            "issue": "Raw Columns Pass-Through Integrity",
            "evidence": "All 31 original raw columns preserved byte-for-byte with 0 missing and 0 unexpected columns across all 1,470 records. Table: reports/phase2/tables/reconciliation.csv",
            "proposed_handling": "Pass through all 31 original columns untouched without imputation, clipping, or transformations.",
            "alternatives": "Drop uninformative or redundant raw columns in preprocessing.",
            "status": "APPROVED",
            "phase": "Phase 2",
        },
        {
            "id": "DL-12",
            "issue": "AgeGroup Right-Inclusive Cut Bins",
            "evidence": f"Right-inclusive bins (17, 25], (25, 35], (35, 45], (45, 60] yield distribution {age_counts} across all 1,470 records. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Establish standard 4-tier demographic age bands. State that Phase 3 must evaluate sensitivity to alternate cut-points.",
            "alternatives": "Decade bins (20s, 30s, etc.); continuous age only.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-13",
            "issue": "TenureBand Right-Inclusive Cut Bins",
            "evidence": f"Right-inclusive bins [-1, 1], (1, 3], (3, 5], (5, 10], (10, 100] yield distribution {tenure_counts}. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Adopt standard 5-band enterprise tenure progression. State that Phase 3 must evaluate sensitivity to alternate cut-points.",
            "alternatives": "Equal-frequency quantiles; 3-tier band.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-14",
            "issue": "EarlyTenure Indicator Definition",
            "evidence": f"YearsAtCompany <= 1 (fewer than 2 completed years) captures exactly {early_tenure_1} employees ({early_tenure_pct}% of enterprise). Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Define EarlyTenure as YearsAtCompany <= 1 (fewer than 2 completed years) to isolate onboarding/retention risk.",
            "alternatives": "Tenure < 1 year (excludes 1-year employees); Tenure <= 2 years.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-15",
            "issue": "CareerStage Experience Bands",
            "evidence": f"TotalWorkingYears right-inclusive bins yield distribution {career_counts}. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Categorize career experience into 0-5, 6-10, 11-20, 21+ years. State that Phase 3 must evaluate sensitivity.",
            "alternatives": "Continuous years only; 3-stage progression (Early, Mid, Late).",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-16",
            "issue": "RoleTenureBand Right-Inclusive Bins",
            "evidence": f"YearsInCurrentRole right-inclusive cuts yield distribution {role_counts}. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Adopt 0-1, 2-3, 4-7, 8+ role tenure bands. State that Phase 3 must evaluate sensitivity.",
            "alternatives": "Continuous role tenure only.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-17",
            "issue": "ManagerTenureBand Right-Inclusive Bins",
            "evidence": f"YearsWithCurrManager right-inclusive cuts yield distribution {mgr_counts}. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Adopt 0-1, 2-3, 4-7, 8+ manager tenure bands. State that Phase 3 must evaluate sensitivity.",
            "alternatives": "Continuous manager tenure only.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-18",
            "issue": "PromotionGapBand Right-Inclusive Bins",
            "evidence": f"YearsSinceLastPromotion cuts yield distribution {prom_counts}. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Adopt 0, 1-2, 3-5, 6+ bands to capture promotion velocity. State that Phase 3 must evaluate sensitivity.",
            "alternatives": "Binary promoted-recently flag; continuous gap years.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-19",
            "issue": "DistanceBand Right-Inclusive Bins",
            "evidence": f"DistanceFromHome right-inclusive cuts yield distribution {dist_counts}. Physical distance unit is undocumented. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Adopt 1-5, 6-10, 11-20, 21-29 bands; document unit ambiguity in schema and reports.",
            "alternatives": "Continuous distance; binary close vs far indicator.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-20",
            "issue": "PriorExperienceYears Derivation",
            "evidence": f"Derived as TotalWorkingYears - YearsAtCompany. Min 0, max 33, exactly {prior_zeros} zero values. Table: reports/phase2/tables/derived_feature_audit.csv",
            "proposed_handling": "Derive PriorExperienceYears directly from consistent tenure fields to circumvent NumCompaniesWorked ambiguity.",
            "alternatives": "Use NumCompaniesWorked directly.",
            "status": "APPROVED",
            "phase": "Phase 2",
        },
        {
            "id": "DL-21",
            "issue": "HasPriorExperience Binary Flag",
            "evidence": f"PriorExperienceYears > 0 flags {prior_exp_1} employees ({prior_exp_pct}%) with pre-company experience. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Create clean binary indicator from PriorExperienceYears.",
            "alternatives": "NumCompaniesWorked > 0 indicator.",
            "status": "APPROVED",
            "phase": "Phase 2",
        },
        {
            "id": "DL-22",
            "issue": "Zero-Denominator Rule for Tenure Shares",
            "evidence": f"{nan_count} employees have YearsAtCompany == 0. RoleTenureShare and PromotionGapShare evaluate strictly to NaN for these {nan_count} records. Table: reports/phase2/tables/nan_audit.csv",
            "proposed_handling": "Leave zero-tenure rows as explicit NaN in ratio features. Never impute with 0 or mean in preprocessing.",
            "alternatives": "Impute 0.0 or 1.0; add epsilon offset (YearsAtCompany + 1).",
            "status": "APPROVED",
            "phase": "Phase 2",
        },
        {
            "id": "DL-23",
            "issue": "PromotionGapEqualsTenure: descriptive flag (YearsAtCompany > 0 and YearsSinceLastPromotion == YearsAtCompany)",
            "evidence": f"YearsAtCompany > 0 & YearsSinceLastPromotion == YearsAtCompany matches {prom_eq_1} employees ({prom_eq_pct}%). Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Provide descriptive flag only; captures cohort where time since last promotion equals entire organizational tenure.",
            "alternatives": "Compute continuous tenure ratio; omit indicator.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-24",
            "issue": "Promotion-Analysis Cohort (YearsAtCompany >= MIN_TENURE_FOR_PROMOTION_ANALYSIS)",
            "evidence": f"YearsAtCompany >= {MIN_TENURE_FOR_PROMOTION_ANALYSIS} identifies {cohort_1} employees ({cohort_pct}%) in eligible cohort for promotion analysis. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": f"Require minimum {MIN_TENURE_FOR_PROMOTION_ANALYSIS} years company tenure before evaluating career progression velocity in subgroup analytics.",
            "alternatives": "Tenure >= 3 years; tenure >= 7 years.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-25",
            "issue": "OutstandingRating Binary Flag",
            "evidence": f"PerformanceRating == 4 isolates exactly {outstanding_1} individuals ({outstanding_pct}%) at top performance tier. Table: reports/phase2/tables/bucket_sizes.csv",
            "proposed_handling": "Treat top rating as binary flag isolating 'Outstanding' performers given lack of 1-2 ratings.",
            "alternatives": "Preserve 3/4 integer scale.",
            "status": "APPROVED",
            "phase": "Phase 2 / Phase 3",
        },
        {
            "id": "DL-26",
            "issue": "YearsSinceLastPromotion / YearsInCurrentRole Ratio (DEFERRED)",
            "evidence": f"Ratio feature YearsSinceLastPromotion / YearsInCurrentRole is undefined when YearsInCurrentRole == 0 ({role_zero_count} employees). Table: reports/phase1/tables/zero_value_audit.csv",
            "proposed_handling": "Deferred: ratio not implemented in Phase 2 because denominator can be zero and ratio is unbounded without established capping rules.",
            "alternatives": "Add epsilon offset to denominator; impute zero; segment by role tenure.",
            "status": "DEFERRED",
            "phase": "Phase 3 (Exploratory Data Analysis & Modeling)",
        },
    ]

    new_rows = []
    for d in new_decisions:
        new_rows.append(
            f"| `{d['id']}` | **{d['issue']}** | {d['evidence']} | {d['proposed_handling']} | {d['alternatives']} | `{d['status']}` | {d['phase']} |"
        )

    content = content.rstrip() + "\n" + "\n".join(new_rows) + "\n"
    assert_safe_write_path(dl_path)
    dl_path.write_text(content, encoding="utf-8")


def run_pipeline() -> None:
    """Execute complete Phase 2 preprocessing and validation pipeline."""
    logger = setup_logger()
    logger.info("Starting Phase 2 Data Cleaning & Preprocessing Pipeline...")

    # Verify raw checksum
    hasher = hashlib.sha256()
    with open(RAW_DATA_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    raw_hash = hasher.hexdigest().lower()

    with open(CHECKSUMS_FILE, "r", encoding="utf-8") as f:
        checksum_content = f.read()
    if raw_hash not in checksum_content:
        raise ValueError("Raw dataset SHA-256 does not match CHECKSUMS.txt!")

    # Load raw
    raw_df, delimiter = load_raw(RAW_DATA_FILE)

    # Build processed
    processed_df = build_processed(raw_df)

    # Save processed CSV
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    assert_safe_write_path(PROCESSED_DATA_FILE)

    # Format float precision to 4 decimals for determinism
    float_cols = ["RoleTenureShare", "PromotionGapShare"]
    processed_to_save = processed_df.copy()
    for fc in float_cols:
        processed_to_save[fc] = processed_to_save[fc].apply(
            lambda v: f"{v:.4f}" if pd.notna(v) else ""
        )

    processed_to_save.to_csv(
        PROCESSED_DATA_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    logger.info("Saved processed dataset to %s", PROCESSED_DATA_FILE)

    # Compute processed CSV hash
    proc_hasher = hashlib.sha256()
    with open(PROCESSED_DATA_FILE, "rb") as f:
        while chunk := f.read(65536):
            proc_hasher.update(chunk)
    proc_hash = proc_hasher.hexdigest().lower()

    # Build & save schema.json
    schema_payload = build_schema_json(processed_df)
    schema_payload["processed_sha256"] = proc_hash
    schema_payload["raw_sha256"] = raw_hash

    assert_safe_write_path(PROCESSED_SCHEMA_FILE)
    with open(PROCESSED_SCHEMA_FILE, "w", encoding="utf-8") as f:
        json.dump(schema_payload, f, indent=2)
    logger.info("Saved schema metadata to %s", PROCESSED_SCHEMA_FILE)

    # Reconciliation audit
    reconciliation = reconciliation_audit(raw_df, processed_df)

    PHASE2_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    reconciliation["reconciliation_table"].to_csv(
        assert_safe_write_path(PHASE2_TABLES_DIR / "reconciliation.csv"), index=False
    )
    reconciliation["bucket_sizes_table"].to_csv(
        assert_safe_write_path(PHASE2_TABLES_DIR / "bucket_sizes.csv"), index=False
    )
    reconciliation["derived_feature_audit"].to_csv(
        assert_safe_write_path(PHASE2_TABLES_DIR / "derived_feature_audit.csv"), index=False
    )
    reconciliation["nan_audit_table"].to_csv(
        assert_safe_write_path(PHASE2_TABLES_DIR / "nan_audit.csv"), index=False
    )
    logger.info("Saved reconciliation CSV tables to %s", PHASE2_TABLES_DIR)

    # Summary JSON payload
    summary_payload = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_file": RAW_DATA_FILENAME,
        "raw_sha256": raw_hash,
        "processed_file": PROCESSED_DATA_FILE.name,
        "processed_sha256": proc_hash,
        "environment": get_environment_metadata(),
        "dimensions": {
            "raw_rows": len(raw_df),
            "processed_rows": len(processed_df),
            "rows_dropped": 0,
            "total_columns": len(processed_df.columns),
            "original_columns": len(EXPECTED_COLUMNS),
            "derived_columns": len(DERIVED_COLUMNS),
        },
        "reconciliation": {
            "original_columns_match": bool(reconciliation["original_cols_identical"]),
            "row_id_valid": bool(reconciliation["row_id_valid"]),
            "role_tenure_share_nans": int(processed_df["RoleTenureShare"].isna().sum()),
            "promotion_gap_share_nans": int(processed_df["PromotionGapShare"].isna().sum()),
        },
    }

    assert_safe_write_path(PHASE2_SUMMARY_JSON)
    with open(PHASE2_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    logger.info("Saved summary JSON to %s", PHASE2_SUMMARY_JSON)

    # Generate cleaning_report.md
    report_md = build_phase2_report(summary_payload, reconciliation)
    assert_safe_write_path(PHASE2_REPORT_MD)
    with open(PHASE2_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Saved cleaning report markdown to %s", PHASE2_REPORT_MD)

    # Update documentation
    update_data_dictionary_phase2(schema_payload)
    update_decision_log_phase2(reconciliation)
    logger.info("Updated docs/data_dictionary.md and docs/decision_log.md")
    logger.info("Phase 2 preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
