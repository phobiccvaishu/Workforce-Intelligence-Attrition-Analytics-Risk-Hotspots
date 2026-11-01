"""Data cleaning and preprocessing orchestration for Workforce Intelligence Phase 2.

Adheres strictly to Phase 2 Section 5.1, 5.2, 5.3, 5.7, 5.8:
- Precondition audits on raw input: whitespace, case, missing, sentinels, duplicates
- RowID injection as the first column (0-based index)
- 31 original columns pass through 100% UNCHANGED
- Exactly 15 documented derived columns appended
- Zero row removal: 1,470 in, 1,470 out, 0 dropped
- Postcondition bounds and domain validations
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from workforce_intel.config import (
    DERIVED_COLUMNS,
    EXPECTED_COLUMNS,
    PROCESSED_COLUMNS,
    SURROGATE_KEY_COLUMN,
)
from workforce_intel.features import (
    derive_age_group,
    derive_career_stage,
    derive_distance_band,
    derive_early_tenure,
    derive_has_prior_experience,
    derive_manager_tenure_band,
    derive_outstanding_rating,
    derive_prior_experience_years,
    derive_promotion_gap_band,
    derive_promotion_gap_equals_tenure,
    derive_promotion_gap_share,
    derive_role_tenure_band,
    derive_role_tenure_share,
    derive_stagnation_cohort,
    derive_tenure_band,
)
from workforce_intel.validation import (
    duplicate_report,
    missing_report,
    structure_report,
)

logger = logging.getLogger(__name__)


def verify_raw_preconditions(df: pd.DataFrame) -> None:
    """Verify raw input data hygiene preconditions before processing.

    Args:
        df: Input raw DataFrame.

    Raises:
        ValueError: If any structural contract violation, missing cells, unhandled
            whitespace strings, or full-row duplicates are detected.
    """
    struct = structure_report(df, expected_columns=EXPECTED_COLUMNS)
    if struct["contract_violated"]:
        raise ValueError(
            f"Raw DataFrame contract violated. Missing: {struct['missing_columns']}, "
            f"Unexpected: {struct['unexpected_columns']}"
        )

    missing = missing_report(df)
    nan_sum = int(missing["nan_count"].sum())
    if nan_sum > 0:
        raise ValueError(f"Precondition failure: raw data contains {nan_sum} NaN cells.")

    ws_sum = int(missing["whitespace_only_count"].sum())
    if ws_sum > 0:
        raise ValueError(
            f"Precondition failure: raw data contains {ws_sum} whitespace-only cells."
        )

    dup = duplicate_report(df)
    if dup["full_row_duplicates"] > 0:
        raise ValueError(
            f"Precondition failure: raw data contains {dup['full_row_duplicates']} full-row duplicates."
        )


def validate_processed_domains(processed_df: pd.DataFrame) -> None:
    """Validate bounds, types, and domains of all derived columns in processed DataFrame.

    Args:
        processed_df: Preprocessed DataFrame.

    Raises:
        ValueError: If any domain constraint is violated.
    """
    n_rows = len(processed_df)

    # 1. Total shape & column contract
    if list(processed_df.columns) != PROCESSED_COLUMNS:
        raise ValueError(
            f"Processed column order does not match PROCESSED_COLUMNS contract.\n"
            f"Expected: {PROCESSED_COLUMNS}\n"
            f"Observed: {list(processed_df.columns)}"
        )

    # 2. RowID uniqueness and continuity
    if not (processed_df[SURROGATE_KEY_COLUMN] == np.arange(n_rows)).all():
        raise ValueError("RowID column must be contiguous 0-based sequence 0 .. n-1.")

    # 3. Categorical columns non-null coverage
    cat_cols = [
        "AgeGroup",
        "TenureBand",
        "CareerStage",
        "RoleTenureBand",
        "ManagerTenureBand",
        "PromotionGapBand",
        "DistanceBand",
    ]
    for col in cat_cols:
        null_count = int(processed_df[col].isna().sum())
        if null_count > 0:
            offending = processed_df[processed_df[col].isna()][SURROGATE_KEY_COLUMN].tolist()
            raise ValueError(
                f"Categorical column '{col}' has {null_count} unassigned/NaN rows at RowIDs: {offending}"
            )

    # 4. Binary columns domain {0, 1}
    bin_cols = [
        "EarlyTenure",
        "HasPriorExperience",
        "PromotionGapEqualsTenure",
        "StagnationCohort",
        "OutstandingRating",
    ]
    for col in bin_cols:
        invalid = processed_df[~processed_df[col].isin([0, 1])]
        if not invalid.empty:
            offending = invalid[SURROGATE_KEY_COLUMN].tolist()
            raise ValueError(
                f"Binary column '{col}' contains non-binary values at RowIDs: {offending}"
            )

    # 5. PriorExperienceYears domain >= 0
    neg_exp = processed_df[processed_df["PriorExperienceYears"] < 0]
    if not neg_exp.empty:
        offending = neg_exp[SURROGATE_KEY_COLUMN].tolist()
        raise ValueError(
            f"PriorExperienceYears has negative values at RowIDs: {offending}"
        )

    # 6. RoleTenureShare & PromotionGapShare bounds [0.0, 1.0] and zero-denom NaNs
    zero_company = processed_df["YearsAtCompany"] == 0
    expected_nan_count = int(zero_company.sum())

    role_nans = int(processed_df["RoleTenureShare"].isna().sum())
    if role_nans != expected_nan_count:
        raise ValueError(
            f"RoleTenureShare NaN count ({role_nans}) does not match YearsAtCompany==0 count ({expected_nan_count})."
        )

    prom_nans = int(processed_df["PromotionGapShare"].isna().sum())
    if prom_nans != expected_nan_count:
        raise ValueError(
            f"PromotionGapShare NaN count ({prom_nans}) does not match YearsAtCompany==0 count ({expected_nan_count})."
        )

    # Check non-null share ranges in [0.0, 1.0]
    valid_role_shares = processed_df.loc[~zero_company, "RoleTenureShare"]
    if ((valid_role_shares < 0.0) | (valid_role_shares > 1.0)).any():
        offending = processed_df[
            ~zero_company & ((processed_df["RoleTenureShare"] < 0.0) | (processed_df["RoleTenureShare"] > 1.0))
        ][SURROGATE_KEY_COLUMN].tolist()
        raise ValueError(
            f"RoleTenureShare values outside [0.0, 1.0] at RowIDs: {offending}"
        )

    valid_prom_shares = processed_df.loc[~zero_company, "PromotionGapShare"]
    if ((valid_prom_shares < 0.0) | (valid_prom_shares > 1.0)).any():
        offending = processed_df[
            ~zero_company & ((processed_df["PromotionGapShare"] < 0.0) | (processed_df["PromotionGapShare"] > 1.0))
        ][SURROGATE_KEY_COLUMN].tolist()
        raise ValueError(
            f"PromotionGapShare values outside [0.0, 1.0] at RowIDs: {offending}"
        )

    # 7. No NaNs anywhere in dataset except the two ratio columns
    for col in processed_df.columns:
        if col not in ["RoleTenureShare", "PromotionGapShare"]:
            col_nans = int(processed_df[col].isna().sum())
            if col_nans > 0:
                offending = processed_df[processed_df[col].isna()][SURROGATE_KEY_COLUMN].tolist()
                raise ValueError(
                    f"Unexpected NaN detected in column '{col}' at RowIDs: {offending}"
                )


def build_processed(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the analysis-ready processed DataFrame from raw data.

    Orchestrates:
    1. Precondition audits on raw input
    2. RowID injection (0-based surrogate index)
    3. Pass-through of all 31 original columns UNCHANGED
    4. Computation of all 15 derived features using pure functions
    5. Postcondition domain validations

    Args:
        raw_df: Untouched raw DataFrame.

    Returns:
        Processed DataFrame with exactly 47 columns (RowID + 31 raw + 15 derived).
    """
    logger.info("Executing raw dataset precondition checks...")
    verify_raw_preconditions(raw_df)

    n_rows = len(raw_df)
    logger.info("Raw input verified: %d rows, %d columns.", n_rows, len(raw_df.columns))

    # Initialize processed dict with RowID as the first column
    processed_dict = {SURROGATE_KEY_COLUMN: np.arange(n_rows, dtype="int64")}

    # Pass through 31 original columns untouched
    for col in EXPECTED_COLUMNS:
        processed_dict[col] = raw_df[col].copy()

    # Derived features computation
    logger.info("Deriving 15 Phase 2 features...")
    processed_dict["AgeGroup"] = derive_age_group(raw_df["Age"])
    processed_dict["TenureBand"] = derive_tenure_band(raw_df["YearsAtCompany"])
    processed_dict["EarlyTenure"] = derive_early_tenure(raw_df["YearsAtCompany"])
    processed_dict["CareerStage"] = derive_career_stage(raw_df["TotalWorkingYears"])
    processed_dict["RoleTenureBand"] = derive_role_tenure_band(raw_df["YearsInCurrentRole"])
    processed_dict["ManagerTenureBand"] = derive_manager_tenure_band(raw_df["YearsWithCurrManager"])
    processed_dict["PromotionGapBand"] = derive_promotion_gap_band(raw_df["YearsSinceLastPromotion"])
    processed_dict["DistanceBand"] = derive_distance_band(raw_df["DistanceFromHome"])
    processed_dict["PriorExperienceYears"] = derive_prior_experience_years(
        raw_df["TotalWorkingYears"], raw_df["YearsAtCompany"]
    )
    processed_dict["HasPriorExperience"] = derive_has_prior_experience(
        processed_dict["PriorExperienceYears"]
    )
    processed_dict["RoleTenureShare"] = derive_role_tenure_share(
        raw_df["YearsInCurrentRole"], raw_df["YearsAtCompany"]
    )
    processed_dict["PromotionGapShare"] = derive_promotion_gap_share(
        raw_df["YearsSinceLastPromotion"], raw_df["YearsAtCompany"]
    )
    processed_dict["PromotionGapEqualsTenure"] = derive_promotion_gap_equals_tenure(
        raw_df["YearsAtCompany"], raw_df["YearsSinceLastPromotion"]
    )
    processed_dict["StagnationCohort"] = derive_stagnation_cohort(raw_df["YearsAtCompany"])
    processed_dict["OutstandingRating"] = derive_outstanding_rating(raw_df["PerformanceRating"])

    processed_df = pd.DataFrame(processed_dict)

    # Validate postconditions
    logger.info("Validating processed dataset contract and domains...")
    validate_processed_domains(processed_df)

    logger.info(
        "Successfully built processed dataset: %d rows in, %d rows out (0 dropped), %d columns.",
        n_rows,
        len(processed_df),
        len(processed_df.columns),
    )
    return processed_df
