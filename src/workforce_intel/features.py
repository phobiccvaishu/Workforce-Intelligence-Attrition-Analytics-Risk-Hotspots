"""Pure feature derivation functions for Workforce Intelligence Phase 2.

Adheres strictly to Phase 2 Section 5.4:
- Pure functions, one per derivation, without side effects
- Vectorized execution returning typed pandas Series
- Explicit cut-points and ordered categoricals defined in config.py
- Zero-denominator rule: ratio features yield NaN when YearsAtCompany == 0 (no warnings)
- Validation: out-of-range inputs raise ValueError explicitly listing offending RowIDs
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd

from workforce_intel.config import (
    AGE_BINS,
    AGE_LABELS,
    CAREER_STAGE_BINS,
    CAREER_STAGE_LABELS,
    DISTANCE_BINS,
    DISTANCE_LABELS,
    EARLY_TENURE_MAX_YEARS,
    MANAGER_TENURE_BINS,
    MANAGER_TENURE_LABELS,
    MIN_TENURE_FOR_PROMOTION_ANALYSIS,
    PROMOTION_GAP_BINS,
    PROMOTION_GAP_LABELS,
    ROLE_TENURE_BINS,
    ROLE_TENURE_LABELS,
    TENURE_BINS,
    TENURE_LABELS,
)


def derive_age_group(age: pd.Series) -> pd.Series:
    """Derive AgeGroup as an ordered categorical: '18-25', '26-35', '36-45', '46-60'.

    Args:
        age: Series of employee ages.

    Returns:
        Ordered Categorical Series.

    Raises:
        ValueError: If any age falls outside documented bounds [18, 60].
    """
    out_of_bounds = age[(age < 18) | (age > 60)]
    if not out_of_bounds.empty:
        offending = out_of_bounds.index.tolist()
        raise ValueError(
            f"Age out-of-range [18, 60] detected at RowIDs: {offending}"
        )

    return pd.cut(
        age,
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
        ordered=True,
    )


def derive_tenure_band(years_at_company: pd.Series) -> pd.Series:
    """Derive TenureBand as an ordered categorical: '0-1', '2-3', '4-5', '6-10', '11+'.

    Args:
        years_at_company: Series of tenure in company.

    Returns:
        Ordered Categorical Series.

    Raises:
        ValueError: If any tenure is negative or exceeds upper bound.
    """
    invalid = years_at_company[years_at_company < 0]
    if not invalid.empty:
        offending = invalid.index.tolist()
        raise ValueError(
            f"Negative YearsAtCompany detected at RowIDs: {offending}"
        )

    return pd.cut(
        years_at_company,
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
        ordered=True,
    )


def derive_early_tenure(
    years_at_company: pd.Series, max_years: int = EARLY_TENURE_MAX_YEARS
) -> pd.Series:
    """Derive EarlyTenure binary flag (1 = YearsAtCompany <= max_years, 0 otherwise).

    Args:
        years_at_company: Series of company tenure.
        max_years: Threshold tenure (default 1, meaning fewer than 2 completed years).

    Returns:
        int64 Series with values in {0, 1}.
    """
    invalid = years_at_company[years_at_company < 0]
    if not invalid.empty:
        offending = invalid.index.tolist()
        raise ValueError(
            f"Negative YearsAtCompany detected at RowIDs: {offending}"
        )

    return (years_at_company <= max_years).astype("int64")


def derive_career_stage(total_working_years: pd.Series) -> pd.Series:
    """Derive CareerStage as an ordered categorical: '0-5', '6-10', '11-20', '21+'.

    Args:
        total_working_years: Series of cumulative career experience.

    Returns:
        Ordered Categorical Series.

    Raises:
        ValueError: If any working years is negative.
    """
    invalid = total_working_years[total_working_years < 0]
    if not invalid.empty:
        offending = invalid.index.tolist()
        raise ValueError(
            f"Negative TotalWorkingYears detected at RowIDs: {offending}"
        )

    return pd.cut(
        total_working_years,
        bins=CAREER_STAGE_BINS,
        labels=CAREER_STAGE_LABELS,
        right=True,
        ordered=True,
    )


def derive_role_tenure_band(years_in_current_role: pd.Series) -> pd.Series:
    """Derive RoleTenureBand as an ordered categorical: '0-1', '2-3', '4-7', '8+'.

    Args:
        years_in_current_role: Series of tenure in current role.

    Returns:
        Ordered Categorical Series.
    """
    invalid = years_in_current_role[years_in_current_role < 0]
    if not invalid.empty:
        offending = invalid.index.tolist()
        raise ValueError(
            f"Negative YearsInCurrentRole detected at RowIDs: {offending}"
        )

    return pd.cut(
        years_in_current_role,
        bins=ROLE_TENURE_BINS,
        labels=ROLE_TENURE_LABELS,
        right=True,
        ordered=True,
    )


def derive_manager_tenure_band(years_with_curr_manager: pd.Series) -> pd.Series:
    """Derive ManagerTenureBand as an ordered categorical: '0-1', '2-3', '4-7', '8+'.

    Args:
        years_with_curr_manager: Series of tenure with current direct manager.

    Returns:
        Ordered Categorical Series.
    """
    invalid = years_with_curr_manager[years_with_curr_manager < 0]
    if not invalid.empty:
        offending = invalid.index.tolist()
        raise ValueError(
            f"Negative YearsWithCurrManager detected at RowIDs: {offending}"
        )

    return pd.cut(
        years_with_curr_manager,
        bins=MANAGER_TENURE_BINS,
        labels=MANAGER_TENURE_LABELS,
        right=True,
        ordered=True,
    )


def derive_promotion_gap_band(years_since_last_promotion: pd.Series) -> pd.Series:
    """Derive PromotionGapBand as an ordered categorical: '0', '1-2', '3-5', '6+'.

    Args:
        years_since_last_promotion: Series of years elapsed since last promotion.

    Returns:
        Ordered Categorical Series.
    """
    invalid = years_since_last_promotion[years_since_last_promotion < 0]
    if not invalid.empty:
        offending = invalid.index.tolist()
        raise ValueError(
            f"Negative YearsSinceLastPromotion detected at RowIDs: {offending}"
        )

    return pd.cut(
        years_since_last_promotion,
        bins=PROMOTION_GAP_BINS,
        labels=PROMOTION_GAP_LABELS,
        right=True,
        ordered=True,
    )


def derive_distance_band(distance_from_home: pd.Series) -> pd.Series:
    """Derive DistanceBand as an ordered categorical: '1-5', '6-10', '11-20', '21-29'.

    NOTE: Physical unit (miles vs km) is undocumented in dataset metadata.

    Args:
        distance_from_home: Series of commute distances.

    Returns:
        Ordered Categorical Series.

    Raises:
        ValueError: If distance falls outside documented range [1, 29].
    """
    out_of_bounds = distance_from_home[(distance_from_home < 1) | (distance_from_home > 29)]
    if not out_of_bounds.empty:
        offending = out_of_bounds.index.tolist()
        raise ValueError(
            f"DistanceFromHome out-of-range [1, 29] detected at RowIDs: {offending}"
        )

    return pd.cut(
        distance_from_home,
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        right=True,
        ordered=True,
    )


def derive_prior_experience_years(
    total_working_years: pd.Series, years_at_company: pd.Series
) -> pd.Series:
    """Derive PriorExperienceYears = TotalWorkingYears - YearsAtCompany.

    Args:
        total_working_years: Total career experience in years.
        years_at_company: Tenure at current organization in years.

    Returns:
        int64 Series representing pre-company professional experience.

    Raises:
        ValueError: If YearsAtCompany > TotalWorkingYears (chronological contradiction C1).
    """
    diff = total_working_years - years_at_company
    negative = diff[diff < 0]
    if not negative.empty:
        offending = negative.index.tolist()
        raise ValueError(
            f"Negative PriorExperienceYears (YearsAtCompany > TotalWorkingYears) at RowIDs: {offending}"
        )

    return diff.astype("int64")


def derive_has_prior_experience(prior_experience_years: pd.Series) -> pd.Series:
    """Derive HasPriorExperience binary flag (1 = PriorExperienceYears > 0, 0 otherwise).

    Derived purely from consistent tenure fields, NOT from ambiguous NumCompaniesWorked.

    Args:
        prior_experience_years: Pre-company tenure in years.

    Returns:
        int64 Series in {0, 1}.
    """
    return (prior_experience_years > 0).astype("int64")


def derive_role_tenure_share(
    years_in_current_role: pd.Series, years_at_company: pd.Series
) -> pd.Series:
    """Derive RoleTenureShare = YearsInCurrentRole / YearsAtCompany.

    ZERO-DENOMINATOR RULE:
    Evaluates strictly to NaN when YearsAtCompany == 0.
    Never imputes with 0, and avoids divide-by-zero runtime warnings.

    Args:
        years_in_current_role: Tenure in current role.
        years_at_company: Company tenure.

    Returns:
        float64 Series bounded in [0.0, 1.0], with NaN for zero company tenure.
    """
    # Safe division avoiding divide-by-zero warnings
    safe_denom = np.where(years_at_company == 0, 1.0, years_at_company.astype(float))
    raw_ratio = years_in_current_role.astype(float) / safe_denom
    result = np.where(years_at_company == 0, np.nan, raw_ratio)
    return pd.Series(result, index=years_at_company.index, dtype="float64")


def derive_promotion_gap_share(
    years_since_last_promotion: pd.Series, years_at_company: pd.Series
) -> pd.Series:
    """Derive PromotionGapShare = YearsSinceLastPromotion / YearsAtCompany.

    ZERO-DENOMINATOR RULE:
    Evaluates strictly to NaN when YearsAtCompany == 0.
    Never imputes with 0, and avoids divide-by-zero runtime warnings.

    Args:
        years_since_last_promotion: Years since most recent promotion.
        years_at_company: Company tenure.

    Returns:
        float64 Series bounded in [0.0, 1.0], with NaN for zero company tenure.
    """
    safe_denom = np.where(years_at_company == 0, 1.0, years_at_company.astype(float))
    raw_ratio = years_since_last_promotion.astype(float) / safe_denom
    result = np.where(years_at_company == 0, np.nan, raw_ratio)
    return pd.Series(result, index=years_at_company.index, dtype="float64")


def derive_promotion_gap_equals_tenure(
    years_at_company: pd.Series, years_since_last_promotion: pd.Series
) -> pd.Series:
    """Derive descriptive flag: YearsAtCompany > 0 AND YearsSinceLastPromotion == YearsAtCompany.

    Descriptive flag identifying employees whose tenure since last promotion matches
    their entire company tenure.

    Args:
        years_at_company: Company tenure.
        years_since_last_promotion: Years elapsed since last promotion.

    Returns:
        int64 Series in {0, 1}.
    """
    condition = (years_at_company > 0) & (years_since_last_promotion == years_at_company)
    return condition.astype("int64")


def derive_promotion_analysis_cohort(
    years_at_company: pd.Series,
    min_tenure: int = MIN_TENURE_FOR_PROMOTION_ANALYSIS,
) -> pd.Series:
    """Derive PromotionAnalysisCohort flag (1 = YearsAtCompany >= min_tenure, 0 otherwise).

    Establishes eligible cohort with sufficient tenure (>= 5 years) for promotion-gap analysis.

    Args:
        years_at_company: Company tenure.
        min_tenure: Minimum company tenure (default 5).

    Returns:
        int64 Series in {0, 1}.
    """
    return (years_at_company >= min_tenure).astype("int64")


# Backward compatibility alias


def derive_outstanding_rating(performance_rating: pd.Series) -> pd.Series:
    """Derive OutstandingRating flag (1 = PerformanceRating == 4, 0 otherwise).

    Reflects the observed restricted scale [3, 4] where 3='Excellent' and 4='Outstanding'.

    Args:
        performance_rating: Annual appraisal rating.

    Returns:
        int64 Series in {0, 1}.
    """
    return (performance_rating == 4).astype("int64")
