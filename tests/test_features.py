"""Unit tests for Phase 2 pure feature derivation functions.

Adheres strictly to Phase 2 Section 5.4, 5.9, and 7:
- Boundary tests for all cut points
- Invalid input raises ValueError explicitly listing offending RowIDs
- Zero-denominator rule: YearsAtCompany == 0 yields NaN without runtime warnings
- Pure vectorized execution returning typed Series
"""

import warnings
import numpy as np
import pandas as pd
import pytest

from workforce_intel.features import (
    derive_age_group,
    derive_career_stage,
    derive_distance_band,
    derive_early_tenure,
    derive_has_prior_experience,
    derive_manager_tenure_band,
    derive_outstanding_rating,
    derive_prior_experience_years,
    derive_promotion_analysis_cohort,
    derive_promotion_gap_band,
    derive_promotion_gap_equals_tenure,
    derive_promotion_gap_share,
    derive_role_tenure_band,
    derive_role_tenure_share,
    derive_tenure_band,
)


def test_age_group_boundaries():
    """Verify AgeGroup boundary cuts (17, 25], (25, 35], (35, 45], (45, 60]."""
    s = pd.Series([18, 25, 26, 35, 36, 45, 46, 60], index=range(8))
    res = derive_age_group(s)
    expected = [
        "18-25",
        "18-25",
        "26-35",
        "26-35",
        "36-45",
        "36-45",
        "46-60",
        "46-60",
    ]
    assert res.tolist() == expected
    assert list(res.cat.categories) == ["18-25", "26-35", "36-45", "46-60"]
    assert res.cat.ordered


def test_age_group_out_of_bounds_raises():
    """Out-of-range age must raise ValueError naming offending RowIDs."""
    s = pd.Series([17, 25, 61], index=[10, 20, 30])
    with pytest.raises(ValueError, match="RowIDs: \\[10, 30\\]"):
        derive_age_group(s)


def test_tenure_band_boundaries():
    """Verify TenureBand boundaries: 0, 1, 2, 3, 5, 6, 10, 11."""
    s = pd.Series([0, 1, 2, 3, 5, 6, 10, 11], index=range(8))
    res = derive_tenure_band(s)
    expected = ["0-1", "0-1", "2-3", "2-3", "4-5", "6-10", "6-10", "11+"]
    assert res.tolist() == expected
    assert list(res.cat.categories) == ["0-1", "2-3", "4-5", "6-10", "11+"]
    assert res.cat.ordered


def test_tenure_band_negative_raises():
    s = pd.Series([-1, 5], index=[101, 102])
    with pytest.raises(ValueError, match="RowIDs: \\[101\\]"):
        derive_tenure_band(s)


def test_early_tenure():
    s = pd.Series([0, 1, 2, 5], index=range(4))
    res = derive_early_tenure(s)
    assert res.tolist() == [1, 1, 0, 0]
    assert res.dtype == "int64"


def test_career_stage_boundaries():
    s = pd.Series([0, 5, 6, 10, 11, 20, 21, 35], index=range(8))
    res = derive_career_stage(s)
    expected = ["0-5", "0-5", "6-10", "6-10", "11-20", "11-20", "21+", "21+"]
    assert res.tolist() == expected
    assert list(res.cat.categories) == ["0-5", "6-10", "11-20", "21+"]
    assert res.cat.ordered


def test_role_tenure_band_boundaries():
    s = pd.Series([0, 1, 2, 3, 4, 7, 8, 15], index=range(8))
    res = derive_role_tenure_band(s)
    expected = ["0-1", "0-1", "2-3", "2-3", "4-7", "4-7", "8+", "8+"]
    assert res.tolist() == expected
    assert list(res.cat.categories) == ["0-1", "2-3", "4-7", "8+"]
    assert res.cat.ordered


def test_manager_tenure_band_boundaries():
    s = pd.Series([0, 1, 2, 3, 4, 7, 8, 12], index=range(8))
    res = derive_manager_tenure_band(s)
    expected = ["0-1", "0-1", "2-3", "2-3", "4-7", "4-7", "8+", "8+"]
    assert res.tolist() == expected
    assert list(res.cat.categories) == ["0-1", "2-3", "4-7", "8+"]
    assert res.cat.ordered


def test_promotion_gap_band_boundaries():
    s = pd.Series([0, 1, 2, 3, 5, 6, 10], index=range(7))
    res = derive_promotion_gap_band(s)
    expected = ["0", "1-2", "1-2", "3-5", "3-5", "6+", "6+"]
    assert res.tolist() == expected
    assert list(res.cat.categories) == ["0", "1-2", "3-5", "6+"]
    assert res.cat.ordered


def test_distance_band_boundaries():
    s = pd.Series([1, 5, 6, 10, 11, 20, 21, 29], index=range(8))
    res = derive_distance_band(s)
    expected = ["1-5", "1-5", "6-10", "6-10", "11-20", "11-20", "21-29", "21-29"]
    assert res.tolist() == expected
    assert list(res.cat.categories) == ["1-5", "6-10", "11-20", "21-29"]
    assert res.cat.ordered


def test_distance_band_out_of_bounds_raises():
    s = pd.Series([0, 5, 30], index=[5, 6, 7])
    with pytest.raises(ValueError, match="RowIDs: \\[5, 7\\]"):
        derive_distance_band(s)


def test_prior_experience_years_and_flag():
    twy = pd.Series([10, 5, 0], index=[0, 1, 2])
    yac = pd.Series([5, 5, 0], index=[0, 1, 2])
    exp = derive_prior_experience_years(twy, yac)
    assert exp.tolist() == [5, 0, 0]
    assert exp.dtype == "int64"

    has_exp = derive_has_prior_experience(exp)
    assert has_exp.tolist() == [1, 0, 0]
    assert has_exp.dtype == "int64"


def test_prior_experience_years_negative_raises():
    twy = pd.Series([5], index=[42])
    yac = pd.Series([8], index=[42])
    with pytest.raises(ValueError, match="RowIDs: \\[42\\]"):
        derive_prior_experience_years(twy, yac)


def test_zero_denominator_rule_and_no_warnings():
    """Verify zero-denominator rule: YearsAtCompany==0 produces NaN with zero warnings."""
    yac = pd.Series([0, 0, 5, 10], index=range(4))
    role = pd.Series([0, 0, 3, 5], index=range(4))
    prom = pd.Series([0, 0, 1, 2], index=range(4))

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        role_share = derive_role_tenure_share(role, yac)
        prom_share = derive_promotion_gap_share(prom, yac)

        # Assert no warnings emitted
        assert len(recorded_warnings) == 0, f"Warnings were emitted: {recorded_warnings}"

    assert np.isnan(role_share.iloc[0])
    assert np.isnan(role_share.iloc[1])
    assert role_share.iloc[2] == 0.6
    assert role_share.iloc[3] == 0.5

    assert np.isnan(prom_share.iloc[0])
    assert np.isnan(prom_share.iloc[1])
    assert prom_share.iloc[2] == 0.2
    assert prom_share.iloc[3] == 0.2


def test_promotion_gap_equals_tenure():
    yac = pd.Series([0, 3, 5, 5], index=range(4))
    prom = pd.Series([0, 3, 2, 5], index=range(4))
    res = derive_promotion_gap_equals_tenure(yac, prom)
    # Row 0: YearsAtCompany==0 -> must be 0
    # Row 1: 3 == 3 -> 1
    # Row 2: 2 != 5 -> 0
    # Row 3: 5 == 5 -> 1
    assert res.tolist() == [0, 1, 0, 1]
    assert res.dtype == "int64"


def test_promotion_analysis_cohort():
    yac = pd.Series([0, 4, 5, 10], index=range(4))
    res = derive_promotion_analysis_cohort(yac, min_tenure=5)
    assert res.tolist() == [0, 0, 1, 1]
    assert res.dtype == "int64"


def test_outstanding_rating():
    perf = pd.Series([3, 4, 3, 4], index=range(4))
    res = derive_outstanding_rating(perf)
    assert res.tolist() == [0, 1, 0, 1]
    assert res.dtype == "int64"
