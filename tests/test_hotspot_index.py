"""Unit and algorithmic verification tests for workforce_intel.hotspot_index module."""

import numpy as np
import pandas as pd
import pytest

from workforce_intel.hotspot_index import (
    compute_greedy_coverage,
    compute_hotspot_index,
    compute_pairwise_overlap,
    compute_shrunken_rate,
    compute_wilson_interval,
    fit_beta_binomial_prior_strength,
    screen_hotspot_tiers,
)


def test_shrinkage_limits():
    """Verify shrinkage limits as m -> 0 and m -> infinity."""
    k = 25
    n = 50
    raw_rate = k / n  # 0.50
    p0 = 0.1612

    # As m -> 0, shrunken rate approaches raw rate
    rate_m_zero = compute_shrunken_rate(k, n, m=1e-6, p0=p0)
    assert rate_m_zero == pytest.approx(raw_rate, abs=1e-4)

    # As m -> infinity, shrunken rate approaches prior mean p0
    rate_m_inf = compute_shrunken_rate(k, n, m=1e9, p0=p0)
    assert rate_m_inf == pytest.approx(p0, abs=1e-4)


def test_beta_binomial_prior_strength_recovery():
    """Verify marginal maximum likelihood recovers known m on simulated Beta-Binomial data."""
    rng = np.random.default_rng(42)
    true_m = 35.0
    true_p0 = 0.20
    alpha = true_m * true_p0  # 7.0
    beta_param = true_m * (1.0 - true_p0)  # 28.0

    # Simulate 200 groups with n ~ Uniform(30, 100)
    n_groups = 200
    counts = []
    for _ in range(n_groups):
        n_j = rng.integers(30, 100)
        p_j = rng.beta(alpha, beta_param)
        k_j = rng.binomial(n_j, p_j)
        counts.append((k_j, n_j))

    est_m = fit_beta_binomial_prior_strength(counts, true_p0)
    # Check that estimated m is within reasonable bounds of true_m (e.g. +/- 15)
    assert 20.0 <= est_m <= 50.0


def test_greedy_coverage_constructed_example():
    """Verify greedy coverage on a constructed example with deterministic sequence."""
    # Create 100 employees:
    # Segment 1: employees 0..39 (n=40, 20 leavers = 50%)
    # Segment 2: employees 30..69 (n=40, 10 leavers = 25%, overlaps 10 with Seg 1)
    # Total workforce leavers: 25 leavers across 100 employees (baseline 25%)
    rows = []
    for i in range(100):
        # Default stayer: leavers for i < 20 (20 leavers)
        lea = 1 if i < 20 else 0
        seg1 = "S1" if i < 40 else "Other"
        seg2 = "S2" if 30 <= i < 70 else "Other"
        rows.append({"EmpID": i, "Attrition": lea, "Seg1": seg1, "Seg2": seg2})

    df_test = pd.DataFrame(rows)
    p0 = df_test["Attrition"].mean()

    # Define tier 1 candidate segments
    t1_df = pd.DataFrame([
        {
            "segment_id": "Seg1=S1",
            "family": "1-way",
            "col1": "Seg1",
            "val1": "S1",
            "col2": "",
            "val2": "",
            "n": 40,
            "k": 20,
            "raw_rate": 0.50,
            "ci_low": 0.35,
            "ci_high": 0.65,
            "shrunken_rate": 0.45,
            "hotspot_index": 180.0,
        },
        {
            "segment_id": "Seg2=S2",
            "family": "1-way",
            "col1": "Seg2",
            "val1": "S2",
            "col2": "",
            "val2": "",
            "n": 40,
            "k": 10,
            "raw_rate": 0.25,
            "ci_low": 0.14,
            "ci_high": 0.40,
            "shrunken_rate": 0.25,
            "hotspot_index": 100.0,
        },
    ])

    greedy_res = compute_greedy_coverage(
        df_test,
        t1_df,
        p0_workforce=p0,
        m_1way=10.0,
        m_2way=10.0,
        max_steps=5,
        min_uncovered_n=30,
    )

    # Step 1 must select Seg1=S1 because of higher rate and lower-bound separation
    assert len(greedy_res) >= 1
    assert greedy_res["segment_id"].iloc[0] == "Seg1=S1"
    assert greedy_res["new_employees"].iloc[0] == 40
    assert greedy_res["new_leavers"].iloc[0] == 20


def test_tier_gating_logic():
    """Verify gating conditions: n >= 30, lower bound > p0, and FDR clearance."""
    p0 = 0.1612
    total_n = 1470
    total_k = 237

    # Candidate A: n=50, k=25 (rate 50%, Wilson lower 36.6% > p0, strong p-val) -> Tier 1
    # Candidate B: n=20, k=15 (rate 75%, but n < 30) -> Neither Tier 1 nor Tier 2
    # Candidate C: n=50, k=8 (rate 16%, Wilson lower 8.3% <= p0) -> Neither
    univ_mock = pd.DataFrame([
        {
            "segment_id": "A",
            "family": "1-way",
            "col1": "X",
            "val1": "a",
            "col2": "",
            "val2": "",
            "n": 50,
            "k": 25,
            "raw_rate": 0.50,
        },
        {
            "segment_id": "B",
            "family": "1-way",
            "col1": "X",
            "val1": "b",
            "col2": "",
            "val2": "",
            "n": 20,
            "k": 15,
            "raw_rate": 0.75,
        },
        {
            "segment_id": "C",
            "family": "1-way",
            "col1": "X",
            "val1": "c",
            "col2": "",
            "val2": "",
            "n": 50,
            "k": 8,
            "raw_rate": 0.16,
        },
    ])

    screened = screen_hotspot_tiers(
        univ_mock,
        total_leavers=total_k,
        total_headcount=total_n,
        p0=p0,
        m_1way=30.0,
        m_2way=10.0,
    )

    res_a = screened[screened["segment_id"] == "A"].iloc[0]
    res_b = screened[screened["segment_id"] == "B"].iloc[0]
    res_c = screened[screened["segment_id"] == "C"].iloc[0]

    assert bool(res_a["is_tier1"]) is True
    assert bool(res_b["is_tier1"]) is False and bool(res_b["is_tier2"]) is False  # Fails n >= 30
    assert bool(res_c["is_tier1"]) is False and bool(res_c["is_tier2"]) is False  # Fails lower bound > p0
