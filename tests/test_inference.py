"""Unit and mathematical verification tests for workforce_intel.inference module."""

import numpy as np
import pytest
from scipy.stats import chi2_contingency, fisher_exact

from workforce_intel.inference import (
    apply_benjamini_hochberg,
    bootstrap_cramers_v,
    breslow_day_tarone_test,
    cochran_armitage_test,
    compute_cramers_v,
    compute_robins_breslow_greenland_ci,
    mann_whitney_u_analysis,
    permutation_chi2_test,
    run_mantel_haenszel,
)


def test_hand_computed_chi2_2x2():
    """Verify Chi-square test on known 2x2 contingency tables."""
    # Independent table: exactly 0 chi2
    ct_indep = np.array([[10, 10], [10, 10]])
    res = chi2_contingency(ct_indep, correction=False)
    assert res[0] == pytest.approx(0.0, abs=1e-9)
    assert res[1] == pytest.approx(1.0, abs=1e-9)

    # Perfectly separated table: chi2 = N = 20
    ct_perf = np.array([[10, 0], [0, 10]])
    res_perf = chi2_contingency(ct_perf, correction=False)
    assert res_perf[0] == pytest.approx(20.0, abs=1e-9)
    assert res_perf[1] < 1e-4


def test_hand_computed_fisher_exact():
    """Verify Fisher exact test on known 2x2 contingency table."""
    ct = [[10, 0], [0, 10]]
    odds, p_val = fisher_exact(ct, alternative="greater")
    assert np.isinf(odds) or odds > 1e6
    assert p_val < 1e-4


def test_hand_computed_cramer_v():
    """Verify Cramer's V calculation on extreme and known tables."""
    # Zero association
    ct_zero = np.array([[20, 20], [20, 20]])
    assert compute_cramers_v(ct_zero) == pytest.approx(0.0, abs=1e-9)

    # Perfect association
    ct_perf = np.array([[20, 0], [0, 20]])
    assert compute_cramers_v(ct_perf) == pytest.approx(1.0, abs=1e-9)

    # Bootstrap CI produces valid bounds containing point estimate
    x = np.array([0] * 50 + [1] * 50)
    y = np.array([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)
    pt, low, high = bootstrap_cramers_v(x, y, n_bootstrap=200, seed=42)
    assert 0.0 <= low <= pt <= high <= 1.0


def test_hand_computed_cochran_armitage():
    """Verify Cochran-Armitage trend test on hand-derived linear trend."""
    # Groups 1, 2, 3 with strictly increasing case counts
    cases = np.array([1, 5, 9])
    totals = np.array([10, 10, 10])
    scores = np.array([1, 2, 3])

    z, chi2_val, p_val = cochran_armitage_test(cases, totals, scores)
    assert z > 0.0  # Positive trend
    assert chi2_val == pytest.approx(z**2, rel=1e-5)
    assert p_val < 0.01

    # Flat trend: exactly zero z
    cases_flat = np.array([5, 5, 5])
    z_f, chi2_f, p_f = cochran_armitage_test(cases_flat, totals, scores)
    assert z_f == pytest.approx(0.0, abs=1e-9)
    assert p_f == pytest.approx(1.0, abs=1e-9)


def test_hand_computed_mantel_haenszel_two_strata():
    """Verify Mantel-Haenszel common OR on hand-computed two-stratum example.

    Stratum 1: a=10, b=20, c=5, d=40 (n=75). ad/n = 400/75 = 5.3333, bc/n = 100/75 = 1.3333. Stratum OR = (10*40)/(20*5) = 4.0.
    Stratum 2: a=20, b=10, c=10, d=20 (n=60). ad/n = 400/60 = 6.6667, bc/n = 100/60 = 1.6667. Stratum OR = (20*20)/(10*10) = 4.0.
    Pooled MH OR = (5.3333 + 6.6667) / (1.3333 + 1.6667) = 12.0 / 3.0 = 4.0.
    Since both stratum ORs equal 4.0 exactly, Breslow-Day Tarone statistic must equal 0.0.
    """
    tables = [
        (10, 20, 5, 40),
        (20, 10, 10, 20),
    ]
    names = ["Stratum 1", "Stratum 2"]
    res = run_mantel_haenszel(names, tables)

    assert res["mh_or"] == pytest.approx(4.0, rel=1e-4)
    # Breslow-Day homogeneity test should be close to 0 with p = 1.0
    assert res["breslow_day_tarone_stat"] == pytest.approx(0.0, abs=1e-4)
    assert res["breslow_day_p"] == pytest.approx(1.0, abs=1e-4)

    # Robins-Breslow-Greenland CI bounds must span 4.0
    assert res["mh_ci_low"] < 4.0 < res["mh_ci_high"]


def test_hand_computed_benjamini_hochberg():
    """Verify BH FDR q-value adjustment on a known p-value vector."""
    # p = [0.01, 0.04, 0.03, 0.20]
    # Sorted: (0.01, rank 1), (0.03, rank 2), (0.04, rank 3), (0.20, rank 4)
    # q4 = 0.20 * 4/4 = 0.20
    # q3 = min(0.20, 0.04 * 4/3) = 0.05333
    # q2 = min(0.05333, 0.03 * 4/2) = 0.05333
    # q1 = min(0.05333, 0.01 * 4/1) = 0.04
    p_vec = [0.01, 0.04, 0.03, 0.20]
    q_vec = apply_benjamini_hochberg(p_vec)

    assert q_vec[0] == pytest.approx(0.04, abs=1e-4)
    assert q_vec[1] == pytest.approx(0.05333, abs=1e-4)
    assert q_vec[2] == pytest.approx(0.05333, abs=1e-4)
    assert q_vec[3] == pytest.approx(0.20, abs=1e-4)


def test_edge_cases_mantel_haenszel():
    """Verify edge cases: zero exposure stratum exclusion and zero cell continuity correction."""
    names = ["Stratum Valid", "Stratum Zero Exposed", "Stratum Zero Cell"]
    # Stratum 1: standard
    # Stratum 2: exposed = 0 (a=0, b=0) -> must be excluded from MH pooling
    # Stratum 3: zero cell (c=0) -> valid for pooling without correction, display gets +0.5
    tables = [
        (10, 10, 5, 20),
        (0, 0, 10, 30),
        (5, 10, 0, 15),
    ]

    res = run_mantel_haenszel(names, tables)
    assert "Stratum Zero Exposed" in res["excluded_strata"]
    assert len(res["stratum_results"]) == 2

    # Check zero cell stratum display adjustment
    s_zero = [r for r in res["stratum_results"] if r["stratum"] == "Stratum Zero Cell"][0]
    assert s_zero["has_zero_cell"] is True
    assert s_zero["adjustment"] == "continuity-corrected"
    # Adjusted OR = (5.5 * 15.5) / (10.5 * 0.5)
    expected_adj_or = (5.5 * 15.5) / (10.5 * 0.5)
    assert s_zero["stratum_or"] == pytest.approx(expected_adj_or, rel=1e-3)


def test_edge_case_single_level_and_n1():
    """Verify functions handle single-level dimensions and n=1 without crashing."""
    # n=1
    pt, l, h = bootstrap_cramers_v(np.array([0]), np.array([1]), n_bootstrap=10)
    assert pt == 0.0

    # Single-level category
    z, chi2_val, p_val = cochran_armitage_test(np.array([5]), np.array([10]), np.array([1]))
    assert z == 0.0
    assert p_val == 1.0

    # Mann-Whitney on single element
    res = mann_whitney_u_analysis(np.array([10.0]), np.array([20.0]), n_bootstrap=10)
    assert res["prob_superiority"] in [0.0, 1.0]
