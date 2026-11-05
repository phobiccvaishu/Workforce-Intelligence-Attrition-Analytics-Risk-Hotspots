"""Statistical utilities for confidence intervals and exact binomial funnel limits."""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np
from scipy import stats

Z_95 = 1.959963984540054


def wilson_interval(
    k: int | float, n: int | float, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int or float
        Number of successes / leavers.
    n : int or float
        Number of trials / total group size.
    confidence : float, default 0.95
        Confidence level.

    Returns
    -------
    tuple of (float, float)
        (ci_lower, ci_upper) bounded in [0.0, 1.0]. Returns (nan, nan) if n <= 0.
    """
    if n <= 0:
        return (float("nan"), float("nan"))

    if confidence == 0.95:
        z = Z_95
    else:
        alpha = 1.0 - confidence
        z = stats.norm.ppf(1.0 - alpha / 2.0)

    p = float(k) / float(n)
    denominator = 1.0 + (z**2) / n
    center_adj = p + (z**2) / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))
    ci_lower = max(0.0, (center_adj - margin) / denominator)
    ci_upper = min(1.0, (center_adj + margin) / denominator)

    if k == 0:
        ci_lower = 0.0
    if k == n:
        ci_upper = 1.0

    return (ci_lower, ci_upper)


def exact_binomial_limits(
    n: int, p0: float, level: float = 0.95
) -> Tuple[float, float]:
    """Calculate exact binomial control limits for a funnel plot at sample size n.

    Parameters
    ----------
    n : int
        Sample size (group size).
    p0 : float
        Overall population baseline proportion.
    level : float, default 0.95
        Confidence level (e.g. 0.95 for 2-sigma warning, 0.998 for 3-sigma action).

    Returns
    -------
    tuple of (float, float)
        (lower_limit_rate, upper_limit_rate) bounded in [0.0, 1.0].
    """
    if n <= 0:
        return (float("nan"), float("nan"))

    alpha = 1.0 - level
    lower_k = stats.binom.ppf(alpha / 2.0, n, p0)
    upper_k = stats.binom.ppf(1.0 - alpha / 2.0, n, p0)

    lower_limit = max(0.0, float(lower_k) / float(n))
    upper_limit = min(1.0, float(upper_k) / float(n))

    return (lower_limit, upper_limit)
