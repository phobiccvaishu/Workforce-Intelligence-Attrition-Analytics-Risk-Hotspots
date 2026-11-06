"""Overlap-aware empirical Bayes hotspot index module for Workforce Intelligence.

Provides pure functions for:
1. Marginal maximum likelihood estimation of beta-binomial prior strength (m).
2. Empirical Bayes shrinkage estimation of segment attrition rates and relative risk index.
3. Two-tier statistical screening (Tier 1: sample size, Wilson lower bound, FDR q < 0.05;
   Tier 2: sample size and Wilson lower bound).
4. Pairwise candidate overlap analysis (intersection counts and Jaccard similarity).
5. Greedy marginal coverage algorithm over uncovered employee populations.
6. Split-sample out-of-sample validation against unseen holdout partitions.
7. Sensitivity analyses across prior strength settings (m in {10, 30, 100}).

All functions are deterministic and side-effect free.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln
from scipy.stats import fisher_exact, spearmanr
from statsmodels.stats.multitest import multipletests

from workforce_intel.config import (
    ALPHA_LEVEL,
    HOTSPOT_ONE_WAY_COLS,
    HOTSPOT_TWO_WAY_PAIRS,
    MAX_GREEDY_STEPS,
    MIN_GROUP_SIZE,
    PRIOR_STRENGTH_SENSITIVITY,
)


def compute_wilson_interval(
    k: int,
    n: int,
    alpha: float = ALPHA_LEVEL,
) -> Tuple[float, float]:
    """Compute exact Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = float(k) / float(n)
    z = 1.959963984540054  # Normal critical value for alpha=0.05
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def fit_beta_binomial_prior_strength(
    counts: List[Tuple[int, int]],
    p0: float,
    bounds: Tuple[float, float] = (0.1, 1000.0),
) -> float:
    """Estimate prior strength m via marginal maximum likelihood for a Beta-Binomial model.

    The prior mean is fixed at the population baseline rate p0, so:
    alpha = m * p0, beta = m * (1 - p0).
    """
    if not counts or p0 <= 0.0 or p0 >= 1.0:
        return 30.0

    def neg_marginal_log_likelihood(m: float) -> float:
        if m <= 0.0:
            return 1e12
        a = m * p0
        b = m * (1.0 - p0)
        ll = 0.0
        for k_val, n_val in counts:
            if n_val <= 0:
                continue
            ll += betaln(k_val + a, (n_val - k_val) + b) - betaln(a, b)
        return -float(ll)

    opt_res = minimize_scalar(neg_marginal_log_likelihood, bounds=bounds, method="bounded")
    return float(opt_res.x)


def compute_shrunken_rate(k: int, n: int, m: float, p0: float) -> float:
    """Compute empirical Bayes posterior mean rate: (k + m * p0) / (n + m)."""
    if n + m <= 0.0:
        return p0
    return float((k + m * p0) / (n + m))


def compute_hotspot_index(shrunken_rate: float, p0: float) -> float:
    """Compute relative risk index scaled to 100 at baseline: 100 * (shrunken_rate / p0)."""
    if p0 <= 0.0:
        return 100.0
    return float(100.0 * (shrunken_rate / p0))


def build_hotspot_universe(
    df: pd.DataFrame,
    one_way_cols: List[str] = HOTSPOT_ONE_WAY_COLS,
    two_way_pairs: List[Tuple[str, str]] = HOTSPOT_TWO_WAY_PAIRS,
) -> pd.DataFrame:
    """Enumerate the complete pre-specified candidate segment universe."""
    records = []

    # One-way segments
    for col in one_way_cols:
        for val in sorted(df[col].unique(), key=lambda x: str(x)):
            mask = df[col] == val
            k = int(df.loc[mask, "Attrition"].sum())
            n = int(mask.sum())
            records.append({
                "segment_id": f"{col}={val}",
                "family": "1-way",
                "col1": col,
                "val1": str(val),
                "col2": "",
                "val2": "",
                "n": n,
                "k": k,
                "raw_rate": float(k / n) if n > 0 else 0.0,
            })

    # Two-way interaction cells
    for c1, c2 in two_way_pairs:
        for v1 in sorted(df[c1].unique(), key=lambda x: str(x)):
            for v2 in sorted(df[c2].unique(), key=lambda x: str(x)):
                mask = (df[c1] == v1) & (df[c2] == v2)
                k = int(df.loc[mask, "Attrition"].sum())
                n = int(mask.sum())
                records.append({
                    "segment_id": f"{c1}={v1} & {c2}={v2}",
                    "family": "2-way",
                    "col1": c1,
                    "val1": str(v1),
                    "col2": c2,
                    "val2": str(v2),
                    "n": n,
                    "k": k,
                    "raw_rate": float(k / n) if n > 0 else 0.0,
                })

    return pd.DataFrame(records)


def screen_hotspot_tiers(
    universe_df: pd.DataFrame,
    total_leavers: int,
    total_headcount: int,
    p0: float,
    m_1way: float,
    m_2way: float,
    min_n: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Screen universe into Tier 1 and Tier 2 categories with multiplicity control."""
    res_df = universe_df.copy()

    ci_lows = []
    ci_highs = []
    p_fishers = []
    shrunken_rates = []
    indices = []

    for _, row in res_df.iterrows():
        k = int(row["k"])
        n = int(row["n"])
        m_val = m_1way if row["family"] == "1-way" else m_2way

        l_val, h_val = compute_wilson_interval(k, n)
        ci_lows.append(l_val)
        ci_highs.append(h_val)

        # Fisher exact test: segment leavers/stayers vs rest of workforce
        rest_k = total_leavers - k
        rest_n = total_headcount - n
        rest_stayers = rest_n - rest_k
        seg_stayers = n - k
        table = [[k, seg_stayers], [rest_k, rest_stayers]]
        _, p_f = fisher_exact(table, alternative="greater")
        p_fishers.append(float(p_f))

        s_rate = compute_shrunken_rate(k, n, m_val, p0)
        shrunken_rates.append(s_rate)
        indices.append(compute_hotspot_index(s_rate, p0))

    res_df["ci_low"] = ci_lows
    res_df["ci_high"] = ci_highs
    res_df["p_fisher"] = p_fishers
    res_df["shrunken_rate"] = shrunken_rates
    res_df["hotspot_index"] = indices

    # Multiplicity control across the entire candidate universe
    _, q_vals, _, _ = multipletests(res_df["p_fisher"].values, alpha=ALPHA_LEVEL, method="fdr_bh")
    res_df["q_bh"] = q_vals

    res_df["meets_min_n"] = res_df["n"] >= min_n
    res_df["lower_bound_separates"] = res_df["ci_low"] > p0
    res_df["fdr_passes"] = res_df["q_bh"] < ALPHA_LEVEL

    # Gate logic:
    # Tier 1: n >= 30 AND Wilson lower > p0 AND BH q < 0.05
    # Tier 2: n >= 30 AND Wilson lower > p0 AND NOT Tier 1
    res_df["is_tier1"] = (
        res_df["meets_min_n"] & res_df["lower_bound_separates"] & res_df["fdr_passes"]
    )
    res_df["is_tier2"] = (
        res_df["meets_min_n"] & res_df["lower_bound_separates"] & (~res_df["is_tier1"])
    )

    return res_df


def compute_pairwise_overlap(
    df: pd.DataFrame,
    candidate_segments: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise intersection headcount and Jaccard similarity across candidate segments."""
    seg_masks = {}
    seg_ids = list(candidate_segments["segment_id"].values)

    for _, row in candidate_segments.iterrows():
        sid = row["segment_id"]
        c1, v1 = row["col1"], row["val1"]
        c2, v2 = row["col2"], row["val2"]
        if row["family"] == "1-way":
            mask = df[c1].astype(str) == v1
        else:
            mask = (df[c1].astype(str) == v1) & (df[c2].astype(str) == v2)
        seg_masks[sid] = mask

    n_segs = len(seg_ids)
    intersect_matrix = np.zeros((n_segs, n_segs), dtype=int)
    jaccard_matrix = np.zeros((n_segs, n_segs), dtype=float)

    for i in range(n_segs):
        m1 = seg_masks[seg_ids[i]]
        for j in range(n_segs):
            m2 = seg_masks[seg_ids[j]]
            intersect_cnt = int((m1 & m2).sum())
            union_cnt = int((m1 | m2).sum())
            intersect_matrix[i, j] = intersect_cnt
            jaccard_matrix[i, j] = float(intersect_cnt / union_cnt) if union_cnt > 0 else 0.0

    intersect_df = pd.DataFrame(intersect_matrix, index=seg_ids, columns=seg_ids)
    jaccard_df = pd.DataFrame(jaccard_matrix, index=seg_ids, columns=seg_ids)
    return intersect_df, jaccard_df


def compute_greedy_coverage(
    df: pd.DataFrame,
    tier1_segments: pd.DataFrame,
    p0_workforce: float,
    m_1way: float,
    m_2way: float,
    max_steps: int = MAX_GREEDY_STEPS,
    min_uncovered_n: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Iteratively cover employees using the Tier 1 segment with highest shrunken index on uncovered pool."""
    total_n = len(df)
    total_leavers = int(df["Attrition"].sum())

    covered_mask = np.zeros(total_n, dtype=bool)
    selected_segments = []
    step_records = []

    # Pre-build boolean masks for Tier 1 segments
    masks = {}
    for _, row in tier1_segments.iterrows():
        sid = row["segment_id"]
        c1, v1 = row["col1"], row["val1"]
        c2, v2 = row["col2"], row["val2"]
        if row["family"] == "1-way":
            m = df[c1].astype(str) == v1
        else:
            m = (df[c1].astype(str) == v1) & (df[c2].astype(str) == v2)
        masks[sid] = m.values

    available_ids = list(tier1_segments["segment_id"].values)

    for step in range(1, max_steps + 1):
        uncovered_mask = ~covered_mask
        n_uncovered_pool = int(uncovered_mask.sum())
        if n_uncovered_pool < min_uncovered_n:
            break

        k_uncovered_pool = int(df.loc[uncovered_mask, "Attrition"].sum())
        p0_uncovered = float(k_uncovered_pool / n_uncovered_pool) if n_uncovered_pool > 0 else 0.0

        best_sid = None
        best_index = -1.0
        best_n_new = 0
        best_k_new = 0
        best_mask_new = None

        for sid in available_ids:
            seg_row = tier1_segments[tier1_segments["segment_id"] == sid].iloc[0]
            m_val = m_1way if seg_row["family"] == "1-way" else m_2way
            seg_mask = masks[sid]
            marginal_mask = seg_mask & uncovered_mask
            n_new = int(marginal_mask.sum())
            k_new = int(df.loc[marginal_mask, "Attrition"].sum())

            if n_new < min_uncovered_n:
                continue

            ci_l, _ = compute_wilson_interval(k_new, n_new)
            if ci_l <= p0_uncovered:
                continue

            s_rate = compute_shrunken_rate(k_new, n_new, m_val, p0_uncovered)
            idx_val = compute_hotspot_index(s_rate, p0_uncovered)

            if idx_val > best_index:
                best_index = idx_val
                best_sid = sid
                best_n_new = n_new
                best_k_new = k_new
                best_mask_new = marginal_mask

        if best_sid is None:
            # No remaining segment meets min sample size and lower-bound separation on uncovered pool
            break

        covered_mask = covered_mask | best_mask_new
        available_ids.remove(best_sid)
        selected_segments.append(best_sid)

        cum_n = int(covered_mask.sum())
        cum_k = int(df.loc[covered_mask, "Attrition"].sum())

        step_records.append({
            "step": step,
            "segment_id": best_sid,
            "new_employees": best_n_new,
            "new_leavers": best_k_new,
            "marginal_rate": float(best_k_new / best_n_new),
            "marginal_shrunken_index": best_index,
            "cum_employees": cum_n,
            "cum_leavers": cum_k,
            "cum_rate": float(cum_k / cum_n),
            "share_of_workforce": float(cum_n / total_n),
            "share_of_leavers": float(cum_k / total_leavers),
            "uncovered_pool_baseline": p0_uncovered,
        })

    return pd.DataFrame(step_records)


def evaluate_split_sample_validation(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    full_tier1_ids: List[str],
    m_1way_train: float,
    m_2way_train: float,
) -> pd.DataFrame:
    """Screen candidate hotspots on training split and validate on unseen holdout."""
    p0_train = float(train_df["Attrition"].mean())
    p0_holdout = float(holdout_df["Attrition"].mean())
    n_train_total = len(train_df)
    k_train_total = int(train_df["Attrition"].sum())

    train_universe = build_hotspot_universe(train_df)
    train_screened = screen_hotspot_tiers(
        train_universe,
        k_train_total,
        n_train_total,
        p0_train,
        m_1way_train,
        m_2way_train,
    )

    # Focus validation on training Tier 1 candidates
    train_t1 = train_screened[train_screened["is_tier1"]].copy()

    val_records = []
    for _, row in train_t1.iterrows():
        sid = row["segment_id"]
        c1, v1 = row["col1"], row["val1"]
        c2, v2 = row["col2"], row["val2"]

        if row["family"] == "1-way":
            h_mask = holdout_df[c1].astype(str) == v1
        else:
            h_mask = (holdout_df[c1].astype(str) == v1) & (holdout_df[c2].astype(str) == v2)

        n_hold = int(h_mask.sum())
        k_hold = int(holdout_df.loc[h_mask, "Attrition"].sum())
        h_rate = float(k_hold / n_hold) if n_hold > 0 else 0.0
        h_ci_l, h_ci_h = compute_wilson_interval(k_hold, n_hold)

        val_records.append({
            "segment_id": sid,
            "family": row["family"],
            "train_n": row["n"],
            "train_k": row["k"],
            "train_rate": row["raw_rate"],
            "train_ci_low": row["ci_low"],
            "train_ci_high": row["ci_high"],
            "train_shrunken_rate": row["shrunken_rate"],
            "train_hotspot_index": row["hotspot_index"],
            "holdout_n": n_hold,
            "holdout_k": k_hold,
            "holdout_rate": h_rate,
            "holdout_ci_low": h_ci_l,
            "holdout_ci_high": h_ci_h,
            "holdout_n_ge_30": n_hold >= MIN_GROUP_SIZE,
            "above_holdout_point": h_rate > p0_holdout,
            "above_holdout_ci_low": h_ci_l > p0_holdout,
            "in_full_tier1": sid in full_tier1_ids,
        })

    return pd.DataFrame(val_records)


def run_prior_sensitivity(
    universe_df: pd.DataFrame,
    p0: float,
    prior_strengths: List[int] = PRIOR_STRENGTH_SENSITIVITY,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit rank stability and membership changes across varying prior strengths."""
    eligible_df = universe_df[universe_df["n"] >= MIN_GROUP_SIZE].copy()

    rank_cols = {}
    rate_cols = {}

    for m_val in prior_strengths:
        s_rates = [
            compute_shrunken_rate(int(r["k"]), int(r["n"]), float(m_val), p0)
            for _, r in eligible_df.iterrows()
        ]
        eligible_df[f"rate_m{m_val}"] = s_rates
        eligible_df[f"rank_m{m_val}"] = eligible_df[f"rate_m{m_val}"].rank(
            ascending=False, method="min"
        )
        rate_cols[m_val] = eligible_df[f"rate_m{m_val}"]
        rank_cols[m_val] = eligible_df[f"rank_m{m_val}"]

    # Pairwise Spearman correlation across sensitivity grid
    corr_records = []
    pairs = [(10, 30), (30, 100), (10, 100)]
    for m_a, m_b in pairs:
        rho_all, _ = spearmanr(rank_cols[m_a], rank_cols[m_b])

        # Top 10 overlap
        top_a = set(eligible_df.nsmallest(10, f"rank_m{m_a}")["segment_id"])
        top_b = set(eligible_df.nsmallest(10, f"rank_m{m_b}")["segment_id"])
        overlap_cnt = len(top_a & top_b)

        corr_records.append({
            "comparison": f"m={m_a} vs m={m_b}",
            "spearman_rho_all_n30": float(rho_all),
            "top10_shared_count": overlap_cnt,
            "top10_jaccard": float(overlap_cnt / len(top_a | top_b)),
        })

    return eligible_df, pd.DataFrame(corr_records)
