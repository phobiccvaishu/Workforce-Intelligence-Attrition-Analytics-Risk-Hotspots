"""Exploratory Data Analysis (EDA) engine for Workforce Intelligence project.

Pure, side-effect-free, individually testable analytical functions.
Implements Phase 3A specifications: subgroup rate tables, exact binomial funnel limits,
concentration metrics, sensitivity evaluations, numeric summaries, and hotspot candidate scanning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from workforce_intel.config import (
    FUNNEL_LEVELS,
    HOTSPOT_MIN_N,
    HOTSPOT_TOP_K,
    MIN_GROUP_SIZE,
    ROUND_DIGITS,
    TARGET_COLUMN,
)
from workforce_intel.stats_utils import exact_binomial_limits, wilson_interval

logger = logging.getLogger(__name__)


def overall_summary(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Compute overall population baseline metrics."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    total_n = len(df)
    if total_n == 0:
        return pd.DataFrame(
            [
                {
                    "metric": "Overall Baseline",
                    "total_employees": 0,
                    "leavers": 0,
                    "retained": 0,
                    "attrition_rate_pct": np.nan,
                    "retention_rate_pct": np.nan,
                    "ci95_low_pct": np.nan,
                    "ci95_high_pct": np.nan,
                    "imbalance_ratio": np.nan,
                }
            ]
        )

    y = df[target_col]
    leavers = int((y == 1).sum())
    retained = int((y == 0).sum())
    rate = (leavers / total_n) * 100.0
    retention_rate = (retained / total_n) * 100.0
    ci_low, ci_high = wilson_interval(leavers, total_n, confidence=0.95)
    imbalance = (retained / leavers) if leavers > 0 else np.nan

    return pd.DataFrame(
        [
            {
                "metric": "Overall Baseline",
                "total_employees": total_n,
                "leavers": leavers,
                "retained": retained,
                "attrition_rate_pct": round(rate, ROUND_DIGITS),
                "retention_rate_pct": round(retention_rate, ROUND_DIGITS),
                "ci95_low_pct": round(ci_low * 100.0, ROUND_DIGITS),
                "ci95_high_pct": round(ci_high * 100.0, ROUND_DIGITS),
                "imbalance_ratio": round(imbalance, ROUND_DIGITS) if not np.isnan(imbalance) else np.nan,
            }
        ]
    )


def rate_table(
    df: pd.DataFrame,
    dim: str,
    min_n: int = MIN_GROUP_SIZE,
    target_col: str = TARGET_COLUMN,
    overall_rate_pct: Optional[float] = None,
) -> pd.DataFrame:
    """Compute one-way attrition rate breakdown for a given dimension."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if dim not in df.columns:
        raise ValueError(f"Grouping column '{dim}' not found in DataFrame.")

    columns = [
        "segment_dim",
        "segment",
        "n",
        "leavers",
        "retained",
        "attrition_rate_pct",
        "ci95_low_pct",
        "ci95_high_pct",
        "share_of_workforce_pct",
        "share_of_leavers_pct",
        "diff_vs_overall_pp",
        "ratio_vs_overall",
        "low_n_flag",
    ]

    total_df_n = len(df)
    total_df_leavers = int((df[target_col] == 1).sum()) if total_df_n > 0 else 0

    if overall_rate_pct is None:
        overall_rate_pct = (
            (total_df_leavers / total_df_n) * 100.0 if total_df_n > 0 else 0.0
        )

    if total_df_n == 0:
        return pd.DataFrame(columns=columns)

    s = df[dim]
    is_categorical = isinstance(s.dtype, pd.CategoricalDtype)
    if is_categorical:
        all_categories = [str(cat) for cat in s.cat.categories]
    else:
        # Sort unique observed values cleanly
        observed_vals = s.dropna().unique()
        try:
            sorted_vals = sorted(observed_vals)
        except TypeError:
            sorted_vals = sorted(observed_vals, key=lambda x: str(x))
        all_categories = [str(v) for v in sorted_vals]

    # Handle missing / NaN values explicitly as "Missing" segment
    has_missing = s.isna().sum() > 0

    rows: List[Dict[str, Any]] = []

    for cat in all_categories:
        mask = (s.astype(str) == cat) & s.notna()
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "segment_dim": dim,
                    "segment": cat,
                    "n": 0,
                    "leavers": 0,
                    "retained": 0,
                    "attrition_rate_pct": np.nan,
                    "ci95_low_pct": np.nan,
                    "ci95_high_pct": np.nan,
                    "share_of_workforce_pct": 0.0,
                    "share_of_leavers_pct": 0.0,
                    "diff_vs_overall_pp": np.nan,
                    "ratio_vs_overall": np.nan,
                    "low_n_flag": True,
                }
            )
            continue

        cat_df = df[mask]
        leavers = int((cat_df[target_col] == 1).sum())
        retained = int((cat_df[target_col] == 0).sum())
        rate = (leavers / n) * 100.0
        ci_low, ci_high = wilson_interval(leavers, n, confidence=0.95)
        wf_share = (n / total_df_n) * 100.0
        leaver_share = (
            (leavers / total_df_leavers) * 100.0 if total_df_leavers > 0 else 0.0
        )
        diff_pp = rate - overall_rate_pct
        ratio = (rate / overall_rate_pct) if overall_rate_pct > 0 else np.nan

        rows.append(
            {
                "segment_dim": dim,
                "segment": cat,
                "n": n,
                "leavers": leavers,
                "retained": retained,
                "attrition_rate_pct": round(rate, ROUND_DIGITS),
                "ci95_low_pct": round(ci_low * 100.0, ROUND_DIGITS),
                "ci95_high_pct": round(ci_high * 100.0, ROUND_DIGITS),
                "share_of_workforce_pct": round(wf_share, ROUND_DIGITS),
                "share_of_leavers_pct": round(leaver_share, ROUND_DIGITS),
                "diff_vs_overall_pp": round(diff_pp, ROUND_DIGITS),
                "ratio_vs_overall": round(ratio, ROUND_DIGITS) if not np.isnan(ratio) else np.nan,
                "low_n_flag": n < min_n,
            }
        )

    if has_missing:
        mask = s.isna()
        n = int(mask.sum())
        cat_df = df[mask]
        leavers = int((cat_df[target_col] == 1).sum())
        retained = int((cat_df[target_col] == 0).sum())
        rate = (leavers / n) * 100.0 if n > 0 else np.nan
        ci_low, ci_high = (
            wilson_interval(leavers, n, confidence=0.95)
            if n > 0
            else (np.nan, np.nan)
        )
        wf_share = (n / total_df_n) * 100.0
        leaver_share = (
            (leavers / total_df_leavers) * 100.0 if total_df_leavers > 0 else 0.0
        )
        diff_pp = (rate - overall_rate_pct) if not np.isnan(rate) else np.nan
        ratio = (
            (rate / overall_rate_pct)
            if (not np.isnan(rate) and overall_rate_pct > 0)
            else np.nan
        )

        rows.append(
            {
                "segment_dim": dim,
                "segment": "Missing",
                "n": n,
                "leavers": leavers,
                "retained": retained,
                "attrition_rate_pct": round(rate, ROUND_DIGITS) if not np.isnan(rate) else np.nan,
                "ci95_low_pct": round(ci_low * 100.0, ROUND_DIGITS) if not np.isnan(ci_low) else np.nan,
                "ci95_high_pct": round(ci_high * 100.0, ROUND_DIGITS) if not np.isnan(ci_high) else np.nan,
                "share_of_workforce_pct": round(wf_share, ROUND_DIGITS),
                "share_of_leavers_pct": round(leaver_share, ROUND_DIGITS),
                "diff_vs_overall_pp": round(diff_pp, ROUND_DIGITS) if not np.isnan(diff_pp) else np.nan,
                "ratio_vs_overall": round(ratio, ROUND_DIGITS) if not np.isnan(ratio) else np.nan,
                "low_n_flag": n < min_n,
            }
        )

    res_df = pd.DataFrame(rows, columns=columns)
    return res_df


def two_way_table(
    df: pd.DataFrame,
    dim1: str,
    dim2: str,
    min_n: int = MIN_GROUP_SIZE,
    target_col: str = TARGET_COLUMN,
    overall_rate_pct: Optional[float] = None,
) -> pd.DataFrame:
    """Compute two-way cross-classification attrition rates across two dimensions."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    if dim1 not in df.columns:
        raise ValueError(f"Grouping column '{dim1}' not found in DataFrame.")
    if dim2 not in df.columns:
        raise ValueError(f"Grouping column '{dim2}' not found in DataFrame.")

    columns = [
        dim1,
        dim2,
        "n",
        "leavers",
        "retained",
        "attrition_rate_pct",
        "ci95_low_pct",
        "ci95_high_pct",
        "share_of_workforce_pct",
        "share_of_leavers_pct",
        "diff_vs_overall_pp",
        "ratio_vs_overall",
        "low_n_flag",
    ]

    total_df_n = len(df)
    total_df_leavers = int((df[target_col] == 1).sum()) if total_df_n > 0 else 0

    if overall_rate_pct is None:
        overall_rate_pct = (
            (total_df_leavers / total_df_n) * 100.0 if total_df_n > 0 else 0.0
        )

    if total_df_n == 0:
        return pd.DataFrame(columns=columns)

    # Determine unique categories for both dimensions
    def get_categories(col: str) -> List[str]:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            return [str(cat) for cat in s.cat.categories]
        observed = s.dropna().unique()
        try:
            return [str(v) for v in sorted(observed)]
        except TypeError:
            return [str(v) for v in sorted(observed, key=lambda x: str(x))]

    cats1 = get_categories(dim1)
    cats2 = get_categories(dim2)

    rows: List[Dict[str, Any]] = []

    for c1 in cats1:
        for c2 in cats2:
            mask = (
                (df[dim1].astype(str) == c1)
                & df[dim1].notna()
                & (df[dim2].astype(str) == c2)
                & df[dim2].notna()
            )
            n = int(mask.sum())
            if n == 0:
                rows.append(
                    {
                        dim1: c1,
                        dim2: c2,
                        "n": 0,
                        "leavers": 0,
                        "retained": 0,
                        "attrition_rate_pct": np.nan,
                        "ci95_low_pct": np.nan,
                        "ci95_high_pct": np.nan,
                        "share_of_workforce_pct": 0.0,
                        "share_of_leavers_pct": 0.0,
                        "diff_vs_overall_pp": np.nan,
                        "ratio_vs_overall": np.nan,
                        "low_n_flag": True,
                    }
                )
                continue

            cell_df = df[mask]
            leavers = int((cell_df[target_col] == 1).sum())
            retained = int((cell_df[target_col] == 0).sum())
            rate = (leavers / n) * 100.0
            ci_low, ci_high = wilson_interval(leavers, n, confidence=0.95)
            wf_share = (n / total_df_n) * 100.0
            leaver_share = (
                (leavers / total_df_leavers) * 100.0 if total_df_leavers > 0 else 0.0
            )
            diff_pp = rate - overall_rate_pct
            ratio = (rate / overall_rate_pct) if overall_rate_pct > 0 else np.nan

            rows.append(
                {
                    dim1: c1,
                    dim2: c2,
                    "n": n,
                    "leavers": leavers,
                    "retained": retained,
                    "attrition_rate_pct": round(rate, ROUND_DIGITS),
                    "ci95_low_pct": round(ci_low * 100.0, ROUND_DIGITS),
                    "ci95_high_pct": round(ci_high * 100.0, ROUND_DIGITS),
                    "share_of_workforce_pct": round(wf_share, ROUND_DIGITS),
                    "share_of_leavers_pct": round(leaver_share, ROUND_DIGITS),
                    "diff_vs_overall_pp": round(diff_pp, ROUND_DIGITS),
                    "ratio_vs_overall": round(ratio, ROUND_DIGITS) if not np.isnan(ratio) else np.nan,
                    "low_n_flag": n < min_n,
                }
            )

    return pd.DataFrame(rows, columns=columns)


def concentration_table(
    df: pd.DataFrame,
    dimensions: List[str],
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Compute leaver concentration table across one or more dimensions."""
    total_leavers = int((df[target_col] == 1).sum())
    total_n = len(df)

    rows: List[Dict[str, Any]] = []

    for dim in dimensions:
        rt = rate_table(df, dim, target_col=target_col)
        # Filter out 0 leavers or sort descending by leavers
        rt_sorted = rt.sort_values(by=["leavers", "n"], ascending=[False, False]).copy()

        cum_leavers = 0
        cum_workforce = 0

        for _, row in rt_sorted.iterrows():
            leavers = int(row["leavers"])
            n = int(row["n"])
            cum_leavers += leavers
            cum_workforce += n

            cum_leaver_share = (cum_leavers / total_leavers * 100.0) if total_leavers > 0 else 0.0
            cum_wf_share = (cum_workforce / total_n * 100.0) if total_n > 0 else 0.0

            rows.append(
                {
                    "dimension": dim,
                    "segment": row["segment"],
                    "leavers": leavers,
                    "n": n,
                    "attrition_rate_pct": row["attrition_rate_pct"],
                    "share_of_leavers_pct": row["share_of_leavers_pct"],
                    "cumulative_share_of_leavers_pct": round(cum_leaver_share, ROUND_DIGITS),
                    "share_of_workforce_pct": row["share_of_workforce_pct"],
                    "cumulative_share_of_workforce_pct": round(cum_wf_share, ROUND_DIGITS),
                }
            )

    return pd.DataFrame(rows)


def funnel_table(
    df: pd.DataFrame,
    dimensions: List[str],
    p0: float,
    levels: Tuple[float, float] = FUNNEL_LEVELS,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Compute exact binomial control limits around baseline proportion p0."""
    rows: List[Dict[str, Any]] = []

    level1, level2 = levels

    for dim in dimensions:
        rt = rate_table(df, dim, target_col=target_col)
        for _, row in rt.iterrows():
            n = int(row["n"])
            rate = row["attrition_rate_pct"]
            leavers = int(row["leavers"])

            if n == 0 or np.isnan(rate):
                low1, up1 = np.nan, np.nan
                low2, up2 = np.nan, np.nan
                status1, status2 = "Undefined", "Undefined"
            else:
                low1, up1 = exact_binomial_limits(n, p0, level=level1)
                low2, up2 = exact_binomial_limits(n, p0, level=level2)

                p_obs = rate / 100.0
                # 95% status
                if p_obs > up1:
                    status1 = "Above"
                elif p_obs < low1:
                    status1 = "Below"
                else:
                    status1 = "Inside"

                # 99.8% status
                if p_obs > up2:
                    status2 = "Above"
                elif p_obs < low2:
                    status2 = "Below"
                else:
                    status2 = "Inside"

            rows.append(
                {
                    "dimension": dim,
                    "segment": row["segment"],
                    "n": n,
                    "leavers": leavers,
                    "attrition_rate_pct": rate,
                    f"lower_{int(level1*100)}_pct": round(low1 * 100.0, ROUND_DIGITS) if not np.isnan(low1) else np.nan,
                    f"upper_{int(level1*100)}_pct": round(up1 * 100.0, ROUND_DIGITS) if not np.isnan(up1) else np.nan,
                    f"status_{int(level1*100)}": status1,
                    f"lower_{round(level2*100, 1)}_pct": round(low2 * 100.0, ROUND_DIGITS) if not np.isnan(low2) else np.nan,
                    f"upper_{round(level2*100, 1)}_pct": round(up2 * 100.0, ROUND_DIGITS) if not np.isnan(up2) else np.nan,
                    f"status_{round(level2*100, 1)}": status2,
                }
            )

    return pd.DataFrame(rows)


def sensitivity_table(
    df: pd.DataFrame,
    specs: List[Dict[str, Any]],
    target_col: str = TARGET_COLUMN,
    min_n: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Evaluate subgroup rate sensitivity across alternative cut-point specifications."""
    rows: List[Dict[str, Any]] = []

    for spec in specs:
        family = spec["family"]
        spec_name = spec["spec_name"]
        stype = spec["type"]
        col = spec["column"]

        # Filter dataset if specified (e.g. PromotionAnalysisCohort)
        if "filter_col" in spec:
            f_col = spec["filter_col"]
            f_val = spec["filter_val"]
            sub_df = df[df[f_col] == f_val].copy()
        else:
            sub_df = df.copy()

        temp_col = f"_sens_{family}"
        if stype == "threshold":
            t = spec["threshold"]
            sub_df[temp_col] = np.where(
                sub_df[col] <= t, f"<= {t}", f"> {t}"
            )
        elif stype == "bins":
            bins = spec["bins"]
            labels = spec["labels"]
            sub_df[temp_col] = pd.cut(
                sub_df[col], bins=bins, labels=labels, right=True
            ).astype(str)
        elif stype == "quantiles":
            q = spec["q"]
            # Compute empirical quartiles from the data
            _, bin_edges = pd.qcut(sub_df[col], q=q, retbins=True, duplicates="drop")
            # Format labels as ranges
            q_labels = []
            for i in range(len(bin_edges) - 1):
                left_b = int(bin_edges[i])
                right_b = int(bin_edges[i + 1])
                q_labels.append(f"Q{i+1}: {left_b+1 if i>0 else left_b}-{right_b}")
            sub_df[temp_col] = pd.qcut(
                sub_df[col], q=q, labels=q_labels, duplicates="drop"
            ).astype(str)

        rt = rate_table(sub_df, temp_col, min_n=min_n, target_col=target_col)

        # Identify highest and lowest rate bands within this spec
        valid_rates = rt.dropna(subset=["attrition_rate_pct"])
        if len(valid_rates) > 0:
            max_rate = valid_rates["attrition_rate_pct"].max()
            min_rate = valid_rates["attrition_rate_pct"].min()
        else:
            max_rate, min_rate = np.nan, np.nan

        for _, r in rt.iterrows():
            rate = r["attrition_rate_pct"]
            is_max = (rate == max_rate) if not np.isnan(rate) else False
            is_min = (rate == min_rate) if not np.isnan(rate) else False

            rows.append(
                {
                    "family": family,
                    "spec_name": spec_name,
                    "band": r["segment"],
                    "n": int(r["n"]),
                    "leavers": int(r["leavers"]),
                    "attrition_rate_pct": rate,
                    "ci95_low_pct": r["ci95_low_pct"],
                    "ci95_high_pct": r["ci95_high_pct"],
                    "low_n_flag": r["low_n_flag"],
                    "is_highest_rate_in_spec": is_max,
                    "is_lowest_rate_in_spec": is_min,
                }
            )

    return pd.DataFrame(rows)


def numeric_summary_by_outcome(
    df: pd.DataFrame,
    numeric_cols: List[str],
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Compute distributional statistics for continuous metrics stratified by outcome."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    rows: List[Dict[str, Any]] = []

    groups = [
        ("Overall", df),
        ("Retained", df[df[target_col] == 0]),
        ("Left", df[df[target_col] == 1]),
    ]

    for col in numeric_cols:
        if col not in df.columns:
            raise ValueError(f"Numeric column '{col}' not found in DataFrame.")

        for label, group_df in groups:
            s = group_df[col].dropna()
            n = len(s)
            if n == 0:
                rows.append(
                    {
                        "metric": col,
                        "outcome": label,
                        "n": 0,
                        "mean": np.nan,
                        "std": np.nan,
                        "median": np.nan,
                        "q1": np.nan,
                        "q3": np.nan,
                        "iqr": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                    }
                )
                continue

            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            rows.append(
                {
                    "metric": col,
                    "outcome": label,
                    "n": n,
                    "mean": round(float(s.mean()), ROUND_DIGITS),
                    "std": round(float(s.std(ddof=1)), ROUND_DIGITS) if n > 1 else 0.0,
                    "median": round(float(s.median()), ROUND_DIGITS),
                    "q1": round(q1, ROUND_DIGITS),
                    "q3": round(q3, ROUND_DIGITS),
                    "iqr": round(q3 - q1, ROUND_DIGITS),
                    "min": round(float(s.min()), ROUND_DIGITS),
                    "max": round(float(s.max()), ROUND_DIGITS),
                }
            )

    return pd.DataFrame(rows)


def monthly_income_by_joblevel(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Compute MonthlyIncome distribution stratified by JobLevel and departure outcome."""
    rows: List[Dict[str, Any]] = []

    job_levels = sorted(df["JobLevel"].unique())
    outcomes = [
        ("Overall", df),
        ("Retained", df[df[target_col] == 0]),
        ("Left", df[df[target_col] == 1]),
    ]

    for lvl in job_levels:
        for out_name, out_df in outcomes:
            s = out_df[out_df["JobLevel"] == lvl]["MonthlyIncome"].dropna()
            n = len(s)
            if n == 0:
                continue
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            rows.append(
                {
                    "job_level": lvl,
                    "outcome": out_name,
                    "n": n,
                    "median": round(float(s.median()), ROUND_DIGITS),
                    "q1": round(q1, ROUND_DIGITS),
                    "q3": round(q3, ROUND_DIGITS),
                    "iqr": round(q3 - q1, ROUND_DIGITS),
                    "mean": round(float(s.mean()), ROUND_DIGITS),
                    "std": round(float(s.std(ddof=1)), ROUND_DIGITS) if n > 1 else 0.0,
                }
            )

    return pd.DataFrame(rows)


def hotspot_candidates(
    two_way_tables: Dict[str, pd.DataFrame] | List[pd.DataFrame],
    overall_rate_pct: float,
    min_n: int = HOTSPOT_MIN_N,
    top_k: int = HOTSPOT_TOP_K,
) -> Tuple[pd.DataFrame, int]:
    """Screen interaction cells for elevated attrition rates where Wilson lower bound exceeds overall baseline.

    Parameters
    ----------
    two_way_tables : dict or list of pd.DataFrame
        Collection of computed two-way interaction tables.
    overall_rate_pct : float
        Overall population attrition rate in percent (e.g. 16.12).
    min_n : int, default 30
        Minimum cell sample size threshold.
    top_k : int, default 15
        Maximum number of candidate cells to return.

    Returns
    -------
    tuple of (pd.DataFrame, int)
        (candidates_dataframe, total_cells_scanned)
    """
    if isinstance(two_way_tables, dict):
        table_list = list(two_way_tables.values())
    else:
        table_list = two_way_tables

    all_cells: List[Dict[str, Any]] = []
    total_cells = 0

    for table in table_list:
        cols = list(table.columns)
        dim1 = cols[0]
        dim2 = cols[1]

        for _, r in table.iterrows():
            total_cells += 1
            n = int(r["n"])
            rate = r["attrition_rate_pct"]
            ci_low = r["ci95_low_pct"]
            ci_high = r["ci95_high_pct"]
            leavers = int(r["leavers"])

            if np.isnan(rate):
                continue

            # Eligibility rule: n >= min_n AND Wilson lower bound > overall baseline
            if (n >= min_n) and (ci_low > overall_rate_pct):
                all_cells.append(
                    {
                        "interaction": f"{dim1} x {dim2}",
                        "subgroup": f"{dim1}={r[dim1]} & {dim2}={r[dim2]}",
                        "n": n,
                        "leavers": leavers,
                        "attrition_rate_pct": rate,
                        "ci95_low_pct": ci_low,
                        "ci95_high_pct": ci_high,
                        "diff_vs_overall_pp": r["diff_vs_overall_pp"],
                        "ratio_vs_overall": r["ratio_vs_overall"],
                    }
                )

    candidates_df = pd.DataFrame(all_cells)
    if len(candidates_df) > 0:
        candidates_df = candidates_df.sort_values(
            by=["attrition_rate_pct", "n"], ascending=[False, False]
        ).head(top_k).reset_index(drop=True)
    else:
        candidates_df = pd.DataFrame(
            columns=[
                "interaction",
                "subgroup",
                "n",
                "leavers",
                "attrition_rate_pct",
                "ci95_low_pct",
                "ci95_high_pct",
                "diff_vs_overall_pp",
                "ratio_vs_overall",
            ]
        )

    return (candidates_df, total_cells)
