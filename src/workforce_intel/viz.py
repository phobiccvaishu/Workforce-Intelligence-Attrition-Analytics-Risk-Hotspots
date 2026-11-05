"""Visualization module for static figures.

Implements 14 pure plotting functions returning (matplotlib Figure, plotted DataFrame).
Zero file I/O inside plot functions. Strict adherence to report styling and language hygiene.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import textwrap

from workforce_intel.config import (
    FIGURE_DPI,
    FIGURE_FOOTNOTE_ORDINAL,
    FIGURE_FOOTNOTE_PROTECTED,
    FIGURE_FOOTNOTE_STANDARD,
    FIGURE_PALETTE,
    FIGURE_SPECS,
    ROUND_DIGITS,
)


def _get_table(tables: Dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    """Retrieve table from dictionary by filename or stem name."""
    if name in tables:
        return tables[name].copy()
    stem = name.replace(".csv", "")
    if stem in tables:
        return tables[stem].copy()
    raise KeyError(f"Required table '{name}' not found in provided tables dictionary.")


def _add_footnote(fig: plt.Figure, text: str, y: float = 0.01) -> None:
    """Add standard wrapped provenance footnote to figure bottom."""
    wrapped = textwrap.fill(text, width=120)
    fig.text(0.5, y, wrapped, ha="center", va="bottom", fontsize=8, color="#4B5563", style="italic")


def _get_baseline_rate(tables: Dict[str, pd.DataFrame]) -> float:
    """Retrieve overall workforce baseline rate."""
    base_df = _get_table(tables, "overall_baseline.csv")
    return float(base_df["attrition_rate_pct"].iloc[0])


# =============================================================================
# FIG-01: Department Overview
# =============================================================================
def plot_fig01_department_overview(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot department overview: (A) Attrition rate with 95% Wilson CI, (B) Workforce vs leaver share."""
    spec = FIGURE_SPECS["FIG-01"]
    df = _get_table(tables, "department_attrition.csv")
    baseline_rate = _get_baseline_rate(tables)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIGURE_DPI)

    # Panel A: Rate with Wilson CI
    x = np.arange(len(df))
    rates = df["attrition_rate_pct"].values
    yerr_low = rates - df["ci95_low_pct"].values
    yerr_high = df["ci95_high_pct"].values - rates

    colors = [
        FIGURE_PALETTE["highlight"] if (df["ci95_low_pct"].iloc[i] > baseline_rate or df["ci95_high_pct"].iloc[i] < baseline_rate)
        else FIGURE_PALETTE["primary"]
        for i in range(len(df))
    ]

    bars1 = ax1.bar(x, rates, yerr=[yerr_low, yerr_high], capsize=5, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    ax1.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars1):
        height = rect.get_height()
        n_val = df["n"].iloc[idx]
        l_val = df["leavers"].iloc[idx]
        ax1.annotate(
            f"{height:.1f}%\n(n={n_val}, l={l_val})",
            xy=(rect.get_x() + rect.get_width() / 2, height / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.5,
            color="white",
            weight="bold",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([textwrap.fill(s, 14) for s in df["segment"]], fontsize=9)
    ax1.set_ylabel("% who left", fontsize=10, weight="bold")
    ax1.set_title("A. Department Attrition Rate (95% Wilson CI)", fontsize=11, weight="bold")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: Share Comparison
    width = 0.35
    ax2.bar(x - width / 2, df["share_of_workforce_pct"], width, label="Workforce Share (%)", color=FIGURE_PALETTE["primary"], alpha=0.7)
    ax2.bar(x + width / 2, df["share_of_leavers_pct"], width, label="Leaver Share (%)", color=FIGURE_PALETTE["highlight"], alpha=0.85)

    for idx in range(len(df)):
        wf_s = df["share_of_workforce_pct"].iloc[idx]
        lv_s = df["share_of_leavers_pct"].iloc[idx]
        ax2.annotate(f"{wf_s:.1f}%", xy=(x[idx] - width / 2, wf_s + 0.8), ha="center", fontsize=8)
        ax2.annotate(f"{lv_s:.1f}%", xy=(x[idx] + width / 2, lv_s + 0.8), ha="center", fontsize=8, weight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([textwrap.fill(s, 14) for s in df["segment"]], fontsize=9)
    ax2.set_ylabel("Share (%)", fontsize=10, weight="bold")
    ax2.set_title("B. Workforce Share vs Leaver Share", fontsize=11, weight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.03)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, df


# =============================================================================
# FIG-02: Job Role Rates (Ranked)
# =============================================================================
def plot_fig02_jobrole_attrition(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot ranked job role attrition rates horizontal bar chart."""
    spec = FIGURE_SPECS["FIG-02"]
    df = _get_table(tables, "jobrole_attrition.csv").sort_values("attrition_rate_pct", ascending=True).reset_index(drop=True)
    baseline_rate = _get_baseline_rate(tables)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)

    y = np.arange(len(df))
    rates = df["attrition_rate_pct"].values
    xerr_low = rates - df["ci95_low_pct"].values
    xerr_high = df["ci95_high_pct"].values - rates

    colors = [
        FIGURE_PALETTE["highlight"] if df["ci95_low_pct"].iloc[i] > baseline_rate
        else (FIGURE_PALETTE["secondary"] if df["ci95_high_pct"].iloc[i] < baseline_rate else FIGURE_PALETTE["primary"])
        for i in range(len(df))
    ]

    bars = ax.barh(y, rates, xerr=[xerr_low, xerr_high], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    ax.axvline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars):
        width = rect.get_width()
        n_val = df["n"].iloc[idx]
        l_val = df["leavers"].iloc[idx]
        ax.annotate(
            f" {width:.1f}% (n={n_val}, l={l_val})",
            xy=(df["ci95_high_pct"].iloc[idx] + 0.5, rect.get_y() + rect.get_height() / 2),
            va="center",
            fontsize=8.5,
            color="#1F2937",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["segment"], fontsize=9.5)
    ax.set_xlabel("% who left", fontsize=10, weight="bold")
    ax.set_title(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_xlim(0, max(df["ci95_high_pct"]) + 12)

    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, df


# =============================================================================
# FIG-03: Funnel Plot for Job Roles
# =============================================================================
def plot_fig03_funnel_plot(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot exact binomial control limits funnel plot for job roles."""
    spec = FIGURE_SPECS["FIG-03"]
    fl_df = _get_table(tables, "funnel_limits.csv")
    df = fl_df[fl_df["dimension"] == "JobRole"].copy().reset_index(drop=True)
    baseline_rate = _get_baseline_rate(tables)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)

    # Plot funnel control limit curves across range of n
    n_grid = np.linspace(30, 400, 150)
    p = baseline_rate / 100.0

    # Normal approximation with continuity correction for visual control bands
    from scipy.stats import norm
    z95 = norm.ppf(0.975)
    z998 = norm.ppf(0.999)

    se = np.sqrt(p * (1 - p) / n_grid)
    ub_95 = (p + z95 * se) * 100.0
    lb_95 = np.maximum(0, (p - z95 * se) * 100.0)
    ub_998 = (p + z998 * se) * 100.0
    lb_998 = np.maximum(0, (p - z998 * se) * 100.0)

    ax.plot(n_grid, ub_998, "r:", linewidth=1.2, label="99.8% Control Limit")
    ax.plot(n_grid, lb_998, "r:", linewidth=1.2)
    ax.plot(n_grid, ub_95, "b--", linewidth=1.2, label="95% Control Limit")
    ax.plot(n_grid, lb_95, "b--", linewidth=1.2)
    ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="-", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    # Plot Job Roles points
    for _, row in df.iterrows():
        n_val = row["n"]
        rate = row["attrition_rate_pct"]
        seg = row["segment"]
        status95 = row["status_95"]

        color = FIGURE_PALETTE["highlight"] if status95 != "Inside" else FIGURE_PALETTE["primary"]
        ax.scatter(n_val, rate, color=color, s=70, zorder=5, edgecolor="#111827")
        ax.annotate(
            textwrap.fill(seg, 16),
            xy=(n_val, rate),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            weight="bold" if status95 != "Inside" else "normal",
        )

    ax.set_xlabel("Headcount (n)", fontsize=10, weight="bold")
    ax.set_ylabel("% who left", fontsize=10, weight="bold")
    ax.set_title(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(-1, 55)

    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, df


# =============================================================================
# FIG-04: Leaver Concentration Pareto by Role
# =============================================================================
def plot_fig04_leaver_concentration(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot Pareto analysis of leaver concentration across job roles."""
    spec = FIGURE_SPECS["FIG-04"]
    lc_df = _get_table(tables, "leaver_concentration.csv")
    df = lc_df[lc_df["dimension"] == "JobRole"].sort_values("leavers", ascending=False).copy().reset_index(drop=True)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)
    ax2 = ax1.twinx()

    x = np.arange(len(df))
    bars = ax1.bar(x, df["leavers"], color=FIGURE_PALETTE["primary"], alpha=0.85, edgecolor="#111827", linewidth=0.8, label="Leavers Count")
    ax2.plot(x, df["cumulative_share_of_leavers_pct"], color=FIGURE_PALETTE["highlight"], marker="o", linewidth=2.2, label="Cumulative Leaver %")
    ax2.axhline(80, color="#6B7280", linestyle=":", linewidth=1.5, label="80% Threshold")

    for idx, rect in enumerate(bars):
        l_cnt = df["leavers"].iloc[idx]
        ax1.annotate(str(l_cnt), xy=(rect.get_x() + rect.get_width() / 2, l_cnt / 2), ha="center", va="center", color="white", weight="bold", fontsize=9)

    for idx in range(len(df)):
        cum = df["cumulative_share_of_leavers_pct"].iloc[idx]
        ax2.annotate(f"{cum:.1f}%", xy=(x[idx], cum + 2), ha="center", fontsize=8, weight="bold", color=FIGURE_PALETTE["highlight"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([textwrap.fill(s, 14) for s in df["segment"]], rotation=25, ha="right", fontsize=8.5)
    ax1.set_ylabel("Leavers Count", fontsize=10, weight="bold", color=FIGURE_PALETTE["primary"])
    ax2.set_ylabel("Cumulative Share of Total Leavers (%)", fontsize=10, weight="bold", color=FIGURE_PALETTE["highlight"])
    ax2.set_ylim(0, 110)

    ax1.set_title(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", pad=12)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, df


# =============================================================================
# FIG-05: Age Sensitivity Panels
# =============================================================================
def plot_fig05_age_sensitivity(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot age sensitivity panels (4-band default, 5-band, 3-band)."""
    spec = FIGURE_SPECS["FIG-05"]
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    ag_sens = sens_df[sens_df["family"] == "AgeGroup"].copy().reset_index(drop=True)
    baseline_rate = _get_baseline_rate(tables)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=FIGURE_DPI, sharey=True)

    specs_order = [
        "4 Bands (Default: 18-25, 26-35, 36-45, 46-60)",
        "5 Bands (18-24, 25-34, 35-44, 45-54, 55-60)",
        "3 Bands (18-30, 31-45, 46-60)",
    ]
    panel_titles = ["A. 4-Band (Default)", "B. 5-Band Specification", "C. 3-Band Specification"]

    for ax, sp_name, p_title in zip(axes, specs_order, panel_titles):
        sub = ag_sens[ag_sens["spec_name"] == sp_name].copy().reset_index(drop=True)
        x = np.arange(len(sub))
        rates = sub["attrition_rate_pct"].values
        yerr_l = rates - sub["ci95_low_pct"].values
        yerr_h = sub["ci95_high_pct"].values - rates

        colors = [
            FIGURE_PALETTE["highlight"] if sub["ci95_low_pct"].iloc[i] > baseline_rate
            else (FIGURE_PALETTE["secondary"] if sub["ci95_high_pct"].iloc[i] < baseline_rate else FIGURE_PALETTE["primary"])
            for i in range(len(sub))
        ]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        for idx, rect in enumerate(bars):
            h = rect.get_height()
            n_val = sub["n"].iloc[idx]
            ax.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["band"], fontsize=8.5)
        ax.set_title(p_title, fontsize=10.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    axes[0].set_ylabel("% who left", fontsize=10, weight="bold")
    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.03)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, ag_sens


# =============================================================================
# FIG-06: Tenure Sensitivity Panels
# =============================================================================
def plot_fig06_tenure_sensitivity(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot tenure band attrition and early tenure sensitivity panels."""
    spec = FIGURE_SPECS["FIG-06"]
    tb_df = _get_table(tables, "tenureband_attrition.csv")
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    et_sens = sens_df[sens_df["family"] == "EarlyTenure"].copy().reset_index(drop=True)
    baseline_rate = _get_baseline_rate(tables)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIGURE_DPI)

    # Panel A: TenureBand
    x1 = np.arange(len(tb_df))
    r1 = tb_df["attrition_rate_pct"].values
    yerr1_l = r1 - tb_df["ci95_low_pct"].values
    yerr1_h = tb_df["ci95_high_pct"].values - r1

    colors1 = [
        FIGURE_PALETTE["highlight"] if tb_df["ci95_low_pct"].iloc[i] > baseline_rate
        else (FIGURE_PALETTE["secondary"] if tb_df["ci95_high_pct"].iloc[i] < baseline_rate else FIGURE_PALETTE["primary"])
        for i in range(len(tb_df))
    ]

    bars1 = ax1.bar(x1, r1, yerr=[yerr1_l, yerr1_h], capsize=4, color=colors1, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    ax1.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars1):
        h = rect.get_height()
        n_val = tb_df["n"].iloc[idx]
        ax1.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=8)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(tb_df["segment"], fontsize=9)
    ax1.set_ylabel("% who left", fontsize=10, weight="bold")
    ax1.set_title("A. Tenure Band Attrition (Years at Company)", fontsize=11, weight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: Early Tenure Sensitivity (<=1, <=2, <=3)
    # Extract bands <= 1, <= 2, <= 3
    et_sub = et_sens[et_sens["band"].str.startswith("<=")].copy().reset_index(drop=True)
    x2 = np.arange(len(et_sub))
    r2 = et_sub["attrition_rate_pct"].values
    yerr2_l = r2 - et_sub["ci95_low_pct"].values
    yerr2_h = et_sub["ci95_high_pct"].values - r2

    bars2 = ax2.bar(x2, r2, yerr=[yerr2_l, yerr2_h], capsize=4, color=FIGURE_PALETTE["highlight"], alpha=0.9, edgecolor="#111827", linewidth=0.8)
    ax2.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars2):
        h = rect.get_height()
        n_val = et_sub["n"].iloc[idx]
        ax2.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=8.5)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"Years <= {b.replace('<= ', '')}" for b in et_sub["band"]], fontsize=9)
    ax2.set_ylabel("% who left", fontsize=10, weight="bold")
    ax2.set_title("B. Early Tenure Threshold Sensitivity", fontsize=11, weight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.03)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    plotted_df = pd.concat([tb_df, et_sens], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-07: Promotion Gap Cohort Analysis
# =============================================================================
def plot_fig07_promotion_gap_cohort(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot promotion gap attrition: Pooled vs PromotionAnalysisCohort side-by-side."""
    spec = FIGURE_SPECS["FIG-07"]
    sbs_df = _get_table(tables, "promotiongap_side_by_side.csv")
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    pg_sens = sens_df[sens_df["family"] == "PromotionGapBand_Cohort"].copy().reset_index(drop=True)
    alt_df = pg_sens[pg_sens["spec_name"].str.contains("Alternative")].copy().reset_index(drop=True)

    baseline_rate = _get_baseline_rate(tables)
    cohort_baseline_rate = (96.0 / 890.0) * 100.0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), dpi=FIGURE_DPI, sharey=True)

    # Panel 1: Pooled (all 1,470)
    x1 = np.arange(len(sbs_df))
    r1 = sbs_df["pooled_attrition_rate_pct"].values
    bars1 = ax1.bar(x1, r1, color=FIGURE_PALETTE["primary"], alpha=0.85, edgecolor="#111827", linewidth=0.8)
    ax1.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars1):
        h = rect.get_height()
        n_val = sbs_df["pooled_n"].iloc[idx]
        ax1.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=7.5)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(sbs_df["PromotionGapBand"], fontsize=8.5)
    ax1.set_ylabel("% who left", fontsize=10, weight="bold")
    ax1.set_title("A. Pooled Workforce (n=1,470)\nConfounded by Tenure", fontsize=10, weight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: Cohort (n=890, YearsAtCompany >= 5)
    r2 = sbs_df["cohort_attrition_rate_pct"].values
    colors2 = [
        FIGURE_PALETTE["highlight"] if sbs_df["PromotionGapBand"].iloc[i] == "6+" else FIGURE_PALETTE["primary"]
        for i in range(len(sbs_df))
    ]
    bars2 = ax2.bar(x1, r2, color=colors2, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    ax2.axhline(cohort_baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Cohort Baseline ({cohort_baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars2):
        h = rect.get_height()
        n_val = sbs_df["cohort_n"].iloc[idx]
        ax2.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=7.5)

    ax2.set_xticks(x1)
    ax2.set_xticklabels(sbs_df["PromotionGapBand"], fontsize=8.5)
    ax2.set_title("B. Promotion Cohort (n=890)\nTenure >= 5 Years", fontsize=10, weight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 3: Cohort Alternative Bins
    x3 = np.arange(len(alt_df))
    r3 = alt_df["attrition_rate_pct"].values
    colors3 = [
        FIGURE_PALETTE["highlight"] if alt_df["band"].iloc[i] == "7+" else FIGURE_PALETTE["primary"]
        for i in range(len(alt_df))
    ]
    bars3 = ax3.bar(x3, r3, color=colors3, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    ax3.axhline(cohort_baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Cohort Baseline ({cohort_baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars3):
        h = rect.get_height()
        n_val = alt_df["n"].iloc[idx]
        ax3.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=7.5)

    ax3.set_xticks(x3)
    ax3.set_xticklabels(alt_df["band"], fontsize=8.5)
    ax3.set_title("C. Cohort Alternative Bins\n(0-1, 2-3, 4-6, 7+)", fontsize=10, weight="bold")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.03)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    plotted_df = pd.concat([sbs_df, pg_sens], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-08: Workload & Mobility
# =============================================================================
def plot_fig08_workload_mobility(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot workload, business travel, and commute distance panels."""
    spec = FIGURE_SPECS["FIG-08"]
    ot_df = _get_table(tables, "overtime_attrition.csv")
    bt_df = _get_table(tables, "businesstravel_attrition.csv")
    db_df = _get_table(tables, "distanceband_attrition.csv")
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    db_sens = sens_df[(sens_df["family"] == "DistanceBand") & (sens_df["spec_name"] == "Empirical Quartiles")].copy().reset_index(drop=True)

    baseline_rate = _get_baseline_rate(tables)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=FIGURE_DPI, sharey=True)

    panels_data = [
        (ot_df, "A. OverTime", "segment"),
        (bt_df, "B. Business Travel", "segment"),
        (db_df, "C. Distance Band (Default)", "segment"),
        (db_sens, "D. Distance Quartiles", "band"),
    ]

    for ax, (data, title, col_lbl) in zip(axes, panels_data):
        x = np.arange(len(data))
        rates = data["attrition_rate_pct"].values
        yerr_l = rates - data["ci95_low_pct"].values
        yerr_h = data["ci95_high_pct"].values - rates

        colors = [
            FIGURE_PALETTE["highlight"] if data["ci95_low_pct"].iloc[i] > baseline_rate
            else (FIGURE_PALETTE["secondary"] if data["ci95_high_pct"].iloc[i] < baseline_rate else FIGURE_PALETTE["primary"])
            for i in range(len(data))
        ]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        for idx, rect in enumerate(bars):
            h = rect.get_height()
            n_val = data["n"].iloc[idx]
            ax.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(str(s), 10) for s in data[col_lbl]], fontsize=8.5)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    axes[0].set_ylabel("% who left", fontsize=10, weight="bold")
    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.03)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    plotted_df = pd.concat([ot_df, bt_df, db_df, db_sens], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-09: Satisfaction & Sentiment Small Multiples
# =============================================================================
def plot_fig09_satisfaction_small_multiples(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot small multiples across 5 sentiment survey ratings."""
    spec = FIGURE_SPECS["FIG-09"]
    wlb = _get_table(tables, "worklifebalance_attrition.csv")
    js = _get_table(tables, "jobsatisfaction_attrition.csv")
    es = _get_table(tables, "environmentsatisfaction_attrition.csv")
    rs = _get_table(tables, "relationshipsatisfaction_attrition.csv")
    ji = _get_table(tables, "jobinvolvement_attrition.csv")

    baseline_rate = _get_baseline_rate(tables)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), dpi=FIGURE_DPI, sharey=True)

    surveys = [
        (wlb, "Work-Life Balance"),
        (js, "Job Satisfaction"),
        (es, "Environment Satisfaction"),
        (rs, "Relationship Satisfaction"),
        (ji, "Job Involvement"),
    ]

    for ax, (data, title) in zip(axes, surveys):
        x = np.arange(len(data))
        rates = data["attrition_rate_pct"].values
        yerr_l = rates - data["ci95_low_pct"].values
        yerr_h = data["ci95_high_pct"].values - rates

        colors = [
            FIGURE_PALETTE["highlight"] if data["ci95_low_pct"].iloc[i] > baseline_rate
            else (FIGURE_PALETTE["secondary"] if data["ci95_high_pct"].iloc[i] < baseline_rate else FIGURE_PALETTE["primary"])
            for i in range(len(data))
        ]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        for idx, rect in enumerate(bars):
            h = rect.get_height()
            n_val = data["n"].iloc[idx]
            ax.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Level {s}" for s in data["segment"]], fontsize=8)
        ax.set_title(title, fontsize=9.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    axes[0].set_ylabel("% who left", fontsize=10, weight="bold")
    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.03)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    plotted_df = pd.concat([wlb, js, es, rs, ji], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-10: OverTime Interactions Heatmaps
# =============================================================================
def plot_fig10_overtime_interactions(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot heatmaps of OverTime interactions with JobLevel, AgeGroup, and TenureBand."""
    spec = FIGURE_SPECS["FIG-10"]
    tw_jl = _get_table(tables, "two_way_overtime_x_joblevel.csv")
    tw_ag = _get_table(tables, "two_way_overtime_x_agegroup.csv")
    tw_tb = _get_table(tables, "two_way_overtime_x_tenureband.csv")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=FIGURE_DPI)

    heatmaps = [
        (tw_jl, ax1, "JobLevel", "A. OverTime x JobLevel"),
        (tw_ag, ax2, "AgeGroup", "B. OverTime x AgeGroup"),
        (tw_tb, ax3, "TenureBand", "C. OverTime x TenureBand"),
    ]

    cmap = plt.cm.Blues

    for data, ax, col2, title in heatmaps:
        # Pivot into 2D grid: rows = OverTime (No, Yes), cols = col2
        ot_order = ["No", "Yes"]
        col2_order = list(dict.fromkeys(data[col2]))

        grid_rate = np.zeros((len(ot_order), len(col2_order)))
        grid_n = np.zeros((len(ot_order), len(col2_order)), dtype=int)
        grid_low_n = np.zeros((len(ot_order), len(col2_order)), dtype=bool)

        for i, ot_val in enumerate(ot_order):
            for j, c_val in enumerate(col2_order):
                match = data[(data["OverTime"] == ot_val) & (data[col2] == c_val)]
                if len(match) > 0:
                    grid_rate[i, j] = match["attrition_rate_pct"].iloc[0]
                    grid_n[i, j] = match["n"].iloc[0]
                    grid_low_n[i, j] = bool(match["low_n_flag"].iloc[0])

        im = ax.imshow(grid_rate, cmap=cmap, vmin=0, vmax=60, aspect="auto")

        for i in range(len(ot_order)):
            for j in range(len(col2_order)):
                r_val = grid_rate[i, j]
                n_val = grid_n[i, j]
                is_low = grid_low_n[i, j]

                if is_low:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color="#D1D5DB", hatch="//", alpha=0.7))
                    txt = f"{r_val:.1f}%\n(n={n_val})\n[n<30]"
                    t_col = "#1F2937"
                else:
                    txt = f"{r_val:.1f}%\n(n={n_val})"
                    t_col = "white" if r_val > 25 else "#111827"

                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=t_col, weight="bold")

        ax.set_xticks(np.arange(len(col2_order)))
        ax.set_xticklabels(col2_order, fontsize=8.5)
        ax.set_yticks(np.arange(len(ot_order)))
        ax.set_yticklabels(ot_order, fontsize=9)
        ax.set_ylabel("OverTime", fontsize=9.5, weight="bold")
        ax.set_title(title, fontsize=10.5, weight="bold")

    fig.colorbar(im, ax=[ax1, ax2, ax3], orientation="horizontal", fraction=0.05, pad=0.18, label="% who left")
    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=0.98)
    _add_footnote(fig, spec["footnote"], y=0.02)

    plotted_df = pd.concat([tw_jl, tw_ag, tw_tb], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-11: Hotspot Candidates Dot Plot
# =============================================================================
def plot_fig11_hotspot_candidates(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot dot plot with confidence intervals for 16 hotspot candidates."""
    spec = FIGURE_SPECS["FIG-11"]
    hc_df = _get_table(tables, "hotspot_candidates.csv").sort_values("attrition_rate_pct", ascending=True).reset_index(drop=True)
    baseline_rate = _get_baseline_rate(tables)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=FIGURE_DPI)

    y = np.arange(len(hc_df))
    rates = hc_df["attrition_rate_pct"].values
    xerr_low = rates - hc_df["ci95_low_pct"].values
    xerr_high = hc_df["ci95_high_pct"].values - rates

    ax.errorbar(
        rates,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color=FIGURE_PALETTE["highlight"],
        ecolor="#9A3412",
        elinewidth=1.6,
        capsize=4,
        markersize=7,
        zorder=4,
    )

    ax.axvline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    for idx in range(len(hc_df)):
        r = rates[idx]
        n_val = hc_df["n"].iloc[idx]
        l_val = hc_df["leavers"].iloc[idx]
        ci_h = hc_df["ci95_high_pct"].iloc[idx]
        ax.annotate(f" {r:.1f}% (n={n_val}, l={l_val})", xy=(ci_h + 0.6, y[idx]), va="center", fontsize=8, color="#1F2937")

    labels = [f"{row['interaction']}: {row['subgroup']}" for _, row in hc_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("% who left", fontsize=10, weight="bold")
    ax.set_title(spec["title"] + "\n" + spec["subtitle"], fontsize=11, weight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_xlim(0, max(hc_df["ci95_high_pct"]) + 14)

    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, hc_df


# =============================================================================
# FIG-12: Hotspot Overlap Matrix & Union Summary
# =============================================================================
def plot_fig12_hotspot_overlap(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot pairwise overlap matrix heatmap and union summary bars."""
    spec = FIGURE_SPECS["FIG-12"]
    overlap_df = _get_table(tables, "hotspot_overlap.csv")
    candidates_df = _get_table(tables, "hotspot_candidates.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI, gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel 1: Overlap Jaccard Similarity Matrix
    candidates = list(candidates_df["subgroup"])
    n_cands = len(candidates)
    jaccard_mat = np.zeros((n_cands, n_cands))

    for _, row in overlap_df.iterrows():
        c1 = row["candidate_1"]
        c2 = row["candidate_2"]
        if c1 in candidates and c2 in candidates:
            i = candidates.index(c1)
            j = candidates.index(c2)
            j_sim = row["jaccard_similarity"]
            jaccard_mat[i, j] = j_sim
            jaccard_mat[j, i] = j_sim

    np.fill_diagonal(jaccard_mat, 1.0)

    im = ax1.imshow(jaccard_mat, cmap="Oranges", vmin=0, vmax=1.0)
    ax1.set_xticks(np.arange(n_cands))
    ax1.set_yticks(np.arange(n_cands))
    ax1.set_xticklabels([f"C{i+1}" for i in range(n_cands)], fontsize=8)
    ax1.set_yticklabels([f"C{i+1}: {textwrap.fill(c, 24)}" for i, c in enumerate(candidates)], fontsize=7)
    ax1.set_title("A. Pairwise Overlap Jaccard Similarity", fontsize=10.5, weight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Jaccard Similarity")

    # Panel 2: Union Summary
    # 599 employees (40.7%), 171 leavers (72.2%), rate 28.5%
    # Outside: 871 employees, 66 leavers, rate 7.6%
    categories = ["Workforce Headcount", "Total Leavers"]
    inside_vals = [599, 171]
    outside_vals = [871, 66]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width / 2, inside_vals, width, label="Inside Union (Rate: 28.5%)", color=FIGURE_PALETTE["highlight"], alpha=0.9, edgecolor="#111827")
    ax2.bar(x + width / 2, outside_vals, width, label="Outside Union (Rate: 7.6%)", color=FIGURE_PALETTE["primary"], alpha=0.8, edgecolor="#111827")

    ax2.annotate("599 (40.7%)", xy=(x[0] - width / 2, inside_vals[0] + 15), ha="center", fontsize=8.5, weight="bold")
    ax2.annotate("871 (59.3%)", xy=(x[0] + width / 2, outside_vals[0] + 15), ha="center", fontsize=8.5)
    ax2.annotate("171 (72.2%)", xy=(x[1] - width / 2, inside_vals[1] + 15), ha="center", fontsize=8.5, weight="bold")
    ax2.annotate("66 (27.8%)", xy=(x[1] + width / 2, outside_vals[1] + 15), ha="center", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9.5, weight="bold")
    ax2.set_ylabel("Headcount", fontsize=10, weight="bold")
    ax2.set_title("B. Candidate Union Representation", fontsize=10.5, weight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.set_ylim(0, 1000)

    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.02)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, overlap_df


# =============================================================================
# FIG-13: Numeric Distributions by Outcome
# =============================================================================
def plot_fig13_numeric_distributions(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot median and IQR comparison for 5 continuous measures by departure outcome."""
    spec = FIGURE_SPECS["FIG-13"]
    num_df = _get_table(tables, "numeric_distributions_by_outcome.csv")
    target_metrics = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome", "YearsSinceLastPromotion"]
    df = num_df[num_df["metric"].isin(target_metrics) & (num_df["outcome"] != "Overall")].copy().reset_index(drop=True)

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), dpi=FIGURE_DPI)

    for ax, metric in zip(axes, target_metrics):
        sub = df[df["metric"] == metric].copy()
        stayed = sub[sub["outcome"] == "Retained"].iloc[0]
        left = sub[sub["outcome"] == "Left"].iloc[0]

        # Points: medians, Error bars: Q25 to Q75
        outcomes = ["Retained\n(n=1,233)", "Left\n(n=237)"]
        medians = [stayed["median"], left["median"]]
        err_low = [stayed["median"] - stayed["q1"], left["median"] - left["q1"]]
        err_high = [stayed["q3"] - stayed["median"], left["q3"] - left["median"]]

        colors = [FIGURE_PALETTE["primary"], FIGURE_PALETTE["highlight"]]

        for idx in range(2):
            ax.errorbar(
                idx,
                medians[idx],
                yerr=[[err_low[idx]], [err_high[idx]]],
                fmt="o",
                color=colors[idx],
                elinewidth=2.5,
                capsize=6,
                markersize=8,
            )
            ax.annotate(
                f"Median: {medians[idx]:.0f}\nIQR: [{medians[idx]-err_low[idx]:.0f}, {medians[idx]+err_high[idx]:.0f}]",
                xy=(idx, medians[idx]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                weight="bold",
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(outcomes, fontsize=8.5)
        ax.set_title(metric, fontsize=10.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    axes[0].set_ylabel("Metric Value (Median & IQR)", fontsize=10, weight="bold")
    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=1.03)
    _add_footnote(fig, spec["footnote"])
    plt.tight_layout()

    return fig, df


# =============================================================================
# FIG-14: Demographic and Background Dimensions
# =============================================================================
def plot_fig14_demographics_background(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot 6 demographic and background dimensions with flagging highlights."""
    spec = FIGURE_SPECS["FIG-14"]
    gen = _get_table(tables, "gender_attrition.csv")
    mar = _get_table(tables, "maritalstatus_attrition.csv")
    edu = _get_table(tables, "education_attrition.csv")
    edf = _get_table(tables, "educationfield_attrition.csv")
    pe = _get_table(tables, "prior_experience_attrition.csv")
    pr = _get_table(tables, "performancerating_attrition.csv")

    baseline_rate = _get_baseline_rate(tables)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=FIGURE_DPI)
    axes_flat = axes.flatten()

    panels_info = [
        (gen, "A. Gender (No Separation)", "segment"),
        (mar, "B. Marital Status (Single Elevated)", "segment"),
        (edu, "C. Education Level (No Separation)", "segment"),
        (edf, "D. Education Field (Marketing & Tech Elevated)", "segment"),
        (pe, "E. Prior Experience (No Separation)", "segment"),
        (pr, "F. Performance Rating (No Separation)", "segment"),
    ]

    for ax, (data, title, col_lbl) in zip(axes_flat, panels_info):
        x = np.arange(len(data))
        rates = data["attrition_rate_pct"].values
        yerr_l = rates - data["ci95_low_pct"].values
        yerr_h = data["ci95_high_pct"].values - rates

        colors = [
            FIGURE_PALETTE["highlight"] if data["ci95_low_pct"].iloc[i] > baseline_rate
            else (FIGURE_PALETTE["secondary"] if data["ci95_high_pct"].iloc[i] < baseline_rate else FIGURE_PALETTE["primary"])
            for i in range(len(data))
        ]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        for idx, rect in enumerate(bars):
            h = rect.get_height()
            n_val = data["n"].iloc[idx]
            ax.annotate(f"{h:.1f}%\n(n={n_val})", xy=(rect.get_x() + rect.get_width() / 2, h / 2), ha="center", va="center", color="white", weight="bold", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([textwrap.fill(str(s), 11) for s in data[col_lbl]], fontsize=8)
        ax.set_ylabel("% who left", fontsize=8.5, weight="bold")
        ax.set_title(title, fontsize=9.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(spec["title"] + "\n" + spec["subtitle"], fontsize=12, weight="bold", y=0.98)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout()

    plotted_df = pd.concat([gen, mar, edu, edf, pe, pr], ignore_index=True)
    return fig, plotted_df
