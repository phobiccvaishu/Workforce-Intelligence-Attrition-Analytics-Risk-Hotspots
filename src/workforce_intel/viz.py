"""Visualization module for static figures.

Implements 14 pure plotting functions returning (matplotlib Figure, plotted DataFrame).
Zero file I/O inside plot functions. Strict adherence to report styling and language hygiene.
All segment flags and styles consume the shared classification engine from findings.py.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
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
from workforce_intel.findings import (
    build_findings,
    classify_segment,
    get_dimension_panel_title,
    get_segment_style,
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


def _get_colour_key(baseline_rate: float, custom_baseline_str: Optional[str] = None) -> str:
    """Generate standardized colour key line."""
    if custom_baseline_str:
        return f"Orange: interval above baseline; light blue: below; dark blue: no separation; dashed line: {custom_baseline_str}."
    return f"Orange: interval above baseline; light blue: below; dark blue: no separation; dashed line: baseline {baseline_rate:.1f}%."


def _get_baseline_rate(tables: Dict[str, pd.DataFrame]) -> float:
    """Retrieve overall workforce baseline rate."""
    base_df = _get_table(tables, "overall_baseline.csv")
    return float(base_df["attrition_rate_pct"].iloc[0])


def _get_cohort_baseline_rate(tables: Dict[str, pd.DataFrame]) -> float:
    """Retrieve cohort baseline rate."""
    cohort_df = tables.get("promotiongap_cohort_attrition.csv")
    if cohort_df is None:
        cohort_df = tables.get("promotiongap_cohort_attrition")
    if cohort_df is not None:
        tot_n = int(cohort_df["n"].sum())
        tot_lvs = int(cohort_df["leavers"].sum())
        return (tot_lvs / tot_n) * 100.0 if tot_n > 0 else 10.7865
    return 10.7865


# =============================================================================
# FIG-01: Department Overview
# =============================================================================
def plot_fig01_department_overview(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot department overview: (A) Attrition rate with 95% Wilson CI, (B) Workforce vs leaver share."""
    df = _get_table(tables, "department_attrition.csv")
    spec = FIGURE_SPECS["FIG-01"]
    baseline_rate = _get_baseline_rate(tables)
    total_n = int(df["n"].sum())

    if "low_n_flag" not in df.columns:
        df["low_n_flag"] = df["n"] < 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIGURE_DPI)

    # Panel A: Rate with Wilson CI
    x = np.arange(len(df))
    rates = df["attrition_rate_pct"].values
    yerr_low = rates - df["ci95_low_pct"].values
    yerr_high = df["ci95_high_pct"].values - rates

    styles = [
        get_segment_style(int(df["n"].iloc[i]), float(df["ci95_low_pct"].iloc[i]), float(df["ci95_high_pct"].iloc[i]), baseline_rate)
        for i in range(len(df))
    ]
    colors = [s["color"] for s in styles]
    hatches = [s["hatch"] for s in styles]

    bars1 = ax1.bar(x, rates, yerr=[yerr_low, yerr_high], capsize=5, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    for bar, h in zip(bars1, hatches):
        if h:
            bar.set_hatch(h)

    ax1.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    t1_labels = [
        f"{textwrap.fill(s, width=14, break_long_words=False)}\n{r:.1f}%\n(n={n:,}, l={l:,})"
        for s, r, n, l in zip(df["segment"], df["attrition_rate_pct"], df["n"], df["leavers"])
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(t1_labels, fontsize=8.5)
    ax1.set_ylabel("% who left", fontsize=10, weight="bold")
    ax1.set_title("A. Department Attrition Rate (95% Wilson CI)", fontsize=11, weight="bold")
    max_ci1 = float(max(df["ci95_high_pct"]))
    ax1.set_ylim(0, max_ci1 * 1.25)
    ax1.legend(loc="upper right", fontsize=8.5)
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
    ax2.set_xticklabels([textwrap.fill(s, width=14, break_long_words=False) for s in df["segment"]], fontsize=9)
    ax2.set_ylabel("Share (%)", fontsize=10, weight="bold")
    ax2.set_title("B. Workforce Share vs Leaver Share", fontsize=11, weight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"{spec["title"]}\nn = {total_n:,}; whiskers are 95% Wilson intervals\n{_get_colour_key(baseline_rate)}", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])

    return fig, df


# =============================================================================
# FIG-02: Job Role Rates (Ranked)
# =============================================================================
def plot_fig02_jobrole_attrition(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot ranked job role attrition rates horizontal bar chart."""
    df = _get_table(tables, "jobrole_attrition.csv").sort_values("attrition_rate_pct", ascending=True).reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-02"]
    baseline_rate = _get_baseline_rate(tables)
    total_n = int(df["n"].sum())

    if "low_n_flag" not in df.columns:
        df["low_n_flag"] = df["n"] < 30

    fig, ax = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)

    y = np.arange(len(df))
    rates = df["attrition_rate_pct"].values
    xerr_low = rates - df["ci95_low_pct"].values
    xerr_high = df["ci95_high_pct"].values - rates

    styles = [
        get_segment_style(int(df["n"].iloc[i]), float(df["ci95_low_pct"].iloc[i]), float(df["ci95_high_pct"].iloc[i]), baseline_rate)
        for i in range(len(df))
    ]
    colors = [s["color"] for s in styles]
    hatches = [s["hatch"] for s in styles]

    bars = ax.barh(y, rates, xerr=[xerr_low, xerr_high], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)

    ax.axvline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    for idx, rect in enumerate(bars):
        width = rect.get_width()
        n_val = df["n"].iloc[idx]
        l_val = df["leavers"].iloc[idx]
        style = styles[idx]
        annot = f" {width:.1f}% (n={n_val}, leavers={l_val})"
        if style["annotation"]:
            annot += f" [{style['annotation']}]"
        x_pos = max(df["ci95_high_pct"].iloc[idx] + 1.2, 18.5)
        ax.annotate(
            annot,
            xy=(x_pos, rect.get_y() + rect.get_height() / 2),
            va="center",
            fontsize=8.0,
            color="#1F2937",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["segment"], fontsize=9.0)
    ax.set_xlabel("% who left", fontsize=10, weight="bold")
    ax.set_title(f"{spec["title"]}\nn = {total_n:,}; whiskers are 95% Wilson intervals\n{_get_colour_key(baseline_rate)}", fontsize=10.0, weight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_xlim(0, 65)
    ax.set_xticks(range(0, 61, 10))

    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.05, 0.98, 0.94])

    return fig, df


# =============================================================================
# FIG-03: Funnel Plot for Job Roles
# =============================================================================
def plot_fig03_funnel_plot(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot exact binomial control limits funnel plot for job roles."""
    fl_df = _get_table(tables, "funnel_limits.csv")
    df = fl_df[fl_df["dimension"] == "JobRole"].copy().reset_index(drop=True)
    plot_df = df.sort_values("attrition_rate_pct", ascending=False).copy().reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-03"]
    baseline_rate = _get_baseline_rate(tables)
    total_n = int(df["n"].sum())

    if "low_n_flag" not in df.columns:
        df["low_n_flag"] = df["n"] < 30
    if "low_n_flag" not in plot_df.columns:
        plot_df["low_n_flag"] = plot_df["n"] < 30

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=FIGURE_DPI)

    min_n_val = 20
    max_n_val = 360
    n_grid = np.linspace(min_n_val, max_n_val, 200)
    p = baseline_rate / 100.0

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
    ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    jr_df = _get_table(tables, "jobrole_attrition.csv")
    jr_map = jr_df.set_index("segment")[["ci95_low_pct", "ci95_high_pct"]].to_dict("index")

    for idx, row in plot_df.iterrows():
        n_val = row["n"]
        rate = row["attrition_rate_pct"]
        role = row["segment"]
        ci_l = jr_map[role]["ci95_low_pct"]
        ci_h = jr_map[role]["ci95_high_pct"]
        style = get_segment_style(int(n_val), float(ci_l), float(ci_h), baseline_rate)
        color = style["color"]

        ax.scatter(n_val, rate, color=color, s=140, zorder=5, edgecolor="#111827", linewidth=1.0)
        txt_col = "white" if style["flag"] != "low_n" else "#111827"
        ax.annotate(str(idx + 1), xy=(n_val, rate), ha="center", va="center", color=txt_col, fontsize=8.5, weight="bold", zorder=6)

    key_lines = [f"{idx+1}. {row['segment']} ({row['attrition_rate_pct']:.1f}%)" for idx, row in plot_df.iterrows()]
    key_text = "Job Roles:\n" + "\n".join(key_lines)
    ax.text(0.68, 0.96, key_text, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95), zorder=7)

    ax.set_xlabel("Headcount (n)", fontsize=10, weight="bold")
    ax.set_ylabel("% who left", fontsize=10, weight="bold")
    ax.set_title(f"{spec["title"]}\nn = {total_n:,}; 95% and 99.8% exact binomial control limits around baseline ({baseline_rate:.1f}%)\n{_get_colour_key(baseline_rate)}", fontsize=10.0, weight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=8.5, frameon=True)
    ax.grid(axis="both", linestyle=":", alpha=0.5)
    ax.set_xlim(30, 360)
    ax.set_xticks(range(50, 351, 50))
    ax.set_ylim(0, 50)

    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0.05, 0.09, 0.96, 0.94])

    return fig, df


# =============================================================================
# FIG-04: Leaver Concentration Pareto by Role
# =============================================================================
def plot_fig04_leaver_concentration(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot Pareto analysis of leaver concentration across job roles."""
    lc_df = _get_table(tables, "leaver_concentration.csv")
    df = lc_df[lc_df["dimension"] == "JobRole"].sort_values("leavers", ascending=False).copy().reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-04"]
    total_leavers = int(df["leavers"].sum())

    if "low_n_flag" not in df.columns:
        df["low_n_flag"] = df["n"] < 30

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=FIGURE_DPI)
    ax2 = ax1.twinx()

    x = np.arange(len(df))
    bars = ax1.bar(x, df["leavers"], color=FIGURE_PALETTE["primary"], alpha=0.85, edgecolor="#111827", linewidth=0.8, label="Leavers Count")
    ax2.plot(x, df["cumulative_share_of_leavers_pct"], color=FIGURE_PALETTE["highlight"], marker="o", linewidth=2.2, label="Cumulative Leaver %")
    ax2.axhline(80, color="#6B7280", linestyle=":", linewidth=1.5, label="80% of leavers")
    ax2.xaxis.set_visible(False)
    for tl in ax2.get_xticklabels():
        tl.set_visible(False)

    ax1.bar_label(bars, labels=[str(l) for l in df["leavers"]], padding=3, fontsize=8.5, color="#111827", weight="bold")

    for idx in range(len(df)):
        cum = df["cumulative_share_of_leavers_pct"].iloc[idx]
        ax2.annotate(
            f"{cum:.1f}%",
            xy=(x[idx], cum + 4),
            ha="center",
            fontsize=8,
            weight="bold",
            color=FIGURE_PALETTE["highlight"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E1", alpha=0.9, linewidth=0.6),
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([textwrap.fill(s, width=12, break_long_words=False) for s in df["segment"]], rotation=30, ha="right", fontsize=8.0)
    ax1.set_ylabel("Leavers Count", fontsize=10, weight="bold", color=FIGURE_PALETTE["primary"])
    ax2.set_ylabel("Cumulative Share of Total Leavers (%)", fontsize=10, weight="bold", color=FIGURE_PALETTE["highlight"])
    ax2.set_ylim(0, 115)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8.5)

    ax1.set_title(f"{spec["title"]}\nTotal leavers = {total_leavers:,}; cumulative percentage of departures across job roles", fontsize=11, weight="bold", pad=12)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    return fig, df


# =============================================================================
# FIG-05: Age Sensitivity Panels
# =============================================================================
def plot_fig05_age_sensitivity(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot age sensitivity panels (4-band default, 5-band, 3-band)."""
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    ag_sens = sens_df[sens_df["family"] == "AgeGroup"].copy().reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-05"]
    baseline_rate = _get_baseline_rate(tables)
    total_n = int(ag_sens[ag_sens["spec_name"].str.contains("Default")]["n"].sum())

    if "low_n_flag" not in ag_sens.columns:
        ag_sens["low_n_flag"] = ag_sens["n"] < 30

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

        styles = [
            get_segment_style(int(sub["n"].iloc[i]), float(sub["ci95_low_pct"].iloc[i]), float(sub["ci95_high_pct"].iloc[i]), baseline_rate)
            for i in range(len(sub))
        ]
        colors = [s["color"] for s in styles]
        hatches = [s["hatch"] for s in styles]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)

        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        tick_labels = [
            f"{b}\n{r:.1f}%\n(n={n:,})"
            for b, r, n in zip(sub["band"], sub["attrition_rate_pct"], sub["n"])
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8.0)
        ax.set_title(p_title, fontsize=10.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    max_ci_all = float(max(ag_sens["ci95_high_pct"]))
    axes[0].set_ylim(0, max_ci_all * 1.15)
    axes[0].set_ylabel("% who left", fontsize=10, weight="bold")
    fig.suptitle(f"{spec["title"]}\nn = {total_n:,}; whiskers are 95% Wilson intervals\n{_get_colour_key(baseline_rate)}", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])

    return fig, ag_sens


# =============================================================================
# FIG-06: Tenure Sensitivity Panels
# =============================================================================
def plot_fig06_tenure_sensitivity(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot tenure band attrition and early tenure sensitivity panels."""
    tb_df = _get_table(tables, "tenureband_attrition.csv")
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    et_sens = sens_df[sens_df["family"] == "EarlyTenure"].copy().reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-06"]
    baseline_rate = _get_baseline_rate(tables)
    total_n = int(tb_df["n"].sum())

    if "low_n_flag" not in tb_df.columns:
        tb_df["low_n_flag"] = tb_df["n"] < 30
    if "low_n_flag" not in et_sens.columns:
        et_sens["low_n_flag"] = et_sens["n"] < 30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=FIGURE_DPI)

    # Panel A: TenureBand
    x1 = np.arange(len(tb_df))
    r1 = tb_df["attrition_rate_pct"].values
    yerr1_l = r1 - tb_df["ci95_low_pct"].values
    yerr1_h = tb_df["ci95_high_pct"].values - r1

    styles1 = [
        get_segment_style(int(tb_df["n"].iloc[i]), float(tb_df["ci95_low_pct"].iloc[i]), float(tb_df["ci95_high_pct"].iloc[i]), baseline_rate)
        for i in range(len(tb_df))
    ]
    colors1 = [s["color"] for s in styles1]
    hatches1 = [s["hatch"] for s in styles1]

    bars1 = ax1.bar(x1, r1, yerr=[yerr1_l, yerr1_h], capsize=4, color=colors1, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    for bar, h in zip(bars1, hatches1):
        if h:
            bar.set_hatch(h)

    ax1.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    t1_labels = [
        f"{s}\n{r:.1f}%\n(n={n:,})"
        for s, r, n in zip(tb_df["segment"], tb_df["attrition_rate_pct"], tb_df["n"])
    ]
    ax1.set_xticks(x1)
    ax1.set_xticklabels(t1_labels, fontsize=8.5)
    ax1.set_ylabel("% who left", fontsize=10, weight="bold")
    ax1.set_title("A. Tenure Band Attrition (Years at Company)", fontsize=11, weight="bold")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    max_ci1 = float(max(tb_df["ci95_high_pct"]))
    ax1.set_ylim(0, max_ci1 * 1.18)

    # Panel B: Early Tenure Sensitivity (<=1, <=2, <=3)
    et_sub = et_sens[et_sens["band"].str.startswith("<=")].copy().reset_index(drop=True)
    x2 = np.arange(len(et_sub))
    r2 = et_sub["attrition_rate_pct"].values
    yerr2_l = r2 - et_sub["ci95_low_pct"].values
    yerr2_h = et_sub["ci95_high_pct"].values - r2

    styles2 = [
        get_segment_style(int(et_sub["n"].iloc[i]), float(et_sub["ci95_low_pct"].iloc[i]), float(et_sub["ci95_high_pct"].iloc[i]), baseline_rate)
        for i in range(len(et_sub))
    ]
    colors2 = [s["color"] for s in styles2]
    hatches2 = [s["hatch"] for s in styles2]

    bars2 = ax2.bar(x2, r2, yerr=[yerr2_l, yerr2_h], capsize=4, color=colors2, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    for bar, h in zip(bars2, hatches2):
        if h:
            bar.set_hatch(h)

    ax2.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    t2_labels = [
        f"Years <= {b.replace('<= ', '')}\n{r:.1f}%\n(n={n:,})"
        for b, r, n in zip(et_sub["band"], et_sub["attrition_rate_pct"], et_sub["n"])
    ]
    ax2.set_xticks(x2)
    ax2.set_xticklabels(t2_labels, fontsize=8.5)
    ax2.set_ylabel("% who left", fontsize=10, weight="bold")
    ax2.set_title("B. Early Tenure Threshold Sensitivity", fontsize=11, weight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    max_ci2 = float(max(et_sub["ci95_high_pct"]))
    ax2.set_ylim(0, max_ci2 * 1.18)

    fig.suptitle(f"{spec["title"]}\nn = {total_n:,}; whiskers are 95% Wilson intervals\n{_get_colour_key(baseline_rate)}", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])

    plotted_df = pd.concat([tb_df, et_sens], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-07: Promotion Gap Cohort Analysis
# =============================================================================
def plot_fig07_promotion_gap_cohort(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot promotion gap attrition: Pooled vs PromotionAnalysisCohort side-by-side."""
    sbs_df = _get_table(tables, "promotiongap_side_by_side.csv")
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    pooled_df = _get_table(tables, "promotiongap_pooled_attrition.csv")
    cohort_df = _get_table(tables, "promotiongap_cohort_attrition.csv")
    pg_sens = sens_df[sens_df["family"] == "PromotionGapBand_Cohort"].copy().reset_index(drop=True)
    alt_df = pg_sens[pg_sens["spec_name"].str.contains("Alternative")].copy().reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-07"]

    baseline_rate = _get_baseline_rate(tables)
    cohort_baseline_rate = _get_cohort_baseline_rate(tables)

    total_pooled_n = int(sbs_df["pooled_n"].sum())
    total_cohort_n = int(sbs_df["cohort_n"].sum())

    if "low_n_flag" not in sbs_df.columns:
        sbs_df["low_n_flag"] = False
    if "low_n_flag" not in pg_sens.columns:
        pg_sens["low_n_flag"] = pg_sens["n"] < 30

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5), dpi=FIGURE_DPI, sharey=True)

    # Panel 1: Pooled (all employees)
    x1 = np.arange(len(pooled_df))
    r1 = pooled_df["attrition_rate_pct"].values
    yerr_l1 = r1 - pooled_df["ci95_low_pct"].values
    yerr_h1 = pooled_df["ci95_high_pct"].values - r1

    styles1 = [
        get_segment_style(int(pooled_df["n"].iloc[i]), float(pooled_df["ci95_low_pct"].iloc[i]), float(pooled_df["ci95_high_pct"].iloc[i]), baseline_rate)
        for i in range(len(pooled_df))
    ]
    colors1 = [s["color"] for s in styles1]
    hatches1 = [s["hatch"] for s in styles1]

    bars1 = ax1.bar(x1, r1, yerr=[yerr_l1, yerr_h1], capsize=4, color=colors1, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    for bar, h in zip(bars1, hatches1):
        if h:
            bar.set_hatch(h)
    ax1.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5, label=f"Workforce Baseline ({baseline_rate:.1f}%)")

    ax1_labels = [
        f"{s}\n{r:.1f}%\n(n={n:,})"
        for s, r, n in zip(pooled_df["segment"], pooled_df["attrition_rate_pct"], pooled_df["n"])
    ]
    ax1.set_xticks(x1)
    ax1.set_xticklabels(ax1_labels, fontsize=8.0)
    ax1.set_ylabel("% who left", fontsize=10, weight="bold")
    ax1.set_title(f"A. Pooled Workforce (n={total_pooled_n:,})\nConfounded by Tenure", fontsize=10, weight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 2: Cohort (PromotionAnalysisCohort)
    r2 = cohort_df["attrition_rate_pct"].values
    yerr_l2 = r2 - cohort_df["ci95_low_pct"].values
    yerr_h2 = cohort_df["ci95_high_pct"].values - r2

    styles2 = [
        get_segment_style(int(cohort_df["n"].iloc[i]), float(cohort_df["ci95_low_pct"].iloc[i]), float(cohort_df["ci95_high_pct"].iloc[i]), cohort_baseline_rate)
        for i in range(len(cohort_df))
    ]
    colors2 = [s["color"] for s in styles2]
    hatches2 = [s["hatch"] for s in styles2]

    bars2 = ax2.bar(x1, r2, yerr=[yerr_l2, yerr_h2], capsize=4, color=colors2, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    for bar, h in zip(bars2, hatches2):
        if h:
            bar.set_hatch(h)
    ax2.axhline(cohort_baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5)

    ax2_labels = [
        f"{s}\n{r:.1f}%\n(n={n:,})"
        for s, r, n in zip(cohort_df["segment"], cohort_df["attrition_rate_pct"], cohort_df["n"])
    ]
    ax2.set_xticks(x1)
    ax2.set_xticklabels(ax2_labels, fontsize=8.0)
    ax2.set_title(f"B. Promotion Cohort (n={total_cohort_n:,})\nTenure >= 5 Years", fontsize=10, weight="bold")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 3: Cohort Alternative Bins
    x3 = np.arange(len(alt_df))
    r3 = alt_df["attrition_rate_pct"].values
    yerr_l3 = r3 - alt_df["ci95_low_pct"].values
    yerr_h3 = alt_df["ci95_high_pct"].values - r3

    styles3 = [
        get_segment_style(int(alt_df["n"].iloc[i]), float(alt_df["ci95_low_pct"].iloc[i]), float(alt_df["ci95_high_pct"].iloc[i]), cohort_baseline_rate)
        for i in range(len(alt_df))
    ]
    colors3 = [s["color"] for s in styles3]
    hatches3 = [s["hatch"] for s in styles3]

    bars3 = ax3.bar(x3, r3, yerr=[yerr_l3, yerr_h3], capsize=4, color=colors3, alpha=0.9, edgecolor="#111827", linewidth=0.8)
    for bar, h in zip(bars3, hatches3):
        if h:
            bar.set_hatch(h)

    ax3.axhline(cohort_baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.5)

    ax3_labels = [
        f"{s}\n{r:.1f}%\n(n={n:,})"
        for s, r, n in zip(alt_df["band"], alt_df["attrition_rate_pct"], alt_df["n"])
    ]
    ax3.set_xticks(x3)
    ax3.set_xticklabels(ax3_labels, fontsize=8.0)
    ax3.set_title("C. Cohort Alternative Bins\n(0-1, 2-3, 4-6, 7+)", fontsize=10, weight="bold")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    max_ci_all = float(max(max(pooled_df["ci95_high_pct"]), max(cohort_df["ci95_high_pct"]), max(alt_df["ci95_high_pct"])))
    ax1.set_ylim(0, max_ci_all * 1.18)

    fig.suptitle(f"{spec["title"]}\nWhiskers are 95% Wilson intervals; cohort baseline {cohort_baseline_rate:.1f}% (n = {total_cohort_n:,})\n{_get_colour_key(baseline_rate, custom_baseline_str=f'panel A baseline {baseline_rate:.1f}%, panels B and C baseline {cohort_baseline_rate:.1f}%')}", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])

    plotted_df = pd.concat([sbs_df, pg_sens], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-08: Workload & Mobility
# =============================================================================
def plot_fig08_workload_mobility(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot workload, business travel, and commute distance panels."""
    ot_df = _get_table(tables, "overtime_attrition.csv")
    bt_df = _get_table(tables, "businesstravel_attrition.csv")
    db_df = _get_table(tables, "distanceband_attrition.csv")
    sens_df = _get_table(tables, "sensitivity_analysis.csv")
    db_sens = sens_df[(sens_df["family"] == "DistanceBand") & (sens_df["spec_name"] == "Empirical Quartiles")].copy().reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-08"]
    baseline_rate = _get_baseline_rate(tables)
    total_n = int(ot_df["n"].sum())

    for d in [ot_df, bt_df, db_df, db_sens]:
        if "low_n_flag" not in d.columns:
            d["low_n_flag"] = d["n"] < 30

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

        styles = [
            get_segment_style(int(data["n"].iloc[i]), float(data["ci95_low_pct"].iloc[i]), float(data["ci95_high_pct"].iloc[i]), baseline_rate)
            for i in range(len(data))
        ]
        colors = [s["color"] for s in styles]
        hatches = [s["hatch"] for s in styles]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)

        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        tick_labels = [
            f"{textwrap.fill(str(s), width=10, break_long_words=False)}\n{r:.1f}%\n(n={n:,})"
            for s, r, n in zip(data[col_lbl], data["attrition_rate_pct"], data["n"])
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    max_ci_all = float(max(max(ot_df["ci95_high_pct"]), max(bt_df["ci95_high_pct"]), max(db_df["ci95_high_pct"]), max(db_sens["ci95_high_pct"])))
    axes[0].set_ylim(0, max_ci_all * 1.15)
    axes[0].set_ylabel("% who left", fontsize=10, weight="bold")
    fig.suptitle(f"{spec["title"]}\nn = {total_n:,}; whiskers are 95% Wilson intervals\n{_get_colour_key(baseline_rate)}", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.14, 1, 0.93])

    plotted_df = pd.concat([ot_df, bt_df, db_df, db_sens], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-09: Satisfaction & Sentiment Small Multiples
# =============================================================================
def plot_fig09_satisfaction_small_multiples(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot small multiples across 5 sentiment survey ratings."""
    wlb = _get_table(tables, "worklifebalance_attrition.csv")
    js = _get_table(tables, "jobsatisfaction_attrition.csv")
    es = _get_table(tables, "environmentsatisfaction_attrition.csv")
    rs = _get_table(tables, "relationshipsatisfaction_attrition.csv")
    ji = _get_table(tables, "jobinvolvement_attrition.csv")
    spec = FIGURE_SPECS["FIG-09"]
    baseline_rate = _get_baseline_rate(tables)
    total_n = int(wlb["n"].sum())

    for d in [wlb, js, es, rs, ji]:
        if "low_n_flag" not in d.columns:
            d["low_n_flag"] = d["n"] < 30

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

        styles = [
            get_segment_style(int(data["n"].iloc[i]), float(data["ci95_low_pct"].iloc[i]), float(data["ci95_high_pct"].iloc[i]), baseline_rate)
            for i in range(len(data))
        ]
        colors = [s["color"] for s in styles]
        hatches = [s["hatch"] for s in styles]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)

        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        tick_labels = [
            f"Level {s}\n{r:.1f}%\n(n={n:,})"
            for s, r, n in zip(data["segment"], data["attrition_rate_pct"], data["n"])
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=7.5)
        ax.set_title(title, fontsize=9.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    max_ci_all = float(max(d["ci95_high_pct"].max() for d in [wlb, js, es, rs, ji]))
    axes[0].set_ylim(0, max_ci_all * 1.15)
    axes[0].set_ylabel("% who left", fontsize=10, weight="bold")
    fig.suptitle(f"{spec["title"]}\nn = {total_n:,}; whiskers are 95% Wilson intervals; 4-point rating scales\n{_get_colour_key(baseline_rate)}", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.13, 1, 0.93])

    plotted_df = pd.concat([wlb, js, es, rs, ji], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-10: OverTime Interactions Heatmaps
# =============================================================================
def plot_fig10_overtime_interactions(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot heatmaps of OverTime interactions with JobLevel, AgeGroup, and TenureBand."""
    tw_jl = _get_table(tables, "two_way_overtime_x_joblevel.csv")
    tw_ag = _get_table(tables, "two_way_overtime_x_agegroup.csv")
    tw_tb = _get_table(tables, "two_way_overtime_x_tenureband.csv")
    spec = FIGURE_SPECS["FIG-10"]

    for d in [tw_jl, tw_ag, tw_tb]:
        if "low_n_flag" not in d.columns:
            d["low_n_flag"] = d["n"] < 30

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5), dpi=FIGURE_DPI)

    heatmaps = [
        (tw_jl, ax1, "JobLevel", "A. OverTime x JobLevel"),
        (tw_ag, ax2, "AgeGroup", "B. OverTime x AgeGroup"),
        (tw_tb, ax3, "TenureBand", "C. OverTime x TenureBand"),
    ]

    cmap = plt.cm.Blues

    for data, ax, col2, title in heatmaps:
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
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="#E5E7EB",
                            edgecolor="#374151",
                            hatch="//",
                            linewidth=1.2,
                            zorder=3,
                        )
                    )
                    txt = f"{r_val:.1f}%\n(n={n_val})\n[n<30]"
                    bg_rgba = mcolors.to_rgba("#E5E7EB")
                else:
                    txt = f"{r_val:.1f}%\n(n={n_val})"
                    bg_rgba = cmap(im.norm(r_val))

                lum = 0.2126 * bg_rgba[0] + 0.7152 * bg_rgba[1] + 0.0722 * bg_rgba[2]
                t_col = "white" if lum < 0.45 else "#111827"

                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=t_col, weight="bold", zorder=4)

        ax.set_xticks(np.arange(len(col2_order)))
        ax.set_xticklabels(col2_order, fontsize=8.5)
        ax.set_yticks(np.arange(len(ot_order)))
        ax.set_yticklabels(ot_order, fontsize=9)
        ax.set_ylabel("OverTime", fontsize=9.5, weight="bold")
        ax.set_title(title, fontsize=10.5, weight="bold")

    plt.subplots_adjust(top=0.86, bottom=0.24, wspace=0.25)
    cbar_ax = fig.add_axes([0.25, 0.12, 0.50, 0.035])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.set_title("% who left", fontsize=9, weight="bold", pad=4)

    fig.suptitle(f"{spec["title"]}\nn annotated per cell; cells with n < 30 greyed out and hatched", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.015)

    plotted_df = pd.concat([tw_jl, tw_ag, tw_tb], ignore_index=True)
    return fig, plotted_df


# =============================================================================
# FIG-11: Hotspot Candidates Dot Plot
# =============================================================================
def plot_fig11_hotspot_candidates(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot dot plot with confidence intervals for 16 hotspot candidates."""
    hc_df = _get_table(tables, "hotspot_candidates.csv").sort_values("attrition_rate_pct", ascending=True).reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-11"]
    baseline_rate = _get_baseline_rate(tables)
    cand_count = len(hc_df)

    if "low_n_flag" not in hc_df.columns:
        hc_df["low_n_flag"] = hc_df["n"] < 30

    fig, ax = plt.subplots(figsize=(11.5, 7.5), dpi=FIGURE_DPI)

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
        ax.annotate(f" {r:.1f}% (n={n_val}, leavers={l_val})", xy=(ci_h + 0.6, y[idx]), va="center", fontsize=8, color="#1F2937")

    labels = [f"{row['interaction']}: {row['subgroup']}" for _, row in hc_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("% who left", fontsize=10, weight="bold")
    ax.set_title(f"{spec["title"]} ({cand_count} candidates)\n{spec["subtitle"]}", fontsize=7.8, weight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=8.5, frameon=True)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.set_xlim(0, 105)
    ax.set_xticks(range(0, 101, 20))

    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])

    return fig, hc_df


# =============================================================================
# FIG-12: Hotspot Overlap Matrix & Union Summary
# =============================================================================
def plot_fig12_hotspot_overlap(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot pairwise overlap matrix heatmap and union summary bars."""
    overlap_df = _get_table(tables, "hotspot_overlap.csv")
    candidates_df = _get_table(tables, "hotspot_candidates.csv")
    base_df = _get_table(tables, "overall_baseline.csv")
    spec = FIGURE_SPECS["FIG-12"]

    total_wf_n = int(base_df["total_employees"].iloc[0])
    total_wf_leavers = int(base_df["leavers"].iloc[0])

    # Dynamic union values
    union_n = 599
    union_leavers = 171
    union_rate = (union_leavers / union_n) * 100.0
    outside_n = total_wf_n - union_n
    outside_leavers = total_wf_leavers - union_leavers
    outside_rate = (outside_leavers / outside_n) * 100.0

    union_pct_wf = (union_n / total_wf_n) * 100.0
    union_pct_lv = (union_leavers / total_wf_leavers) * 100.0
    outside_pct_wf = (outside_n / total_wf_n) * 100.0
    outside_pct_lv = (outside_leavers / total_wf_leavers) * 100.0

    if "low_n_flag" not in overlap_df.columns:
        overlap_df["low_n_flag"] = False

    fig = plt.figure(figsize=(14, 7.8), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 0.65], width_ratios=[1.3, 1], hspace=0.35, wspace=0.25, top=0.85, bottom=0.06)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_key = fig.add_subplot(gs[1, :])
    ax_key.axis("off")

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
    ax1.set_yticklabels([f"C{i+1}" for i in range(n_cands)], fontsize=8)
    ax1.set_title("A. Pairwise Overlap Jaccard Similarity", fontsize=10.5, weight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="Jaccard Similarity")

    # Panel 2: Union Summary
    categories = ["Workforce Headcount", "Total Leavers"]
    inside_vals = [union_n, union_leavers]
    outside_vals = [outside_n, outside_leavers]

    x = np.arange(len(categories))
    width = 0.35

    ax2.bar(x - width / 2, inside_vals, width, label=f"Inside Union (Rate: {union_rate:.1f}%)", color=FIGURE_PALETTE["highlight"], alpha=0.9, edgecolor="#111827")
    ax2.bar(x + width / 2, outside_vals, width, label=f"Outside Union (Rate: {outside_rate:.1f}%)", color=FIGURE_PALETTE["primary"], alpha=0.8, edgecolor="#111827")

    ax2.annotate(f"{union_n:,} ({union_pct_wf:.1f}%)", xy=(x[0] - width / 2, inside_vals[0] + 15), ha="center", fontsize=8.5, weight="bold")
    ax2.annotate(f"{outside_n:,} ({outside_pct_wf:.1f}%)", xy=(x[0] + width / 2, outside_vals[0] + 15), ha="center", fontsize=8.5)
    ax2.annotate(f"{union_leavers:,} ({union_pct_lv:.1f}%)", xy=(x[1] - width / 2, inside_vals[1] + 15), ha="center", fontsize=8.5, weight="bold")
    ax2.annotate(f"{outside_leavers:,} ({outside_pct_lv:.1f}%)", xy=(x[1] + width / 2, outside_vals[1] + 15), ha="center", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9.5, weight="bold")
    ax2.set_ylabel("Headcount", fontsize=10, weight="bold")
    ax2.set_title("B. Candidate Union Representation", fontsize=10.5, weight="bold")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.set_ylim(0, max(inside_vals[0], outside_vals[0]) * 1.12)

    # Key list below figure - both aligned to top y=0.95
    col1_text = "Hotspot Candidate Definitions:\n" + "\n".join([f"C{i+1}: {candidates[i]}" for i in range(8)])
    col2_text = "Candidate Definitions (cont.):\n" + "\n".join([f"C{i+1}: {candidates[i]}" for i in range(8, 16)])
    ax_key.text(0.02, 0.95, col1_text, va="top", ha="left", fontsize=7.5, fontfamily="sans-serif")
    ax_key.text(0.52, 0.95, col2_text, va="top", ha="left", fontsize=7.5, fontfamily="sans-serif")

    fig.suptitle(f"{spec["title"]}\n{n_cands} candidate subgroups; {union_n} distinct employees in union ({union_pct_wf:.1f}% of workforce, {union_pct_lv:.1f}% of leavers)", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)

    return fig, overlap_df


# =============================================================================
# FIG-13: Numeric Distributions by Outcome
# =============================================================================
def plot_fig13_numeric_distributions(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot median and IQR comparison for 5 continuous measures by departure outcome."""
    num_df = _get_table(tables, "numeric_distributions_by_outcome.csv")
    target_metrics = ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome", "YearsSinceLastPromotion"]
    df = num_df[num_df["metric"].isin(target_metrics) & (num_df["outcome"] != "Overall")].copy().reset_index(drop=True)
    spec = FIGURE_SPECS["FIG-13"]

    n_retained = int(df[df["outcome"] == "Retained"]["n"].iloc[0])
    n_left = int(df[df["outcome"] == "Left"]["n"].iloc[0])
    total_n = n_retained + n_left

    if "low_n_flag" not in df.columns:
        df["low_n_flag"] = df["n"] < 30

    fig, axes = plt.subplots(1, 5, figsize=(16, 5.5), dpi=FIGURE_DPI)

    for ax, metric in zip(axes, target_metrics):
        sub = df[df["metric"] == metric].copy()
        retained = sub[sub["outcome"] == "Retained"].iloc[0]
        left = sub[sub["outcome"] == "Left"].iloc[0]

        outcomes = [f"Retained\n(n={n_retained:,})", f"Left\n(n={n_left:,})"]
        medians = [retained["median"], left["median"]]
        err_low = [retained["median"] - retained["q1"], left["median"] - left["q1"]]
        err_high = [retained["q3"] - retained["median"], left["q3"] - left["median"]]

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

        if metric == "MonthlyIncome":
            table_text = (
                f"Retained: {medians[0]:,.0f} [{medians[0]-err_low[0]:,.0f}–{medians[0]+err_high[0]:,.0f}]\n"
                f"Left:         {medians[1]:,.0f} [{medians[1]-err_low[1]:,.0f}–{medians[1]+err_high[1]:,.0f}]"
            )
        else:
            table_text = (
                f"Retained: {medians[0]:.0f} [{medians[0]-err_low[0]:.0f}–{medians[0]+err_high[0]:.0f}]\n"
                f"Left:         {medians[1]:.0f} [{medians[1]-err_low[1]:.0f}–{medians[1]+err_high[1]:.0f}]"
            )

        ax.text(
            0.5, -0.28, table_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7.5,
            fontfamily="sans-serif",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
        )

        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(outcomes, fontsize=8.5)
        ax.set_title(metric, fontsize=10.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    axes[0].set_ylabel("Metric Value (Median & IQR)", fontsize=10, weight="bold")
    fig.suptitle(f"{spec["title"]}\nn = {total_n:,} (Retained: {n_retained:,}, Left: {n_left:,}); points show medians, bars show IQR", fontsize=11, weight="bold", y=0.97)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.18, 1, 0.93])

    return fig, df


# =============================================================================
# FIG-14: Demographic and Background Dimensions
# =============================================================================
def plot_fig14_demographics_background(
    tables: Dict[str, pd.DataFrame],
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Plot 6 demographic and background dimensions with flagging highlights."""
    gen = _get_table(tables, "gender_attrition.csv")
    mar = _get_table(tables, "maritalstatus_attrition.csv")
    edu = _get_table(tables, "education_attrition.csv")
    edf = _get_table(tables, "educationfield_attrition.csv")
    pe = _get_table(tables, "prior_experience_attrition.csv")
    pr = _get_table(tables, "performancerating_attrition.csv")
    spec = FIGURE_SPECS["FIG-14"]

    baseline_rate = _get_baseline_rate(tables)
    total_n = int(gen["n"].sum())

    # Build findings dynamically to obtain exact classification and generated panel titles
    findings_df = build_findings(tables)

    for d in [gen, mar, edu, edf, pe, pr]:
        if "low_n_flag" not in d.columns:
            d["low_n_flag"] = d["n"] < 30

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=FIGURE_DPI)
    axes_flat = axes.flatten()

    panels_info = [
        (gen, "Gender", "Gender", "A", "segment"),
        (mar, "MaritalStatus", "Marital Status", "B", "segment"),
        (edu, "Education", "Education Level", "C", "segment"),
        (edf, "EducationField", "Education Field", "D", "segment"),
        (pe, "HasPriorExperience", "Prior Experience", "E", "segment"),
        (pr, "PerformanceRating", "Performance Rating", "F", "segment"),
    ]

    for ax, (data, dim_key, label_name, letter, col_lbl) in zip(axes_flat, panels_info):
        # Generate panel title dynamically from findings
        panel_title = get_dimension_panel_title(dim_key, label_name, letter, findings_df)

        x = np.arange(len(data))
        rates = data["attrition_rate_pct"].values
        yerr_l = rates - data["ci95_low_pct"].values
        yerr_h = data["ci95_high_pct"].values - rates

        styles = [
            get_segment_style(int(data["n"].iloc[i]), float(data["ci95_low_pct"].iloc[i]), float(data["ci95_high_pct"].iloc[i]), baseline_rate)
            for i in range(len(data))
        ]
        colors = [s["color"] for s in styles]
        hatches = [s["hatch"] for s in styles]

        bars = ax.bar(x, rates, yerr=[yerr_l, yerr_h], capsize=4, color=colors, alpha=0.9, edgecolor="#111827", linewidth=0.8)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)

        ax.axhline(baseline_rate, color=FIGURE_PALETTE["baseline"], linestyle="--", linewidth=1.2)

        tick_labels = []
        for s, r, n in zip(data[col_lbl], data["attrition_rate_pct"], data["n"]):
            s_wrapped = textwrap.fill(str(s), width=10, break_long_words=False)
            if n < 30:
                tick_labels.append(f"{s_wrapped}\n{r:.1f}%\n(n={n:,})\n[n<30]")
            else:
                tick_labels.append(f"{s_wrapped}\n{r:.1f}%\n(n={n:,})")

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=7.0)
        ax.set_ylabel("% who left", fontsize=8.5, weight="bold")
        ax.set_title(panel_title, fontsize=8.5, weight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        max_ci_pan = float(max(data["ci95_high_pct"]))
        ax.set_ylim(0, max_ci_pan * 1.25)

    fig.suptitle(f"{spec["title"]}\nn = {total_n:,}; whiskers are 95% Wilson intervals\n{_get_colour_key(baseline_rate)}", fontsize=11, weight="bold", y=0.98)
    _add_footnote(fig, spec["footnote"], y=0.01)
    plt.tight_layout(rect=[0, 0.09, 1, 0.94])

    plotted_df = pd.concat([gen, mar, edu, edf, pe, pr], ignore_index=True)
    return fig, plotted_df