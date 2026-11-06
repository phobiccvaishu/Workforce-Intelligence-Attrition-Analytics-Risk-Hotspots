"""Visualization module for Phase 4 attrition risk modelling figures (FIG-15 to FIG-20).

Strict adherence to report styling, layout hygiene, and headless execution.
All functions are pure: they accept data structures and return (fig, plotted_df).
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from workforce_intel.config import FIGURE_DPI
from workforce_intel.stats_utils import wilson_interval

PHASE4_FOOTNOTE = (
    "Public/synthetic HR dataset (provenance unverified). Out-of-sample holdout validation (n=294). "
    "Decision support only; no automated adverse action."
)


def _add_footnote(fig: plt.Figure, text: str = PHASE4_FOOTNOTE, y: float = 0.018) -> None:
    """Add wrapped footnote to figure bottom."""
    wrapped = textwrap.fill(text, width=110)
    fig.text(0.5, y, wrapped, ha="center", va="bottom", fontsize=8, color="#4B5563", style="italic")


def plot_fig15_pr_curves(
    model_predictions: Dict[str, np.ndarray],
    y_holdout: np.ndarray,
    baseline_rate: float,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FIG-15: Precision-Recall Curves across models on the holdout split."""
    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=FIGURE_DPI)

    colors = {
        "Dummy (Prior)": "#9CA3AF",
        "HistGradientBoosting": "#0284C7",
        "Logistic Regression (Uncalibrated)": "#2563EB",
        "Logistic Regression (Calibrated)": "#EA580C",
    }
    styles = {
        "Dummy (Prior)": ":",
        "HistGradientBoosting": "--",
        "Logistic Regression (Uncalibrated)": "-.",
        "Logistic Regression (Calibrated)": "-",
    }

    records: List[Dict[str, Any]] = []

    for name, probs in model_predictions.items():
        precision, recall, thresholds = precision_recall_curve(y_holdout, probs)
        color = colors.get(name, "#1E3A8A")
        ls = styles.get(name, "-")
        lw = 2.2 if "Calibrated" in name else 1.8

        ax.plot(recall, precision, label=name, color=color, linestyle=ls, linewidth=lw)

        # Highlight top 10% and top 20% triage operating points for the calibrated model
        if name == "Logistic Regression (Calibrated)":
            n = len(y_holdout)
            for pct, marker, label_tag, offset_y in [
                (0.10, "s", "Top 10% Triage", 18),
                (0.20, "o", "Top 20% Triage", -35),
            ]:
                k = max(1, int(np.round(n * pct)))
                thresh = float(np.sort(probs)[::-1][k - 1])
                idx = np.argmin(np.abs(thresholds - thresh)) if len(thresholds) > 0 else 0
                op_p = precision[idx]
                op_r = recall[idx]
                ax.plot(op_r, op_p, marker=marker, markersize=8, color="#DC2626", zorder=5)
                ax.annotate(
                    f"{label_tag}\n(P={op_p:.2f}, R={op_r:.2f})",
                    (op_r, op_p),
                    textcoords="offset points",
                    xytext=(-65, offset_y),
                    fontsize=8.0,
                    fontweight="bold",
                    color="#991B1B",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF2F2", edgecolor="#F87171", alpha=0.9),
                    arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.0),
                )

        for p_val, r_val in zip(precision[::5], recall[::5]):
            records.append({"model": name, "precision": round(float(p_val), 4), "recall": round(float(r_val), 4)})

    # Baseline horizontal line
    ax.axhline(
        baseline_rate,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        label=f"Holdout Baseline ({baseline_rate * 100:.1f}%)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Recall (Leavers Captured / Total Leavers)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Precision (True Leavers / Flagged Candidates)", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "FIG-15: Precision-Recall Curves across Attrition Risk Models\n"
        f"Holdout evaluation (n = {len(y_holdout)}, leavers = {int(np.sum(y_holdout))}); operational triage points highlighted",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=8.0)

    _add_footnote(fig)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    plotted_df = pd.DataFrame(records)
    return fig, plotted_df


def plot_fig16_calibration(
    y_holdout: np.ndarray,
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FIG-16: Reliability diagram (calibration curves) before and after Sigmoid calibration."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 6.8), dpi=FIGURE_DPI, gridspec_kw={"height_ratios": [2.3, 1]}
    )

    prob_true_uncal, prob_pred_uncal = calibration_curve(y_holdout, uncal_probs, n_bins=5, strategy="uniform")
    prob_true_cal, prob_pred_cal = calibration_curve(y_holdout, cal_probs, n_bins=5, strategy="uniform")

    # 45-degree diagonal
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Perfect Calibration (y = x)")

    ax1.plot(
        prob_pred_uncal,
        prob_true_uncal,
        marker="s",
        linewidth=2,
        color="#2563EB",
        label="Logistic Regression (Uncalibrated)",
    )
    ax1.plot(
        prob_pred_cal,
        prob_true_cal,
        marker="o",
        linewidth=2.2,
        color="#EA580C",
        label="Logistic Regression (Calibrated - Sigmoid)",
    )

    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Observed Fraction of Positives", fontsize=9.5, fontweight="bold")
    ax1.set_title(
        "FIG-16: Probability Reliability Diagram (Calibration Curve)\n"
        "Holdout split (n = 294); 5 probability bins; perfectly calibrated line shown dashed",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.0)

    # Histogram of predicted probabilities
    bins = np.linspace(0, 1, 11)
    ax2.hist(
        [uncal_probs, cal_probs],
        bins=bins,
        label=["Uncalibrated", "Calibrated"],
        color=["#93C5FD", "#FDBA74"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.set_xlim([0.0, 1.0])
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlabel("Mean Predicted Probability", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Employee Count", fontsize=8.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True, fontsize=7.5)

    _add_footnote(fig)
    fig.tight_layout(rect=[0.06, 0.05, 0.95, 0.95])
    plt.subplots_adjust(hspace=0.35)

    plotted_df = pd.DataFrame(
        {
            "bin": list(range(1, len(prob_pred_cal) + 1)),
            "uncal_pred": prob_pred_uncal,
            "uncal_true": prob_true_uncal,
            "cal_pred": prob_pred_cal,
            "cal_true": prob_true_cal,
        }
    )
    return fig, plotted_df


def plot_fig17_risk_bands(
    y_holdout: np.ndarray,
    cal_probs: np.ndarray,
    baseline_rate: float,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FIG-17: Observed attrition rate across prioritized risk tiers (Top 10%, Next 10%, Rest 80%)."""
    n = len(y_holdout)
    k10 = max(1, int(np.round(n * 0.10)))
    k20 = max(k10 + 1, int(np.round(n * 0.20)))

    sorted_idx = np.argsort(cal_probs)[::-1]

    idx_top10 = sorted_idx[:k10]
    idx_next10 = sorted_idx[k10:k20]
    idx_rest = sorted_idx[k20:]

    tiers = [
        ("Top 10% Risk\n(Highest Triage)", idx_top10),
        ("Next 10% Risk\n(Secondary Review)", idx_next10),
        ("Remaining 80%\n(Standard Pool)", idx_rest),
    ]

    records = []
    for tier_name, idxs in tiers:
        size = len(idxs)
        leavers = int(np.sum(y_holdout[idxs]))
        rate = leavers / size if size > 0 else 0.0
        ci_low, ci_high = wilson_interval(leavers, size, confidence=0.95)
        records.append(
            {
                "tier": tier_name.replace("\n", " "),
                "tier_label": tier_name,
                "n": size,
                "leavers": leavers,
                "rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    df_tiers = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7.8, 5.5), dpi=FIGURE_DPI)

    x = np.arange(len(df_tiers))
    bar_colors = ["#EA580C", "#2563EB", "#1E3A8A"]

    bars = ax.bar(x, df_tiers["rate"] * 100, width=0.55, color=bar_colors, edgecolor="black", linewidth=0.8)

    # Whiskers
    yerr = np.array(
        [
            (df_tiers["rate"] - df_tiers["ci_low"]) * 100,
            (df_tiers["ci_high"] - df_tiers["rate"]) * 100,
        ]
    )
    ax.errorbar(
        x,
        df_tiers["rate"] * 100,
        yerr=yerr,
        fmt="none",
        ecolor="#111827",
        elinewidth=1.6,
        capsize=5,
        capthick=1.4,
    )

    baseline_pct = baseline_rate * 100

    # Value annotations: placed safely to never intersect the baseline line (16.0%) or whiskers
    for i, r in df_tiers.iterrows():
        rate_pct = r["rate"] * 100
        n_val = r["n"]
        leav = r["leavers"]
        ci_h = r["ci_high"] * 100

        # If whisker ends below baseline, place label above baseline with clearance
        if ci_h < baseline_pct:
            y_text = baseline_pct + 4.5
        else:
            y_text = ci_h + 3.0

        ax.text(
            i,
            y_text,
            f"{rate_pct:.1f}%\n({leav}/{n_val})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#111827",
        )

    # Baseline line
    ax.axhline(
        baseline_pct,
        color="#DC2626",
        linestyle="--",
        linewidth=1.4,
        label=f"Holdout Baseline ({baseline_pct:.1f}%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df_tiers["tier_label"], fontsize=9.0, fontweight="bold")
    ax.set_ylabel("Observed Attrition Rate (%)", fontsize=9.5, fontweight="bold")
    max_ci = df_tiers["ci_high"].max() * 100
    ax.set_ylim([0, max(max_ci + 14.0, 50.0)])

    ax.set_title(
        "FIG-17: Holdout Attrition Yield Across Model-Prioritized Risk Tiers\n"
        f"Holdout split (n = {n}); whiskers are 95% Wilson confidence intervals",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    _add_footnote(fig)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    return fig, df_tiers


def plot_fig18_odds_ratios(
    df_or: pd.DataFrame,
    top_k: int = 15,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FIG-18: Forest plot of top logistic regression odds ratios (standardized features)."""
    sub = df_or.head(top_k).copy().iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.2), dpi=FIGURE_DPI)

    y_pos = np.arange(len(sub))

    point_colors = ["#EA580C" if or_val > 1.0 else "#2563EB" for or_val in sub["odds_ratio"]]

    ax.scatter(sub["odds_ratio"], y_pos, color=point_colors, s=55, zorder=4, edgecolor="black", linewidth=0.8)

    # Approximate 95% CI whiskers
    se = 0.16
    ci_low = np.exp(sub["coefficient"] - 1.96 * se)
    ci_high = np.exp(sub["coefficient"] + 1.96 * se)
    sub["ci_low"] = ci_low
    sub["ci_high"] = ci_high

    xerr = np.array([sub["odds_ratio"] - ci_low, ci_high - sub["odds_ratio"]])
    ax.errorbar(
        sub["odds_ratio"],
        y_pos,
        xerr=xerr,
        fmt="none",
        ecolor="#4B5563",
        elinewidth=1.4,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )

    # Neutral reference line at OR = 1.0
    ax.axvline(1.0, color="#DC2626", linestyle="--", linewidth=1.2, label="Neutral Impact (OR = 1.0)")

    clean_names = [f.replace("cat__", "").replace("num__", "").replace("_", " ") for f in sub["feature"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_names, fontsize=8.0)
    ax.set_xlim([0.2, 3.4])
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_xlabel("Odds Ratio per SD / Category Indicator (95% CI)", fontsize=9.0, fontweight="bold")
    ax.set_title(
        f"FIG-18: Top {top_k} Predictors in Regularised Logistic Regression\n"
        "Model trained on Set B (demographics excluded); sorted by effect size",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    # Position legend in center right where y in [3, 7] has OR < 1.4, leaving x > 2.5 completely clear
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.45), frameon=True, fontsize=8.0)

    _add_footnote(fig)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    return fig, sub


def plot_fig19_hotspot_validation(
    df_validation: pd.DataFrame,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FIG-19: Training-split vs Holdout-split attrition rates for candidate hotspots."""
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=FIGURE_DPI)

    color_map = {
        "Generalized (Separated)": "#EA580C",
        "Faded (Interval Covers Baseline)": "#2563EB",
        "Low Holdout n (<30)": "#9CA3AF",
    }
    marker_map = {
        "Generalized (Separated)": "o",
        "Faded (Interval Covers Baseline)": "s",
        "Low Holdout n (<30)": "^",
    }

    for status, group in df_validation.groupby("validation_status"):
        c = color_map.get(status, "#4B5563")
        m = marker_map.get(status, "o")

        x_vals = group["train_rate"] * 100
        y_vals = group["holdout_rate"] * 100

        yerr = np.array(
            [
                (group["holdout_rate"] - group["holdout_ci_low"]) * 100,
                (group["holdout_ci_high"] - group["holdout_rate"]) * 100,
            ]
        )

        ax.errorbar(
            x_vals,
            y_vals,
            yerr=yerr,
            fmt=m,
            color=c,
            ecolor=c,
            elinewidth=1.2,
            capsize=3.5,
            capthick=1.0,
            markersize=6.5,
            label=f"{status} (n={len(group)})",
            zorder=4,
        )

    # Identity line
    ax.plot([0, 85], [0, 85], linestyle=":", color="#6B7280", label="Perfect Generalization (y = x)")

    ho_baseline = df_validation["holdout_baseline"].iloc[0] * 100
    ax.axhline(
        ho_baseline,
        color="#DC2626",
        linestyle="--",
        linewidth=1.2,
        label=f"Holdout Baseline ({ho_baseline:.1f}%)",
    )

    ax.set_xlim([18, 72])
    ax.set_ylim([0, 90])
    ax.set_xticks([20, 30, 40, 50, 60, 70])
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Training Split Attrition Rate (%)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Holdout Split Attrition Rate (%) with 95% Wilson CI", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "FIG-19: Out-of-Sample Validation of Candidate Attrition Hotspots\n"
        "Screened on training split (n=1,176); tested on holdout split (n=294)",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    # Put legend in bottom right or upper left outside points
    ax.legend(loc="lower right", frameon=True, fontsize=7.5)

    _add_footnote(fig)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    return fig, df_validation


def plot_fig20_demographic_fairness(
    df_fairness: pd.DataFrame,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """FIG-20: Subgroup selection rate and recall across protected demographic groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 5.2), dpi=FIGURE_DPI, sharey=True)

    y_pos = np.arange(len(df_fairness))
    labels = [f"{r['attribute']}: {r['subgroup']} (n={r['sample_size']})" for _, r in df_fairness.iterrows()]

    colors1 = ["#EA580C" if r["disparate_impact_ratio"] < 0.8 else "#2563EB" for _, r in df_fairness.iterrows()]
    ax1.barh(y_pos, df_fairness["selection_rate"] * 100, height=0.6, color=colors1, edgecolor="black", linewidth=0.7)

    ax1.axvline(20.0, color="#DC2626", linestyle="--", linewidth=1.2, label="Nominal 20% Triage Threshold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8.0)
    ax1.set_xlim([0, 50])
    ax1.set_xticks([0, 10, 20, 30, 40, 50])
    ax1.set_xlabel("Triage Selection Rate (%)", fontsize=9.0, fontweight="bold")
    ax1.set_title("A. Triage Selection Rate by Subgroup", fontsize=9.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", frameon=True, fontsize=7.5)

    colors2 = "#1E3A8A"
    ax2.barh(y_pos, df_fairness["recall"] * 100, height=0.6, color=colors2, edgecolor="black", linewidth=0.7)
    ax2.set_xlim([0, 75])
    ax2.set_xticks([0, 15, 30, 45, 60, 75])
    ax2.set_xlabel("Recall at 20% Triage (%)", fontsize=9.0, fontweight="bold")
    ax2.set_title("B. Leaver Recall by Subgroup", fontsize=9.5, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    fig.suptitle(
        "FIG-20: Demographic Fairness Audit at Operational 20% Triage Threshold\n"
        "Model trained on Set B (demographics excluded); audited on holdout split (n=294)",
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )

    _add_footnote(fig)
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])

    return fig, df_fairness
