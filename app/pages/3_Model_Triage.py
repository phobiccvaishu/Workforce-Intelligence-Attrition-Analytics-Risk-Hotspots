"""Model Triage page for Workforce Intelligence dashboard.

Presents Phase 4 holdout evaluation benchmarks, model comparisons, calibration
diagnostics, demographic parity audits, and static figures.
Enforces pure presentation: no statistical calculations, no individual employee scores,
and no row-level data.
"""

import sys
from pathlib import Path

# Ensure project root and src are on sys.path for cloud deployment
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st
import pandas as pd

from app.components import (
    render_advisory_banner,
    render_static_filter_notice,
    render_table_download_button,
    render_provenance_footer,
)
from app.data_access import (
    get_holdout_metrics,
    get_model_comparison,
    get_paired_model_differences,
    get_calibration_bins,
    get_demographic_fairness,
    get_verified_figure_path,
)
from workforce_intel.kpis import kpi_10_model_triage_yield

# Advisory Banner
render_advisory_banner()

st.title("Model Triage and Benchmark Evaluation")
st.markdown(
    "Workforce-level out-of-sample holdout validation (n = 294, leavers = 47), model comparison "
    "benchmarks, probability calibration diagnostics, and demographic parity audits."
)

# Static Page Notice
render_static_filter_notice()

# -----------------------------------------------------------------------------
# Section 1: KPI-10 Model Triage Yield
# -----------------------------------------------------------------------------
st.header("KPI-10: Calibrated Model Triage Yield")
st.markdown(
    "Review yield and capture metrics on the held-out sample (20% partition, n = 294, base attrition rate = 16.0%) "
    "using the calibrated baseline Logistic Regression model."
)

kpi10 = kpi_10_model_triage_yield()
val10 = kpi10["value"]
top_10 = val10["top_10_percent"]
top_20 = val10["top_20_percent"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Top 10% Review (k = 29)",
        value=f"{top_10['recall']:.1%} Recall",
        help="Proportion of all holdout leavers identified in the top decile.",
    )
    st.caption(f"Precision: {top_10['precision']:.1%} | Lift: {top_10['lift']:.2f}x | TP: {top_10['tp']}")

with col2:
    st.metric(
        label="Top 20% Review (k = 59)",
        value=f"{top_20['recall']:.1%} Recall",
        help="Proportion of all holdout leavers identified in the top quintile.",
    )
    st.caption(f"Precision: {top_20['precision']:.1%} | Lift: {top_20['lift']:.2f}x | TP: {top_20['tp']}")

with col3:
    st.metric(
        label="Holdout Base Rate",
        value=f"{val10['holdout_base_rate']:.1%}",
        help="Prevalence of attrition in the held-out test split (47 leavers / 294 total).",
    )
    st.caption("Baseline prevalence across holdout partition")

with col4:
    st.metric(
        label="Evaluation Scope",
        value="Holdout Test Set",
        help="Reliability classification: workforce-level, not filterable.",
    )
    st.caption("Reliability: workforce-level, not filterable")

# -----------------------------------------------------------------------------
# Section 2: Model Comparison Benchmarks
# -----------------------------------------------------------------------------
st.header("Holdout Model Performance Benchmarks")
st.markdown(
    "Comparison of benchmark models evaluated on the held-out sample (n = 294). Confidence intervals "
    "are 95% bootstrap intervals across 2,000 resamples."
)

try:
    df_comparison = get_model_comparison()
    st.dataframe(df_comparison, width="stretch")
    render_table_download_button(df_comparison, "model_comparison_benchmarks.csv", "reports/phase4/tables/model_comparison.csv")
except Exception as e:
    st.error(f"Error loading model comparison benchmarks: {e}")

st.subheader("Paired Bootstrap Model Differences (LR vs HGB)")
st.markdown(
    "Paired difference distributions between Logistic Regression and HistGradientBoosting across identical "
    "holdout bootstrap resamples:"
)

try:
    df_paired = get_paired_model_differences()
    lr_hgb_diffs = df_paired[df_paired["comparison"].str.contains("LR (Uncalibrated) - HGB", regex=False)]
    st.dataframe(lr_hgb_diffs, width="stretch")
except Exception as e:
    st.error(f"Error loading paired differences: {e}")

st.info(
    "Confidence intervals for holdout PR-AUC, ROC-AUC, and Brier score overlap zero; empirical performance "
    "between Logistic Regression and HistGradientBoosting does not demonstrate a statistical difference."
)

# -----------------------------------------------------------------------------
# Section 3: Calibration Assessment and Bin Counts
# -----------------------------------------------------------------------------
st.header("Probability Calibration Assessment")
st.markdown(
    "Assessment of predicted risk probabilities against empirical attrition rates. Sigmoid (Platt) scaling "
    "yields consistent Brier scores (0.0995 uncalibrated vs 0.1006 calibrated). Probability intervals with "
    "cell counts under 10 are flagged for small-sample instability."
)

try:
    df_cal_bins = get_calibration_bins()
    st.dataframe(df_cal_bins, width="stretch")
    render_table_download_button(df_cal_bins, "calibration_bins.csv", "reports/phase4/tables/calibration_bins.csv")
except Exception as e:
    st.error(f"Error loading calibration bins: {e}")

# -----------------------------------------------------------------------------
# Section 4: Feature Set Governance (Set A vs Set B)
# -----------------------------------------------------------------------------
st.header("Feature Set Governance Policy")
st.markdown(
    "**Policy Specification**: Set B (operational and job-context features only) is selected and locked for triage. "
    "Set A (which incorporated protected demographic variables Age, Gender, and MaritalStatus) is excluded."
)
st.markdown(
    "Paired holdout validation demonstrates that the difference in PR-AUC between Set A and Set B is "
    "+0.0061 (95% bootstrap CI [-0.0481, 0.0616], including zero). Incorporating protected demographic variables "
    "yields negligible empirical ranking improvement while introducing severe compliance and fairness risks."
)

# -----------------------------------------------------------------------------
# Section 5: Demographic Fairness and Disparate Impact Audit
# -----------------------------------------------------------------------------
st.header("Demographic Subgroup Parity Audit")
st.markdown(
    "Evaluation of selection rates, recall parity, and calibration ratios across protected demographic attributes "
    "at the top-20% review threshold (k = 59). Subgroups with sample sizes under 30 or fewer than 30 leavers "
    "are flagged for cautious interpretation."
)

try:
    df_fairness = get_demographic_fairness()
    st.dataframe(df_fairness, width="stretch")
    render_table_download_button(df_fairness, "demographic_fairness_audit.csv", "reports/phase4/tables/demographic_fairness.csv")
except Exception as e:
    st.error(f"Error loading demographic fairness audit: {e}")

st.warning(
    "Demographic subgroups with small sample sizes (such as AgeGroup 18-25 with n=27 or Divorced with leavers=4) "
    "show wider intervals. Operational monitoring of selection ratios and recall parity across protected classes "
    "is required for governance compliance."
)

# -----------------------------------------------------------------------------
# Section 6: Phase 4 Diagnostic Visualizations (Static Hash-Verified PNGs)
# -----------------------------------------------------------------------------
st.header("Diagnostic Visualizations")
st.markdown("Cryptographically verified static figures generated during Phase 4 model evaluation:")

fig_tabs = st.tabs([
    "Precision-Recall",
    "Calibration",
    "Risk Tiers",
    "Feature Odds Ratios",
    "Hotspot Validation",
    "Demographic Parity",
])

with fig_tabs[0]:
    try:
        p15 = get_verified_figure_path("reports/phase4/figures/fig15_precision_recall_curves.png")
        st.image(str(p15), caption="FIG-15: Precision-Recall Curves across candidate models (holdout split)")
    except Exception as e:
        st.error(f"Figure integrity check failed: {e}")

with fig_tabs[1]:
    try:
        p16 = get_verified_figure_path("reports/phase4/figures/fig16_calibration_curves.png")
        st.image(str(p16), caption="FIG-16: Probability calibration reliability curves (5-bin and 10-bin)")
    except Exception as e:
        st.error(f"Figure integrity check failed: {e}")

with fig_tabs[2]:
    try:
        p17 = get_verified_figure_path("reports/phase4/figures/fig17_risk_tier_attrition.png")
        st.image(str(p17), caption="FIG-17: Empirical attrition rate progression across predicted risk tiers")
    except Exception as e:
        st.error(f"Figure integrity check failed: {e}")

with fig_tabs[3]:
    try:
        p18 = get_verified_figure_path("reports/phase4/figures/fig18_feature_odds_ratios.png")
        st.image(str(p18), caption="FIG-18: Nominal feature odds ratios and bootstrap confidence intervals")
    except Exception as e:
        st.error(f"Figure integrity check failed: {e}")

with fig_tabs[4]:
    try:
        p19 = get_verified_figure_path("reports/phase4/figures/fig19_hotspot_validation.png")
        st.image(str(p19), caption="FIG-19: Split-sample replication of candidate hotspot segments")
    except Exception as e:
        st.error(f"Figure integrity check failed: {e}")

with fig_tabs[5]:
    try:
        p20 = get_verified_figure_path("reports/phase4/figures/fig20_demographic_fairness.png")
        st.image(str(p20), caption="FIG-20: Selection rate and recall parity across protected demographics")
    except Exception as e:
        st.error(f"Figure integrity check failed: {e}")

render_provenance_footer()
