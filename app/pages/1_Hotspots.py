"""Workforce Intelligence Dashboard - Overlap-Aware Hotspot Index.

Static presentation of Phase 5 Empirical Bayes hotspot screening, split-sample
out-of-sample replication validation, and greedy marginal coverage progression.
Verified via SHA-256 integrity checksums. Active sidebar filters do not apply.
"""

import sys
from pathlib import Path

# Ensure project root and src are on sys.path for cloud deployment
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.components import (
    render_advisory_banner,
    render_provenance_footer,
    render_static_page_filter_notice,
    render_aggregate_download_button,
)
from app.data_access import (
    get_hotspot_universe,
    get_hotspot_split_validation,
    get_greedy_coverage,
    get_static_table,
)
from workforce_intel.kpis import (
    kpi_07_hotspot_universe_summary,
    kpi_08_hotspot_holdout_validation,
    kpi_09_hotspot_coverage,
)

st.set_page_config(
    page_title="Workforce Intelligence - Hotspots",
    page_icon="🎯",
    layout="wide",
)

render_advisory_banner()
render_static_page_filter_notice()

st.title("Overlap-Aware Hotspot Index & Out-of-Sample Validation")
st.markdown(
    "Empirical Bayes shrinkage index and non-redundant greedy coverage algorithm "
    "designed for proactive retention review and operational workload audits. "
    "All demographic attributes are strictly excluded from hotspot definitions."
)

# 1. KPI-07: Hotspot Universe Summary
k7 = kpi_07_hotspot_universe_summary()
val7 = k7["value"]

st.markdown("### 1. Hotspot Universe Architecture (KPI-07)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Screened Segments", value=f"{val7['universe_size']}")
with col2:
    st.metric(label="Sample Size Floor Met (n >= 30)", value=f"{val7['segments_n_ge_30']}")
with col3:
    st.metric(label="Tier 1 Priority Hotspots", value=f"{val7['tier1_priority_hotspots']}")
with col4:
    st.metric(label="Tier 2 Secondary Hotspots", value=f"{val7['tier2_secondary_hotspots']}")

st.caption(
    f"Empirical Bayes prior strength estimated via marginal maximum likelihood: "
    f"m = {val7['fitted_prior_m_1way']:.2f} (one-way families), m = {val7['fitted_prior_m_2way']:.2f} (two-way interactions). "
    f"Prior mean anchored at enterprise baseline (16.12%)."
)

# Load verified tables
df_universe = get_hotspot_universe()
df_split = get_hotspot_split_validation()
df_greedy = get_greedy_coverage()
df_sens = get_static_table("reports/phase5/tables/hotspot_sensitivity.csv")

# 2. Hotspot Universe Table & Interval Plot
st.markdown("### 2. Tier 1 Priority Hotspot Rankings")
t1_df = df_universe[df_universe["is_tier1"]].sort_values("hotspot_index", ascending=False).copy()

disp_cols = ["segment_id", "family", "n", "k", "raw_rate", "ci_low", "ci_high", "shrunken_rate", "hotspot_index"]
t1_disp = t1_df[disp_cols].copy()
t1_disp.columns = ["Segment Identifier", "Family", "Headcount (n)", "Leavers (k)", "Observed Rate", "95% CI Lower", "95% CI Upper", "Shrunken Rate", "Hotspot Index"]
t1_disp["Observed Rate"] = (t1_disp["Observed Rate"] * 100).round(1).astype(str) + "%"
t1_disp["95% CI Lower"] = (t1_disp["95% CI Lower"] * 100).round(1).astype(str) + "%"
t1_disp["95% CI Upper"] = (t1_disp["95% CI Upper"] * 100).round(1).astype(str) + "%"
t1_disp["Shrunken Rate"] = (t1_disp["Shrunken Rate"] * 100).round(1).astype(str) + "%"
t1_disp["Hotspot Index"] = t1_disp["Hotspot Index"].round(1)

st.dataframe(t1_disp, width="stretch", hide_index=True)

# One interval plot: Top 10 Tier 1 Hotspot Confidence Intervals
fig_intervals = go.Figure()
top10_t1 = t1_df.head(10).iloc[::-1]

fig_intervals.add_trace(go.Scatter(
    x=top10_t1["raw_rate"] * 100,
    y=top10_t1["segment_id"],
    mode="markers",
    name="Observed Rate",
    marker=dict(color="#B91C1C", size=9),
    error_x=dict(
        type="data",
        symmetric=False,
        array=(top10_t1["ci_high"] - top10_t1["raw_rate"]) * 100,
        arrayminus=(top10_t1["raw_rate"] - top10_t1["ci_low"]) * 100,
        color="#B91C1C",
        width=4,
    )
))
fig_intervals.add_vline(x=16.12, line_dash="dash", line_color="#4B5563", annotation_text="Enterprise Baseline (16.12%)")
fig_intervals.update_layout(
    title="Top 10 Tier 1 Hotspots: Observed Departure Rates and Wilson 95% Intervals",
    xaxis_title="Departure Rate (%)",
    yaxis_title="Segment",
    height=420,
    margin=dict(l=10, r=10, t=40, b=30),
)
st.plotly_chart(fig_intervals, width="stretch")

render_aggregate_download_button(
    t1_disp,
    filename="tier1_hotspots.csv",
    label="Download Tier 1 Hotspot Table (CSV)",
)

st.markdown("---")

# 3. KPI-08: Split-Sample Validation
st.markdown("### 3. Split-Sample Out-of-Sample Validation (KPI-08)")
k8 = kpi_08_hotspot_holdout_validation()
val8 = k8["value"]

st.markdown(
    "To rigorously guard against selection optimism, candidate hotspots screened on the training partition "
    f"(n = 1,176; leavers = 190) were sequestered and audited on the held-out validation sample "
    f"(n = 294; leavers = 47; holdout baseline = 15.99%)."
)

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.metric(
        label="Training-Screened Segments Meeting Floor",
        value=f"{val8['holdout_sample_floor_met']} / {val8['screened_training_tier1']}",
        help="Segments meeting the sample size floor (holdout n >= 30) out of 25 screened training segments.",
    )
with sc2:
    st.metric(
        label="Replication Above Point Estimate",
        value=f"{val8['holdout_above_baseline_point']} / {val8['holdout_sample_floor_met']} (85.7%)",
        help="Proportion of floor-cleared segments exceeding the 15.99% holdout baseline by point estimate.",
    )
with sc3:
    st.metric(
        label="Replication Above Wilson Lower Bound",
        value=f"{val8['holdout_above_baseline_lower_bound']} / {val8['holdout_sample_floor_met']} (50.0%)",
        help="Proportion of floor-cleared segments whose Wilson 95% lower bound exceeds the holdout baseline.",
    )

st.markdown(
    "**Governance Label**: Full-data rows are labeled `optimistic: holdout was used to select`. "
    "Segments experience expected statistical regression toward the population mean upon holdout evaluation."
)

split_disp = df_split[[
    "segment_id", "train_n", "train_rate", "holdout_n", "holdout_k", "holdout_rate",
    "holdout_n_ge_30", "above_holdout_point", "above_holdout_ci_low", "in_full_tier1"
]].copy()
split_disp.columns = [
    "Segment Identifier", "Train n", "Train Rate", "Holdout n", "Holdout Leavers",
    "Holdout Rate", "Holdout n >= 30", "Above Point", "Above Lower Bound", "In Full Tier 1"
]
split_disp["Train Rate"] = (split_disp["Train Rate"] * 100).round(1).astype(str) + "%"
split_disp["Holdout Rate"] = (split_disp["Holdout Rate"] * 100).round(1).astype(str) + "%"
split_disp["Holdout n >= 30"] = split_disp["Holdout n >= 30"].map({True: "Yes", False: "No"})
split_disp["Above Point"] = split_disp["Above Point"].map({True: "Yes", False: "No"})
split_disp["Above Lower Bound"] = split_disp["Above Lower Bound"].map({True: "Yes", False: "No"})
split_disp["In Full Tier 1"] = split_disp["In Full Tier 1"].map({True: "Yes", False: "No (Optimistic)"})

st.dataframe(split_disp, width="stretch", hide_index=True)

render_aggregate_download_button(
    split_disp,
    filename="hotspot_split_validation.csv",
    label="Download Split-Sample Validation Table (CSV)",
)

st.markdown("---")

# 4. KPI-09: Greedy Marginal Coverage
st.markdown("### 4. Non-Redundant Greedy Marginal Coverage (KPI-09)")
k9 = kpi_09_hotspot_coverage()
val9 = k9["value"]

st.markdown(
    "**Candidate Overlap Reality**: Candidate risk segments overlap extensively and are not independent. "
    "An iterative greedy marginal coverage algorithm was applied to construct a concise, non-redundant review queue. "
    f"At step 5, the queue accounts for **{val9['step_5_leavers_share']*100:.1f}% of workforce leavers** across **{val9['step_5_workforce_share']*100:.1f}% of total employees**, "
    f"reaching **{val9['final_leavers_share']*100:.1f}% of leavers** across **{val9['final_workforce_share']*100:.1f}% of headcount** across all {val9['total_steps']} steps."
)

# Coverage curve chart
fig_cov = go.Figure()
fig_cov.add_trace(go.Scatter(
    x=df_greedy["step"],
    y=df_greedy["share_of_leavers"] * 100,
    mode="lines+markers",
    name="Cumulative Leaver Coverage (%)",
    line=dict(color="#B91C1C", width=3),
    marker=dict(size=8),
))
fig_cov.add_trace(go.Scatter(
    x=df_greedy["step"],
    y=df_greedy["share_of_workforce"] * 100,
    mode="lines+markers",
    name="Cumulative Workforce Headcount (%)",
    line=dict(color="#1D4ED8", width=3, dash="dot"),
    marker=dict(size=8),
))
fig_cov.update_layout(
    title="Greedy Marginal Coverage Progression across Selection Steps",
    xaxis=dict(title="Greedy Selection Step", tickmode="linear", dtick=1),
    yaxis=dict(title="Cumulative Proportion (%)", range=[0, 100]),
    height=380,
    margin=dict(l=10, r=10, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_cov, width="stretch")

greedy_disp = df_greedy[[
    "step", "segment_id", "new_employees", "new_leavers", "marginal_rate",
    "cum_employees", "cum_leavers", "share_of_workforce", "share_of_leavers"
]].copy()
greedy_disp.columns = [
    "Step", "Selected Segment", "New Headcount", "New Leavers", "Marginal Rate",
    "Cumulative Headcount", "Cumulative Leavers", "Workforce Share", "Leaver Share"
]
greedy_disp["Marginal Rate"] = (greedy_disp["Marginal Rate"] * 100).round(1).astype(str) + "%"
greedy_disp["Workforce Share"] = (greedy_disp["Workforce Share"] * 100).round(1).astype(str) + "%"
greedy_disp["Leaver Share"] = (greedy_disp["Leaver Share"] * 100).round(1).astype(str) + "%"

st.dataframe(greedy_disp, width="stretch", hide_index=True)

render_aggregate_download_button(
    greedy_disp,
    filename="greedy_coverage.csv",
    label="Download Greedy Coverage Progression Table (CSV)",
)

# 5. Prior Sensitivity Analysis
st.markdown("### 5. Empirical Bayes Prior Strength Sensitivity")
st.markdown(
    "Audits stability across varying prior strength parameters m in {10, 30, 100}. "
    "Higher values of m exert stronger shrinkage toward the enterprise baseline rate (16.12%)."
)

sens_disp = df_sens.copy()
st.dataframe(sens_disp, width="stretch", hide_index=True)

render_provenance_footer()
