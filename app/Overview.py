"""Workforce Intelligence Dashboard - Overview & Live KPI Monitoring.

Multipage entry point. Provides interactive filtering across authorized dimensions
and renders LIVE KPIs 1 through 5 evaluated dynamically via workforce_intel.kpis.
"""

import sys
from pathlib import Path

# Ensure project root and src are on sys.path for cloud deployment (e.g. Streamlit Cloud)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import streamlit as st
import pandas as pd

from app.components import (
    render_advisory_banner,
    render_provenance_footer,
    render_kpi_card,
    render_aggregate_download_button,
)
from app.data_access import get_processed_dataset
from app.filters import render_sidebar_filters, render_active_filter_summary
from workforce_intel.kpis import (
    kpi_01_attrition_rate,
    kpi_02_headcount_leavers,
    kpi_03_relative_attrition,
    kpi_04_overtime_association,
    kpi_05_promotion_gap_cohort,
)

st.set_page_config(
    page_title="Workforce Intelligence - Overview",
    page_icon="👥",
    layout="wide",
)

render_advisory_banner()

st.title("Workforce Intelligence: Overview & Live Performance Indicators")
st.markdown(
    "Decision support analytics platform for organizational retention review and workload monitoring. "
    "All reported metrics reflect observational associations across a public synthetic workforce dataset."
)

df = get_processed_dataset()
active_filters = render_sidebar_filters(df)
render_active_filter_summary(active_filters)

# Evaluate the 5 LIVE KPIs
k1 = kpi_01_attrition_rate(df, filters=active_filters)
k2 = kpi_02_headcount_leavers(df, filters=active_filters)
k3 = kpi_03_relative_attrition(df, filters=active_filters)
k4 = kpi_04_overtime_association(df, filters=active_filters)
k5 = kpi_05_promotion_gap_cohort(df, filters=active_filters)

st.markdown("### Primary Workforce Risk Metrics (Live Evaluators)")

col1, col2 = st.columns(2)
with col1:
    render_kpi_card(
        k1,
        help_text="Proportion of leavers within filtered subpopulation with Wilson 95% score interval.",
    )
    render_kpi_card(
        k3,
        help_text="Subpopulation departure rate contrast relative to enterprise baseline (16.12%).",
    )
    render_kpi_card(
        k5,
        help_text="Promotion analysis cohort (Tenure >= 5, non-exec) comparing 6+ years vs 0-5 years.",
    )

with col2:
    render_kpi_card(
        k2,
        help_text="Headcount and total leavers count with workforce representation share.",
    )
    render_kpi_card(
        k4,
        help_text="Crude odds ratio for OverTime within filtered group (Woolf 95% interval).",
    )

st.markdown("---")
st.markdown("### Tabular Summary of Active Live Indicators")

table_rows = [
    {
        "KPI ID": k1["kpi_id"],
        "Name": k1["name"],
        "Value": f"{k1['value']*100:.2f}%" if k1["value"] is not None else "No data",
        "Headcount (n)": k1["n"],
        "95% CI": f"[{k1['ci_low']:.4f}, {k1['ci_high']:.4f}]" if k1["ci_low"] is not None else "-",
        "Reliability": k1["reliability"],
    },
    {
        "KPI ID": k2["kpi_id"],
        "Name": k2["name"],
        "Value": f"Headcount={k2['value']['headcount']}, Leavers={k2['value']['leavers']}" if k2["value"] else "No data",
        "Headcount (n)": k2["n"],
        "95% CI": "-",
        "Reliability": k2["reliability"],
    },
    {
        "KPI ID": k3["kpi_id"],
        "Name": k3["name"],
        "Value": f"{k3['value']['rate_diff']*100:+.2f} pp ({k3['value']['separation_flag']})" if k3["value"] else "No data",
        "Headcount (n)": k3["n"],
        "95% CI": f"[{k3['ci_low']:.4f}, {k3['ci_high']:.4f}]" if k3["ci_low"] is not None else "-",
        "Reliability": k3["reliability"],
    },
    {
        "KPI ID": k4["kpi_id"],
        "Name": k4["name"],
        "Value": f"OR={k4['value']['odds_ratio']:.2f} (OT share={k4['value']['overtime_share']*100:.1f}%)" if k4["value"] and k4["value"]["odds_ratio"] is not None else "OR=None",
        "Headcount (n)": k4["n"],
        "95% CI": f"[{k4['ci_low']:.4f}, {k4['ci_high']:.4f}]" if k4["ci_low"] is not None else "-",
        "Reliability": k4["reliability"],
    },
    {
        "KPI ID": k5["kpi_id"],
        "Name": k5["name"],
        "Value": f"Rate={k5['value']['cohort_attrition_rate']*100:.2f}%, OR={k5['value']['crude_or_6plus_vs_0to5']:.2f}" if k5["value"] and k5["value"]["crude_or_6plus_vs_0to5"] is not None else "OR=None",
        "Headcount (n)": k5["n"],
        "95% CI": f"[{k5['ci_low']:.4f}, {k5['ci_high']:.4f}]" if k5["ci_low"] is not None else "-",
        "Reliability": k5["reliability"],
    },
]

df_live_summary = pd.DataFrame(table_rows)
st.dataframe(df_live_summary, width="stretch", hide_index=True)

render_aggregate_download_button(
    df_live_summary,
    filename="live_kpi_summary.csv",
    label="Download Active KPI Table (CSV)",
)

render_provenance_footer()
