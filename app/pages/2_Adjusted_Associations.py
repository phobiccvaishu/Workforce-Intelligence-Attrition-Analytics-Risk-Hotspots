"""Workforce Intelligence Dashboard - Adjusted Associations & Multiplicity Audits.

Static presentation of Phase 5 stratified Mantel-Haenszel epidemiological adjustments,
unpenalised multivariable logistic regressions (Models M1 and M2), and Benjamini-Hochberg
multiplicity family controls across 28 organizational dimensions.
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
import streamlit as st

from app.components import (
    render_advisory_banner,
    render_provenance_footer,
    render_static_page_filter_notice,
    render_aggregate_download_button,
)
from app.data_access import (
    get_overtime_comparison,
    get_logistic_models,
    get_bh_family_audit,
)
from workforce_intel.kpis import kpi_06_adjusted_overtime

st.set_page_config(
    page_title="Workforce Intelligence - Adjusted Associations",
    page_icon="⚖️",
    layout="wide",
)

render_advisory_banner()
render_static_page_filter_notice()

st.title("Adjusted Associations & Epidemiological Adjustment")
st.markdown(
    "Formal inferential evaluation comparing crude observational disparities against stratified "
    "and multivariable adjustments. Demonstrates whether the OverTime disparity persists across "
    "organizational strata and multivariable conditioning."
)

st.info(
    "**Inferential Governance Principle**: Descriptive association; adjusted estimates differ by method; "
    "no single causal reading. Confounding adjustments reduce bias from observed covariates but cannot "
    "rule out unmeasured confounding or non-random workload allocation."
)

# 1. KPI-06 OverTime Association Comparison
k6 = kpi_06_adjusted_overtime()
val6 = k6["value"]

st.markdown("### 1. OverTime Disparity Across Methodological Frameworks (KPI-06)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        label="MH Stratified (JobLevel)",
        value=f"{val6['mantel_haenszel_joblevel']['odds_ratio']:.2f}",
        help="Mantel-Haenszel common odds ratio stratified by JobLevel (5 strata).",
    )
with c2:
    st.metric(
        label="Multivariable M1 (Operational)",
        value=f"{val6['multivariable_m1']['odds_ratio']:.2f}",
        help="Unpenalised logistic regression conditioned on 11 operational covariates. Max VIF = 2.32.",
    )
with c3:
    st.metric(
        label="Multivariable M2 (+ Controls)",
        value=f"{val6['multivariable_m2']['odds_ratio']:.2f}",
        help="Model M1 with Age, Gender, and MaritalStatus included as adjustment controls.",
    )
with c4:
    st.metric(
        label="Phase 4 Regularised Contrast",
        value=f"{val6['phase4_regularised_contrast']['odds_ratio']:.2f}",
        help="L2-penalised symmetric contrast (Yes vs No) from 500 bootstrap refits. 95% CI: [2.97, 5.51].",
    )

df_ot_comp = get_overtime_comparison()
st.dataframe(df_ot_comp, width="stretch", hide_index=True)

st.markdown(
    "**Mantel-Haenszel Zero-Cell Exclusion Policy**: Strata with zero exposed or zero unexposed records "
    "are excluded from pooled Mantel-Haenszel calculations without adding continuity correction to the pooled estimate. "
    "In the PromotionAnalysisCohort evaluation, tenure stratum 4-5 contains zero exposed employees and was excluded. "
    "Continuity corrections (+0.5) are strictly restricted to displayed stratum-specific breakdown rows containing zero cells."
)

render_aggregate_download_button(
    df_ot_comp,
    filename="overtime_methodology_comparison.csv",
    label="Download OverTime Comparison Table (CSV)",
)

st.markdown("---")

# 2. Multivariable Logistic Regression Parameter Estimates
st.markdown("### 2. Adjusted Multivariable Logistic Regression (Models M1 and M2)")
st.markdown(
    "Two unpenalised logistic regression specifications were estimated to evaluate whether OverTime "
    "remains associated with attrition when controlling for key operational and satisfaction dimensions:  \n"
    "- **Model M1 (Core Operational)**: 11 parameters (21.55 events per parameter). Max VIF = 2.32.  \n"
    "- **Model M2 (With Demographics)**: 15 parameters (15.80 events per parameter), adding Age, Gender, and MaritalStatus controls. Max VIF = 2.34."
)

df_m1_m2 = get_logistic_models()
disp_m = df_m1_m2[["model", "term", "coef", "std_err", "z", "p_value", "odds_ratio", "or_ci_low", "or_ci_high", "vif"]].copy()
disp_m.columns = ["Model", "Term", "Coefficient", "Std Error", "z-statistic", "p-value", "Odds Ratio", "OR 95% CI Lower", "OR 95% CI Upper", "VIF"]
disp_m["Coefficient"] = disp_m["Coefficient"].round(4)
disp_m["Std Error"] = disp_m["Std Error"].round(4)
disp_m["z-statistic"] = disp_m["z-statistic"].round(2)
disp_m["p-value"] = disp_m["p-value"].apply(lambda x: f"{x:.4e}")
disp_m["Odds Ratio"] = disp_m["Odds Ratio"].round(2)
disp_m["OR 95% CI Lower"] = disp_m["OR 95% CI Lower"].round(2)
disp_m["OR 95% CI Upper"] = disp_m["OR 95% CI Upper"].round(2)
disp_m["VIF"] = disp_m["VIF"].round(2)

st.dataframe(disp_m, width="stretch", hide_index=True)

render_aggregate_download_button(
    disp_m,
    filename="logistic_m1_m2_parameters.csv",
    label="Download Multivariable Model Parameters (CSV)",
)

st.markdown("---")

# 3. Benjamini-Hochberg Multiplicity Family Audit
st.markdown("### 3. Multiplicity Control: Benjamini-Hochberg FDR Audit (28 Dimensions)")
st.markdown(
    "To control the False Discovery Rate across exploratory subgroup analyses, 28 discrete organizational "
    "dimensions were pre-registered into a unified categorical/ordinal family (family size m = 28, alpha = 0.05).  \n"
    "**Audit Outcome**: Exactly 23 dimensions clear the multiplicity threshold of BH-adjusted q < 0.05. "
    "Exactly 5 dimensions do not meet FDR clearance: HasPriorExperience (p = 0.0582), RelationshipSatisfaction (p = 0.1550), "
    "Gender (p = 0.2778), Education (p = 0.5455), and PerformanceRating (p = 0.9217)."
)

df_bh = get_bh_family_audit()
disp_bh = df_bh.copy()
disp_bh.columns = ["Dimension", "Levels", "Chi2 Stat", "p-value", "BH q-value", "FDR Cleared", "Population Scope"]
disp_bh["Chi2 Stat"] = disp_bh["Chi2 Stat"].round(2)
disp_bh["p-value"] = disp_bh["p-value"].apply(lambda x: f"{x:.4e}")
disp_bh["BH q-value"] = disp_bh["BH q-value"].apply(lambda x: f"{x:.4e}")
disp_bh["FDR Cleared"] = disp_bh["FDR Cleared"].map({True: "Cleared (q < 0.05)", False: "Not Met"})

st.dataframe(disp_bh, width="stretch", hide_index=True)

render_aggregate_download_button(
    disp_bh,
    filename="bh_family_audit.csv",
    label="Download Multiplicity Family Audit (CSV)",
)

render_provenance_footer()
