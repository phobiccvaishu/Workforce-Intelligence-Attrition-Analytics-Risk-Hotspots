"""Governance, Transparency, and Limitations page for Workforce Intelligence dashboard.

Presents the formal decision status registry, findings consistency audits, dataset
provenance and structural limitations, master KPI definitions, and reliability classification policy.
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
    get_decision_status,
    get_consistency_audit,
)
from workforce_intel.config import KPI_REGISTRY

# Advisory Banner
render_advisory_banner()

st.title("Methodological Governance and Transparency")
st.markdown(
    "Formal tracking of architectural decisions, exploratory findings replication audits, "
    "data provenance limitations, metric registries, and reliability classification policies."
)

# Static Page Notice
render_static_filter_notice()

# -----------------------------------------------------------------------------
# Section 1: Decision Status Registry
# -----------------------------------------------------------------------------
st.header("Decision Status Registry")
st.markdown(
    "Audit log of formal methodological, statistical, and operational decisions recorded throughout "
    "the analytical lifecycle. Reflects verified consensus states without retroactive alteration."
)

try:
    df_decisions = get_decision_status()
    
    # Status summary metrics
    total_decisions = len(df_decisions)
    approved_count = sum(1 for s in df_decisions["Status"] if s == "APPROVED")
    proposed_count = sum(1 for s in df_decisions["Status"] if s == "PROPOSED")
    deferred_count = sum(1 for s in df_decisions["Status"] if s == "DEFERRED")

    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        st.metric("Total Decisions", total_decisions)
    with scol2:
        st.metric("Approved", approved_count)
    with scol3:
        st.metric("Proposed", proposed_count)
    with scol4:
        st.metric("Deferred", deferred_count)
        
    st.dataframe(df_decisions, width="stretch")
    render_table_download_button(df_decisions, "decision_status_registry.csv", "docs/decision_status.md")
except Exception as e:
    st.error(f"Error loading decision registry: {e}")

# -----------------------------------------------------------------------------
# Section 2: Findings Consistency and Replication Audit
# -----------------------------------------------------------------------------
st.header("Exploratory Findings Replication Audit")
st.markdown(
    "Formal evaluation of all 57 Phase 3 exploratory findings against the held-out sample "
    "and locked feature specifications in Phase 4."
)

try:
    df_consistency = get_consistency_audit()
    status_col = "phase4_status" if "phase4_status" in df_consistency.columns else "status"
    agrees_count = sum(1 for s in df_consistency[status_col] if s == "agrees")
    direction_count = sum(1 for s in df_consistency[status_col] if s == "direction only")
    unstable_count = sum(1 for s in df_consistency[status_col] if s == "unstable")
    set_b_count = sum(1 for s in df_consistency[status_col] if s == "untestable in Set B")

    acol1, acol2, acol3, acol4 = st.columns(4)
    with acol1:
        st.metric("Replicated (Agrees)", agrees_count)
    with acol2:
        st.metric("Direction Consistent", direction_count)
    with acol3:
        st.metric("Unstable Subgroups", unstable_count)
    with acol4:
        st.metric("Excluded from Set B", set_b_count)
        
    st.markdown(
        "- **Agrees (21 findings)**: Observed attrition rate difference replicates in holdout split with overlapping confidence intervals.\n"
        "- **Direction only (18 findings)**: Sign of association holds in holdout, but confidence intervals widen due to reduced holdout sample size.\n"
        "- **Unstable (11 findings)**: Subgroup sample sizes are small (n < 30) or point estimates reverse direction across sample partitions.\n"
        "- **Untestable in Set B (7 findings)**: Findings involving protected demographic features (Age, Gender, MaritalStatus) excluded under model governance."
    )
    
    df_display = df_consistency.copy()
    if "multivariable_effect" in df_display.columns:
        df_display["multivariable_effect"] = df_display["multivariable_effect"].astype(str).str.replace("confirms", "indicates", regex=False)
    st.dataframe(df_display, width="stretch")
    render_table_download_button(df_consistency, "findings_consistency_audit.csv", "reports/phase4/tables/consistency_audit.csv")
except Exception as e:
    st.error(f"Error loading consistency audit: {e}")

# -----------------------------------------------------------------------------
# Section 3: Data Provenance and Structural Limitations
# -----------------------------------------------------------------------------
st.header("Dataset Provenance and Structural Limitations")
st.markdown(
    """
1. **Public Synthetic Provenance**: The source dataset originates from a synthetic benchmark repository (IBM HR Analytics benchmark dataset). Its generating mechanisms and ground-truth parameters are unverified. It represents illustrative organizational records rather than proprietary corporate census data.
2. **Cross-Sectional Observational Structure**: The dataset comprises a single static observation per employee (n = 1,470 records). There is no longitudinal panel, time-series, or event history tracking individual career trajectories over time.
3. **Unrecorded Attrition Voluntariness**: The target outcome (`Attrition = Yes`) records separation without distinguishing between voluntary resignations, performance terminations, workforce reductions, or retirements.
4. **Absence of Survival Windows**: Tenures and promotion intervals are retrospectively reported without censoring dates or uniform cohort entry periods, precluding continuous-time survival analysis.
5. **Absence of Causal Identification**: All reported patterns represent descriptive empirical associations. Observed odds ratios and differences must not be interpreted as causal mechanisms.
"""
)

# -----------------------------------------------------------------------------
# Section 4: Master KPI Registry Definitions
# -----------------------------------------------------------------------------
st.header("Master KPI Registry Definitions")
st.markdown(
    "Standardized definitions and contracts for the 10 workforce analytics indicators (5 live filterable, 5 static workforce-level):"
)

df_registry = pd.DataFrame(KPI_REGISTRY)
st.dataframe(df_registry, width="stretch")

# -----------------------------------------------------------------------------
# Section 5: Reliability Classification Policy (DL-41)
# -----------------------------------------------------------------------------
st.header("Reliability Classification Policy (DL-41)")
st.markdown(
    """
To prevent misleading decision support from small samples or ill-conditioned subsets, every indicator 
is evaluated against formal reliability contracts:

- `reliable`: Sample size satisfies n >= 30, variance is bounded, and intervals are well-defined.
- `low_n`: Subgroup sample size satisfies 1 <= n < 30. Wide confidence intervals reflect high sampling error; estimates must be interpreted with caution.
- `no_data`: Subgroup contains zero observations (n = 0). The indicator is mathematically undefined.
- `exposure_fixed_by_filter`: The exposure variable is held constant by an active filter (e.g., filtering exclusively to OverTime = Yes). Comparative associations are undefined.
- `unreliable_zero_cell`: The cross-tabulation table contains an unobserved cell (zero count). Odds ratios cannot be computed without ad-hoc continuity adjustments.
- `workforce-level, not filterable`: Indicators computed on fixed enterprise partitions or holdout evaluation sets. Dynamic slice filters are not applied.
"""
)

render_provenance_footer()
