"""Reusable UI presentation components for the Workforce Intelligence dashboard.

Adheres strictly to project governance rules:
1. No statistical computation inside app/ (pure presentation layer).
2. Non-negotiable advisory banner on all decision-support views.
3. Reliability status always rendered as explicit text (not color alone).
4. Zero forbidden words, zero dollar signs, aggregates only.
"""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

ADVISORY_BANNER_TEXT = (
    "Advisory decision support for retention review and workload planning. "
    "Not for automated or individual adverse personnel decisions."
)

PROVENANCE_FOOTER_TEXT = (
    "Data Governance Notice: Observational analytics conducted on a public synthetic workforce dataset. "
    "Cross-sectional records with unrecorded attrition voluntariness and no longitudinal time window. "
    "All outputs represent observational associations for planning review, not individual classifications."
)


def render_advisory_banner() -> None:
    """Render the standard non-negotiable advisory decision support banner."""
    st.info(f"**Governance Standard**: {ADVISORY_BANNER_TEXT}")


def render_provenance_footer() -> None:
    """Render the public/synthetic dataset provenance and methodological scope note."""
    st.markdown("---")
    st.caption(PROVENANCE_FOOTER_TEXT)


def render_static_page_filter_notice() -> None:
    """Render explicit notice that active sidebar filters are not applied to static benchmarks."""
    st.warning(
        "**Benchmark Filter Policy**: Active subpopulation filters do not apply to this view. "
        "The metrics and tables presented on this page are pre-computed workforce-level benchmarks "
        "or split-sample inferential validations verified by SHA-256 integrity checks."
    )


RELIABILITY_BADGE_TEXT: Dict[str, str] = {
    "reliable": "Reliable (n >= 30, robust cell sample size)",
    "low_n": "Low Sample Size (n < 30 or sparse contingency cells; unranked)",
    "no_data": "Empty Subpopulation (n = 0 matching records)",
    "exposure_fixed_by_filter": "Exposure Fixed by Active Filter (contrast unavailable; odds ratio undefined)",
    "unreliable_zero_cell": "Sparse Zero Cell (odds ratio undefined in calculation paths)",
    "workforce-level, not filterable": "Workforce-Level Benchmark (static pre-computed)",
}


def format_reliability_badge(reliability: str) -> str:
    """Format reliability label into a clear text descriptor."""
    return RELIABILITY_BADGE_TEXT.get(reliability, reliability)


# Aliases for consistent naming across views
render_static_filter_notice = render_static_page_filter_notice


def format_kpi_card_display(kpi_res: Dict[str, Any]) -> Dict[str, str]:
    """Pure formatting helper returning textual strings for KPI display without rendering widgets."""
    kpi_id = kpi_res.get("kpi_id", "")
    name = kpi_res.get("name", "")
    val = kpi_res.get("value")
    n = kpi_res.get("n")
    ci_low = kpi_res.get("ci_low")
    ci_high = kpi_res.get("ci_high")
    rel = kpi_res.get("reliability", "reliable")

    if val is None or rel == "no_data":
        val_str = "No data (n = 0)"
    elif isinstance(val, float):
        val_str = f"{val * 100:.1f}%"
    else:
        val_str = str(val)

    n_str = f"n = {n:,}" if n is not None else "n = -"
    if ci_low is not None and ci_high is not None:
        ci_str = f"[{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]"
    else:
        ci_str = "-"

    return {
        "title": f"{kpi_id}: {name}",
        "value_str": val_str,
        "n_str": n_str,
        "ci_str": ci_str,
        "badge_text": rel,
    }


def render_kpi_card(
    kpi_dict: Dict[str, Any],
    help_text: Optional[str] = None,
) -> None:
    """Render a standardized KPI card displaying metric name, value, sample size, CI, and reliability."""
    kpi_id = kpi_dict.get("kpi_id", "")
    name = kpi_dict.get("name", "")
    val = kpi_dict.get("value")
    n = kpi_dict.get("n")
    ci_low = kpi_dict.get("ci_low")
    ci_high = kpi_dict.get("ci_high")
    reliability = kpi_dict.get("reliability", "reliable")

    with st.container(border=True):
        st.subheader(f"{kpi_id}: {name}")
        if help_text:
            st.caption(help_text)

        # Value display
        if val is None or reliability == "no_data":
            st.metric(label="Metric Value", value="No data (n = 0)")
        elif kpi_id == "KPI-01":
            rate_pct = f"{val * 100:.2f}%" if isinstance(val, (int, float)) else str(val)
            st.metric(label="Observed Attrition Rate", value=rate_pct)
        elif kpi_id == "KPI-02" and isinstance(val, dict):
            st.metric(
                label="Filtered Subpopulation Headcount",
                value=f"{val.get('headcount', 0)} employees",
                delta=f"{val.get('leavers', 0)} leavers ({val.get('headcount_share', 0.0)*100:.1f}% workforce share)",
                delta_color="off",
            )
        elif kpi_id == "KPI-03" and isinstance(val, dict):
            rate_diff = val.get("rate_diff")
            diff_pp = rate_diff * 100 if rate_diff is not None else None
            ratio = val.get("rate_ratio")
            flag = val.get("separation_flag", "no_separation")
            val_str = f"{diff_pp:+.2f} pp ({ratio:.2f}x baseline)" if diff_pp is not None and ratio is not None else "No data"
            st.metric(label="Relative Attrition vs Enterprise Baseline", value=val_str)
            st.markdown(f"**Separation Status**: `{flag}`")
        elif kpi_id == "KPI-04" and isinstance(val, dict):
            odds_ratio = val.get("odds_ratio")
            ot_share = val.get("overtime_share", 0.0)
            if odds_ratio is not None:
                st.metric(
                    label="Crude OverTime Odds Ratio",
                    value=f"OR = {odds_ratio:.2f}",
                    delta=f"Exposure Share: {ot_share*100:.1f}%",
                    delta_color="off",
                )
            else:
                st.metric(label="Crude OverTime Odds Ratio", value="OR = None")
                st.caption(f"OverTime exposure share in filtered group: {ot_share*100:.1f}%")
        elif kpi_id == "KPI-05" and isinstance(val, dict):
            crate = val.get("cohort_attrition_rate")
            cor = val.get("crude_or_6plus_vs_0to5")
            chc = val.get("cohort_headcount", 0)
            clv = val.get("cohort_leavers", 0)
            rate_disp = f"{crate*100:.2f}%" if crate is not None else "No data"
            st.metric(
                label="Promotion Cohort Rate (Tenure >= 5)",
                value=rate_disp,
                delta=f"{clv} leavers / {chc} in cohort",
                delta_color="off",
            )
            if cor is not None:
                st.markdown(f"**Crude Contrast (6+ vs 0-5 years)**: OR = {cor:.2f}")
            else:
                st.markdown("**Crude Contrast (6+ vs 0-5 years)**: OR = None")
        else:
            st.write(val)

        # Denominator & Confidence Interval
        col_n, col_ci = st.columns(2)
        with col_n:
            st.markdown(f"**Evaluated Headcount (n)**: {n if n is not None else '-'}")
        with col_ci:
            if ci_low is not None and ci_high is not None:
                st.markdown(f"**95% Score Interval**: `[{ci_low:.4f}, {ci_high:.4f}]`")
            else:
                st.markdown("**95% Score Interval**: `-`")

        # Textual reliability classification (never color alone)
        st.markdown(f"**Reliability Status**: `{format_reliability_badge(reliability)}`")


def render_aggregate_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download Aggregate Table (CSV)",
    provenance_note: str = "# Source: Workforce Intelligence Analytics Platform (Aggregated summary view; individual records suppressed)",
) -> None:
    """Provide CSV download strictly for aggregated summary tables with a provenance comment header."""
    csv_content = f"{provenance_note}\n" + df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv_content.encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# Alias
render_table_download_button = render_aggregate_download_button
