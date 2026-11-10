"""Sidebar filtering system for the Workforce Intelligence dashboard.

Constructs interactive controls across authorized dimensions in
src.workforce_intel.config.ALLOWED_FILTER_DIMENSIONS.
Enforces governance separation: protected demographic attributes are isolated
under a dedicated 'Descriptive context only' group.
Persists filter selections across multipage navigation via streamlit.session_state.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

from workforce_intel.config import ALLOWED_FILTER_DIMENSIONS

# Explicit governance classification of allowed dimensions
PROTECTED_DEMOGRAPHIC_DIMENSIONS = ["AgeGroup", "Gender", "MaritalStatus"]

OPERATIONAL_FILTER_DIMENSIONS = [
    dim for dim in sorted(ALLOWED_FILTER_DIMENSIONS)
    if dim not in PROTECTED_DEMOGRAPHIC_DIMENSIONS
]


def init_filter_session_state() -> None:
    """Initialize persistent filter dictionary in session state if not already set."""
    if "active_filters" not in st.session_state:
        st.session_state["active_filters"] = {}


def reset_filters() -> None:
    """Clear all active filter selections in session state."""
    st.session_state["active_filters"] = {}


def render_sidebar_filters(df: pd.DataFrame) -> Dict[str, Any]:
    """Render interactive sidebar filter controls and return active filter dictionary."""
    init_filter_session_state()

    st.sidebar.title("Workforce Filters")
    st.sidebar.caption("Conjunctive filter criteria (AND logic across dimensions)")

    if st.sidebar.button("Reset all filters", width="stretch"):
        reset_filters()
        st.rerun()

    active: Dict[str, Any] = {}

    # 1. Operational & Role Attributes
    st.sidebar.markdown("### Operational & Role Filters")
    for dim in OPERATIONAL_FILTER_DIMENSIONS:
        if dim not in df.columns:
            continue
        # Extract unique domain levels preserving natural or alphabetical order
        levels = sorted(list(df[dim].dropna().unique()), key=lambda x: (str(type(x)), x))
        options = ["All"] + [str(x) for x in levels]

        current_val = st.session_state["active_filters"].get(dim)
        default_idx = 0
        if current_val is not None and str(current_val) in options:
            default_idx = options.index(str(current_val))

        selected = st.sidebar.selectbox(
            label=f"{dim}",
            options=options,
            index=default_idx,
            key=f"filter_operational_{dim}",
        )
        if selected != "All":
            # Match the original column data type (e.g. int vs str)
            orig_match = next(x for x in levels if str(x) == selected)
            active[dim] = orig_match

    # 2. Descriptive Demographic Attributes (Protected Context)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Descriptive Context Only")
    st.sidebar.info(
        "Notice: Demographic attributes are provided exclusively for descriptive context "
        "and baseline review. They are strictly excluded from automated decision workflows "
        "and operational risk screening."
    )
    for dim in PROTECTED_DEMOGRAPHIC_DIMENSIONS:
        if dim not in df.columns:
            continue
        levels = sorted(list(df[dim].dropna().unique()), key=lambda x: str(x))
        options = ["All"] + [str(x) for x in levels]

        current_val = st.session_state["active_filters"].get(dim)
        default_idx = 0
        if current_val is not None and str(current_val) in options:
            default_idx = options.index(str(current_val))

        selected = st.sidebar.selectbox(
            label=f"{dim} (Descriptive)",
            options=options,
            index=default_idx,
            key=f"filter_demographic_{dim}",
        )
        if selected != "All":
            orig_match = next(x for x in levels if str(x) == selected)
            active[dim] = orig_match

    # Update session state with current active selections
    st.session_state["active_filters"] = active
    return active


def render_active_filter_summary(active_filters: Dict[str, Any]) -> None:
    """Render a clean summary banner of active filter criteria on the page."""
    if not active_filters:
        st.markdown("**Active Filter Scope**: `Unfiltered Workforce Baseline (n = 1,470)`")
    else:
        filter_pills = [f"`{k} = {v}`" for k, v in active_filters.items()]
        st.markdown(f"**Active Filter Scope ({len(active_filters)} criteria)**: " + " AND ".join(filter_pills))
