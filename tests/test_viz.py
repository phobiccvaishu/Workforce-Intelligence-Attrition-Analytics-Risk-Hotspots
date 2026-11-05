"""Test suite for Phase 3B static visualization engine (src/workforce_intel/viz.py)."""

from pathlib import Path
import pytest
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from workforce_intel.config import (
    FIGURE_SPECS,
    PHASE3_FIGURES_DIR,
    PHASE3_MANIFEST_CSV,
    PHASE3_TABLES_DIR,
    ROUND_DIGITS,
)
import workforce_intel.viz as viz


@pytest.fixture(scope="module")
def loaded_tables():
    """Load all Phase 3A tables for viz testing."""
    tables = {}
    for f in PHASE3_TABLES_DIR.glob("*.csv"):
        tables[f.name] = pd.read_csv(f)
        tables[f.stem] = tables[f.name]
    return tables


def test_figure_files_and_manifest_contract():
    """Verify all 14 figure PNGs exist, have valid headers, 150 DPI, and valid manifest entries."""
    assert PHASE3_MANIFEST_CSV.exists(), "Manifest CSV must exist."
    manifest_df = pd.read_csv(PHASE3_MANIFEST_CSV)

    assert len(manifest_df) == 14, f"Expected 14 manifest rows, got {len(manifest_df)}"
    expected_cols = [
        "figure_id",
        "file",
        "title",
        "source_tables",
        "plotted_rows",
        "plotted_data_sha256",
        "alt_text",
    ]
    assert list(manifest_df.columns) == expected_cols

    for _, row in manifest_df.iterrows():
        fid = row["figure_id"]
        fname = row["file"]
        png_path = PHASE3_FIGURES_DIR / fname

        assert png_path.exists(), f"Figure file {fname} for {fid} does not exist."
        assert png_path.stat().st_size > 5000, f"Figure file {fname} is suspiciously small."

        with Image.open(png_path) as img:
            assert img.format == "PNG", f"{fname} must be a valid PNG."
            dpi = img.info.get("dpi")
            if dpi:
                # 150 dpi with float tolerance
                assert round(dpi[0]) == 150, f"{fname} DPI expected 150, got {dpi[0]}"
            width, height = img.size
            assert width >= 1000 and height >= 500, f"{fname} unexpected dimensions: {img.size}"

        # Verify source tables listed in manifest actually exist
        srcs = [s.strip() for s in row["source_tables"].split(";")]
        for s in srcs:
            tbl_path = PHASE3_TABLES_DIR / s
            assert tbl_path.exists(), f"Source table {s} for {fid} must exist in {PHASE3_TABLES_DIR}"

        # Alt text must be substantive
        assert len(row["alt_text"]) > 20, f"Alt text for {fid} is too short."


def test_plot_functions_contract(loaded_tables):
    """Verify each pure plot function returns (Figure, plotted_df) matching source tables."""
    funcs = [
        ("FIG-01", viz.plot_fig01_department_overview, "department_attrition.csv"),
        ("FIG-02", viz.plot_fig02_jobrole_attrition, "jobrole_attrition.csv"),
        ("FIG-03", viz.plot_fig03_funnel_plot, "funnel_limits.csv"),
        ("FIG-04", viz.plot_fig04_leaver_concentration, "leaver_concentration.csv"),
        ("FIG-05", viz.plot_fig05_age_sensitivity, "agegroup_attrition.csv"),
        ("FIG-06", viz.plot_fig06_tenure_sensitivity, "tenureband_attrition.csv"),
        ("FIG-07", viz.plot_fig07_promotion_gap_cohort, "promotiongap_side_by_side.csv"),
        ("FIG-08", viz.plot_fig08_workload_mobility, "overtime_attrition.csv"),
        ("FIG-09", viz.plot_fig09_satisfaction_small_multiples, "worklifebalance_attrition.csv"),
        ("FIG-10", viz.plot_fig10_overtime_interactions, "two_way_overtime_x_joblevel.csv"),
        ("FIG-11", viz.plot_fig11_hotspot_candidates, "hotspot_candidates.csv"),
        ("FIG-12", viz.plot_fig12_hotspot_overlap, "hotspot_overlap.csv"),
        ("FIG-13", viz.plot_fig13_numeric_distributions, "numeric_distributions_by_outcome.csv"),
        ("FIG-14", viz.plot_fig14_demographics_background, "gender_attrition.csv"),
    ]

    for fid, func, primary_src in funcs:
        fig, plotted_df = func(loaded_tables)
        assert isinstance(fig, plt.Figure), f"{fid} must return a matplotlib Figure"
        assert isinstance(plotted_df, pd.DataFrame), f"{fid} must return a DataFrame"
        assert len(plotted_df) > 0, f"{fid} plotted_df must not be empty"

        # Check plot-data consistency for primary table rows where present
        if primary_src in loaded_tables and "n" in loaded_tables[primary_src].columns:
            src_df = loaded_tables[primary_src]
            if "segment" in src_df.columns and "segment" in plotted_df.columns:
                for seg in src_df["segment"]:
                    src_row = src_df[src_df["segment"] == seg].iloc[0]
                    plot_row = plotted_df[plotted_df["segment"] == seg]
                    if len(plot_row) > 0:
                        p_row = plot_row.iloc[0]
                        assert int(p_row["n"]) == int(src_row["n"]), f"n mismatch in {fid} for {seg}"
                        assert int(p_row["leavers"]) == int(src_row["leavers"]), f"leavers mismatch in {fid} for {seg}"
                        if "attrition_rate_pct" in p_row:
                            assert round(float(p_row["attrition_rate_pct"]), ROUND_DIGITS) == round(float(src_row["attrition_rate_pct"]), ROUND_DIGITS)

        plt.close(fig)


def test_missing_table_error_handling(loaded_tables):
    """Verify informative KeyError is raised when required table is missing."""
    empty_tables = {}
    with pytest.raises(KeyError, match="department_attrition"):
        viz.plot_fig01_department_overview(empty_tables)

    with pytest.raises(KeyError, match="jobrole_attrition"):
        viz.plot_fig02_jobrole_attrition({"overall_baseline.csv": loaded_tables["overall_baseline.csv"]})
