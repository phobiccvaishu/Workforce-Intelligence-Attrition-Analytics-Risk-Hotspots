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

def test_figure_hatched_bars_match_low_n_count(loaded_tables):
    """Verify that hatched bar/patch count equals low_n_flag count in plotted DataFrame for all figures."""
    funcs = [
        ("FIG-01", viz.plot_fig01_department_overview),
        ("FIG-02", viz.plot_fig02_jobrole_attrition),
        ("FIG-03", viz.plot_fig03_funnel_plot),
        ("FIG-04", viz.plot_fig04_leaver_concentration),
        ("FIG-05", viz.plot_fig05_age_sensitivity),
        ("FIG-06", viz.plot_fig06_tenure_sensitivity),
        ("FIG-07", viz.plot_fig07_promotion_gap_cohort),
        ("FIG-08", viz.plot_fig08_workload_mobility),
        ("FIG-09", viz.plot_fig09_satisfaction_small_multiples),
        ("FIG-10", viz.plot_fig10_overtime_interactions),
        ("FIG-11", viz.plot_fig11_hotspot_candidates),
        ("FIG-12", viz.plot_fig12_hotspot_overlap),
        ("FIG-13", viz.plot_fig13_numeric_distributions),
        ("FIG-14", viz.plot_fig14_demographics_background),
    ]

    for fid, func in funcs:
        fig, plotted_df = func(loaded_tables)
        hatched_count = 0
        for ax in fig.axes:
            for patch in ax.patches:
                if patch.get_hatch() is not None:
                    hatched_count += 1

        exp_low_n = int(plotted_df["low_n_flag"].sum()) if "low_n_flag" in plotted_df.columns else 0
        assert hatched_count == exp_low_n, f"{fid}: Hatched patches ({hatched_count}) != expected low_n rows ({exp_low_n})"
        plt.close(fig)


def test_fig14_plotted_colors_agree_with_findings_flags(loaded_tables):
    """Verify that bar colors in FIG-14 match findings classifications from findings.py."""
    from workforce_intel.findings import build_findings
    from workforce_intel.config import FIGURE_PALETTE
    import matplotlib.colors as mcolors

    fig, plotted_df = viz.plot_fig14_demographics_background(loaded_tables)
    findings_df = build_findings(loaded_tables)

    dims = ["Gender", "MaritalStatus", "Education", "EducationField", "HasPriorExperience", "PerformanceRating"]

    # Verify each panel
    for ax_idx, dim in enumerate(dims):
        ax = fig.axes[ax_idx]
        bars = ax.patches
        dim_findings = findings_df[findings_df["dimension"] == dim]

        for b in bars:
            fc = b.get_facecolor()
            hatch = b.get_hatch()

            # If hatched, must be low-n
            if hatch == "//":
                assert mcolors.to_hex(fc[:3]).lower() == FIGURE_PALETTE["low_n"].lower()
            elif mcolors.to_hex(fc[:3]).lower() == FIGURE_PALETTE["highlight"].lower():
                # Must be flagged above
                assert len(dim_findings[dim_findings["direction"] == "above baseline"]) > 0
            elif mcolors.to_hex(fc[:3]).lower() == FIGURE_PALETTE["secondary"].lower():
                # Must be flagged below
                assert len(dim_findings[dim_findings["direction"] == "below baseline"]) > 0
            else:
                # Must be neutral primary
                assert mcolors.to_hex(fc[:3]).lower() == FIGURE_PALETTE["primary"].lower()

    # Specifically verify EducationField Human Resources is hatched grey
    edf_ax = fig.axes[3]
    hatched_in_edf = [b for b in edf_ax.patches if b.get_hatch() == "//"]
    assert len(hatched_in_edf) == 1, "FIG-14 panel D (EducationField) must have exactly 1 hatched bar (Human Resources, n=27)."

    plt.close(fig)



def test_color_coded_figures_agree_with_findings_flags(loaded_tables):
    """Verify that bar colors in FIG-07, FIG-08, FIG-09, and FIG-14 agree with findings flags."""
    from workforce_intel.findings import build_findings
    from workforce_intel.config import FIGURE_PALETTE
    import matplotlib.colors as mcolors

    findings_df = build_findings(loaded_tables)

    # 1. FIG-07
    fig, plotted_df = viz.plot_fig07_promotion_gap_cohort(loaded_tables)
    pg_findings = findings_df[findings_df["dimension"] == "PromotionGapBand"]
    ax_a = fig.axes[0]
    for b in ax_a.patches:
        fc = mcolors.to_hex(b.get_facecolor()[:3]).lower()
        if fc == FIGURE_PALETTE["secondary"].lower():
            assert len(pg_findings[pg_findings["direction"] == "below baseline"]) > 0
        else:
            assert fc == FIGURE_PALETTE["primary"].lower()

    pg_c_findings = findings_df[findings_df["dimension"] == "PromotionGapBand_Cohort"]
    ax_b = fig.axes[1]
    for b in ax_b.patches:
        fc = mcolors.to_hex(b.get_facecolor()[:3]).lower()
        if fc == FIGURE_PALETTE["highlight"].lower():
            assert len(pg_c_findings[pg_c_findings["direction"] == "above baseline"]) > 0
        else:
            assert fc == FIGURE_PALETTE["primary"].lower()
    plt.close(fig)

    # 2. FIG-08: OverTime, BusinessTravel, DistanceBand
    fig, plotted_df = viz.plot_fig08_workload_mobility(loaded_tables)
    for ax_idx, dim in enumerate(["OverTime", "BusinessTravel", "DistanceBand"]):
        ax = fig.axes[ax_idx]
        dim_findings = findings_df[findings_df["dimension"] == dim]
        for b in ax.patches:
            fc = mcolors.to_hex(b.get_facecolor()[:3]).lower()
            hatch = b.get_hatch()
            if hatch == "//":
                assert fc == FIGURE_PALETTE["low_n"].lower()
            elif fc == FIGURE_PALETTE["highlight"].lower():
                assert len(dim_findings[dim_findings["direction"] == "above baseline"]) > 0
            elif fc == FIGURE_PALETTE["secondary"].lower():
                assert len(dim_findings[dim_findings["direction"] == "below baseline"]) > 0
            else:
                assert fc == FIGURE_PALETTE["primary"].lower()
    plt.close(fig)

    # 3. FIG-09: Satisfaction Small Multiples
    fig, plotted_df = viz.plot_fig09_satisfaction_small_multiples(loaded_tables)
    sat_dims = ["WorkLifeBalance", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "JobInvolvement"]
    for ax_idx, dim in enumerate(sat_dims):
        ax = fig.axes[ax_idx]
        dim_findings = findings_df[findings_df["dimension"] == dim]
        for b in ax.patches:
            fc = mcolors.to_hex(b.get_facecolor()[:3]).lower()
            hatch = b.get_hatch()
            if hatch == "//":
                assert fc == FIGURE_PALETTE["low_n"].lower()
            elif fc == FIGURE_PALETTE["highlight"].lower():
                assert len(dim_findings[dim_findings["direction"] == "above baseline"]) > 0
            elif fc == FIGURE_PALETTE["secondary"].lower():
                assert len(dim_findings[dim_findings["direction"] == "below baseline"]) > 0
            else:
                assert fc == FIGURE_PALETTE["primary"].lower()
    plt.close(fig)


def test_figures_with_whiskers_in_subtitle_have_errorbars(loaded_tables):
    """Verify that any figure whose title or subtitle mentions whiskers draws errorbar artists."""
    from matplotlib.container import ErrorbarContainer

    fig_funcs = [
        ("FIG-01", viz.plot_fig01_department_overview),
        ("FIG-02", viz.plot_fig02_jobrole_attrition),
        ("FIG-03", viz.plot_fig03_funnel_plot),
        ("FIG-04", viz.plot_fig04_leaver_concentration),
        ("FIG-05", viz.plot_fig05_age_sensitivity),
        ("FIG-06", viz.plot_fig06_tenure_sensitivity),
        ("FIG-07", viz.plot_fig07_promotion_gap_cohort),
        ("FIG-08", viz.plot_fig08_workload_mobility),
        ("FIG-09", viz.plot_fig09_satisfaction_small_multiples),
        ("FIG-10", viz.plot_fig10_overtime_interactions),
        ("FIG-11", viz.plot_fig11_hotspot_candidates),
        ("FIG-12", viz.plot_fig12_hotspot_overlap),
        ("FIG-13", viz.plot_fig13_numeric_distributions),
        ("FIG-14", viz.plot_fig14_demographics_background),
    ]

    for fid, func in fig_funcs:
        fig, plotted_df = func(loaded_tables)
        all_text = ""
        if fig._suptitle:
            all_text += fig._suptitle.get_text() + " "
        for ax in fig.axes:
            all_text += ax.get_title() + " "

        mentions_whiskers = "whisker" in all_text.lower()
        if mentions_whiskers:
            has_errorbars = False
            for ax in fig.axes:
                for c in ax.containers:
                    if isinstance(c, ErrorbarContainer):
                        has_errorbars = True
                        break
                if has_errorbars:
                    break
            assert has_errorbars, f"{fid} subtitle mentions whiskers but draws no ErrorbarContainer artists."
        plt.close(fig)
