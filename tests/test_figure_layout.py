"""Layout and typography geometry tests for report figures.

Validates that all 14 figures satisfy presentation invariants:
(a) every text bounding box is within the figure boundary;
(b) no two text bounding boxes overlap;
(c) legend bounding boxes overlap no text artists;
(d) rate/sample-size annotations intersect neither errorbar segments nor baseline lines;
(e) whisker endpoints are inside axes ylim/xlim (no clipping);
(f) legend bounding boxes overlap no errorbar whisker segments;
(g) text annotations do not leave their own axes bounds;
(h) light/white text is backed by a dark patch.
Also validates that every baseline value stated in a colour key equals an axhline/axvline value drawn in that figure.
"""

import re
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.transforms import Bbox

import workforce_intel.viz as viz


def line_intersects_box(p1: np.ndarray, p2: np.ndarray, box: Bbox) -> bool:
    """Determine if a 2D line segment between p1 and p2 intersects a 2D bounding box."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    bx0, by0, bx1, by1 = box.x0, box.y0, box.x1, box.y1

    # Segment bounding box check
    if max(x1, x2) < bx0 or min(x1, x2) > bx1 or max(y1, y2) < by0 or min(y1, y2) > by1:
        return False

    # Vertical line segment
    if abs(x2 - x1) < 1e-4:
        return (bx0 <= x1 <= bx1) and not (max(y1, y2) < by0 or min(y1, y2) > by1)

    # Horizontal line segment
    if abs(y2 - y1) < 1e-4:
        return (by0 <= y1 <= by1) and not (max(x1, x2) < bx0 or min(x1, x2) > bx1)

    # General line intersection with box edges
    for x_edge in (bx0, bx1):
        t = (x_edge - x1) / (x2 - x1)
        if 0.0 <= t <= 1.0:
            y_at_edge = y1 + t * (y2 - y1)
            if by0 <= y_at_edge <= by1:
                return True

    for y_edge in (by0, by1):
        t = (y_edge - y1) / (y2 - y1)
        if 0.0 <= t <= 1.0:
            x_at_edge = x1 + t * (x2 - x1)
            if bx0 <= x_at_edge <= bx1:
                return True

    # Also check if either endpoint is inside the box
    if (bx0 <= x1 <= bx1 and by0 <= y1 <= by1) or (bx0 <= x2 <= bx1 and by0 <= y2 <= by1):
        return True

    return False


def _luminance(color) -> float:
    """Return perceptual luminance of a colour (0=black, 1=white)."""
    try:
        rgba = mcolors.to_rgba(color)
        return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    except Exception:
        return 0.5


def collect_figure_layout_violations(fig: plt.Figure, fid: str) -> List[str]:
    """Inspect all visible text, legend, and line elements on figure for geometry violations."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w, fig_h = fig.get_size_inches() * fig.dpi
    violations = []

    # 1. Collect all visible text artists
    texts: List[Tuple[matplotlib.text.Text, str, str]] = []

    # Figure-level texts (suptitle, footnote, general annotations)
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            texts.append((t, "figure_text", t.get_text().strip().replace("\n", " ")[:35]))

    # Collect legends and internal legend texts
    legend_boxes: List[Tuple[Bbox, str]] = []
    legend_text_objs = set()

    for leg in fig.legends:
        if leg.get_visible():
            legend_boxes.append((leg.get_window_extent(renderer), "fig_legend"))
            for lt in leg.get_texts():
                legend_text_objs.add(lt)

    for ax_idx, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        if leg and leg.get_visible():
            legend_boxes.append((leg.get_window_extent(renderer), f"ax[{ax_idx}]_legend"))
            for lt in leg.get_texts():
                legend_text_objs.add(lt)

    # Axes-level texts
    for ax_idx, ax in enumerate(fig.axes):
        # Title
        if ax.title.get_visible() and ax.title.get_text().strip():
            texts.append((ax.title, f"ax[{ax_idx}]_title", ax.title.get_text().strip().replace("\n", " ")[:35]))

        # Axis labels
        if ax.xaxis.get_visible() and ax.xaxis.label.get_visible() and ax.xaxis.label.get_text().strip():
            texts.append((ax.xaxis.label, f"ax[{ax_idx}]_xlabel", ax.xaxis.label.get_text().strip().replace("\n", " ")[:25]))
        if ax.yaxis.get_visible() and ax.yaxis.label.get_visible() and ax.yaxis.label.get_text().strip():
            texts.append((ax.yaxis.label, f"ax[{ax_idx}]_ylabel", ax.yaxis.label.get_text().strip().replace("\n", " ")[:25]))

        # Tick labels (only for visible axes)
        if ax.xaxis.get_visible():
            for tl in ax.get_xticklabels():
                if tl.get_visible() and tl.get_text().strip():
                    texts.append((tl, f"ax[{ax_idx}]_xtick", tl.get_text().strip().replace("\n", " ")[:25]))
        if ax.yaxis.get_visible():
            for tl in ax.get_yticklabels():
                if tl.get_visible() and tl.get_text().strip():
                    texts.append((tl, f"ax[{ax_idx}]_ytick", tl.get_text().strip().replace("\n", " ")[:25]))

        # Inner annotations / text objects (exclude legend texts)
        for at in ax.texts:
            if at.get_visible() and at.get_text().strip() and at not in legend_text_objs:
                texts.append((at, f"ax[{ax_idx}]_annot", at.get_text().strip().replace("\n", " ")[:25]))

    # (a) Check every text bbox is inside figure bbox
    tol = 1.5
    for t_obj, kind, label in texts:
        bb = t_obj.get_window_extent(renderer)
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > fig_w + tol or bb.y1 > fig_h + tol:
            violations.append(f"{fid} [OUTSIDE_FIGURE]: {kind} '{label}' bbox=({bb.x0:.1f}, {bb.y0:.1f}, {bb.x1:.1f}, {bb.y1:.1f}) exceeds canvas ({fig_w:.1f}x{fig_h:.1f})")

    # (b) Check no two text bboxes overlap
    for i in range(len(texts)):
        t1, k1, l1 = texts[i]
        b1 = t1.get_window_extent(renderer)
        for j in range(i + 1, len(texts)):
            t2, k2, l2 = texts[j]
            b2 = t2.get_window_extent(renderer)
            if b1.overlaps(b2):
                ox = max(0.0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0))
                oy = max(0.0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))
                # Consider overlapping if intersection area exceeds 2.0 pixels^2
                if ox * oy > 2.0:
                    violations.append(f"{fid} [TEXT_OVERLAP]: '{l1}' ({k1}) overlaps '{l2}' ({k2}) area={ox*oy:.1f}px^2")

    # (c) Check legend bboxes overlap no text artists outside the legend
    for leg_bb, leg_name in legend_boxes:
        for t_obj, kind, label in texts:
            if t_obj not in legend_text_objs:
                tb = t_obj.get_window_extent(renderer)
                if leg_bb.overlaps(tb):
                    ox = max(0.0, min(leg_bb.x1, tb.x1) - max(leg_bb.x0, tb.x0))
                    oy = max(0.0, min(leg_bb.y1, tb.y1) - max(leg_bb.y0, tb.y0))
                    if ox * oy > 2.0:
                        violations.append(f"{fid} [LEGEND_OVERLAP]: {leg_name} overlaps {kind} '{label}' area={ox*oy:.1f}px^2")

    # (d) Rate/n annotations intersect no errorbar segment and no baseline line
    # (e) Whisker endpoints inside axes limits (not clipped)
    # (f) Legend does not overlap whisker segments
    # (g) Annotations do not leave their own axes bounds
    # (h) Light text is backed by a dark patch
    for ax_idx, ax in enumerate(fig.axes):
        ax_bb = ax.get_window_extent(renderer)
        baseline_lines = [l for l in ax.lines if l.get_linestyle() == "--"]

        # Collect errorbar segments in display coordinates and raw data coordinates
        err_segments_disp: List[np.ndarray] = []
        raw_segments_data: List[np.ndarray] = []

        for c in ax.containers:
            eb = getattr(c, "errorbar", None)
            if eb is None and isinstance(c, plt.matplotlib.container.ErrorbarContainer):
                eb = c
            if eb is not None:
                for item in eb.lines:
                    items = item if isinstance(item, tuple) else [item]
                    for sub in items:
                        if hasattr(sub, "get_segments"):
                            for seg in sub.get_segments():
                                raw_segments_data.append(seg)
                                err_segments_disp.append(ax.transData.transform(seg))
                        elif hasattr(sub, "get_data"):
                            d = sub.get_data()
                            if len(d) == 2 and len(d[0]) == 2:
                                pts = np.column_stack([d[0], d[1]])
                                raw_segments_data.append(pts)
                                err_segments_disp.append(ax.transData.transform(pts))

        # Check whisker clipped outside axes limits
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        for seg in raw_segments_data:
            for pt in seg:
                x_val, y_val = float(pt[0]), float(pt[1])
                if x_val < min(xlim) - 1e-3 or x_val > max(xlim) + 1e-3:
                    violations.append(f"{fid} [WHISKER_CLIPPED]: ax[{ax_idx}] whisker x={x_val:.2f} exceeds xlim [{min(xlim):.2f}, {max(xlim):.2f}]")
                if y_val < min(ylim) - 1e-3 or y_val > max(ylim) + 1e-3:
                    violations.append(f"{fid} [WHISKER_CLIPPED]: ax[{ax_idx}] whisker y={y_val:.2f} exceeds ylim [{min(ylim):.2f}, {max(ylim):.2f}]")

        # Check legend overlapping whiskers
        ax_leg = ax.get_legend()
        if ax_leg and ax_leg.get_visible():
            leg_bb = ax_leg.get_window_extent(renderer)
            for seg in err_segments_disp:
                if line_intersects_box(seg[0], seg[1], leg_bb):
                    violations.append(f"{fid} [LEGEND_WHISKER_OVERLAP]: ax[{ax_idx}] legend overlaps errorbar whisker segment")
                    break

        # Check annotations inside this axis
        for at in ax.texts:
            if not at.get_visible() or not at.get_text().strip() or at in legend_text_objs:
                continue
            tb = at.get_window_extent(renderer)
            txt_lbl = at.get_text().strip().replace("\n", " ")[:20]

            # Text leaving axes (skip under-axis text tables deliberately positioned below panel)
            is_under_axes_table = (at.get_transform() == ax.transAxes and at.get_position()[1] < 0)
            if not is_under_axes_table:
                if tb.x0 < ax_bb.x0 - 2.0 or tb.x1 > ax_bb.x1 + 2.0 or tb.y0 < ax_bb.y0 - 2.0 or tb.y1 > ax_bb.y1 + 2.0:
                    violations.append(f"{fid} [OUTSIDE_AXES]: ax[{ax_idx}] annotation '{txt_lbl}' bbox leaves axes bounds")

            # Light text on light background
            col = at.get_color()
            lum = _luminance(col)
            if lum > 0.8:
                cx = (tb.x0 + tb.x1) / 2
                cy = (tb.y0 + tb.y1) / 2
                covered = False
                if len(ax.images) > 0:
                    # Heatmap cell background provides contrast
                    covered = True
                for p in ax.patches:
                    p_bb = p.get_window_extent(renderer)
                    if p_bb.contains(cx, cy):
                        p_lum = _luminance(p.get_facecolor())
                        if p_lum < 0.5:
                            covered = True
                            break
                for coll in ax.collections:
                    offsets = coll.get_offsets()
                    if len(offsets) > 0:
                        disp_offsets = ax.transData.transform(offsets)
                        for pt in disp_offsets:
                            if np.hypot(pt[0] - cx, pt[1] - cy) < 14.0:
                                fcs = coll.get_facecolors()
                                if len(fcs) > 0 and _luminance(fcs[0]) < 0.5:
                                    covered = True
                                    break
                if not covered:
                    violations.append(f"{fid} [INVISIBLE_TEXT]: ax[{ax_idx}] light text '{txt_lbl}' (lum={lum:.2f}) not backed by dark patch")

            # Whisker intersection (whisker through label)
            for seg in err_segments_disp:
                if line_intersects_box(seg[0], seg[1], tb):
                    violations.append(f"{fid} [WHISKER_INTERSECTION]: ax[{ax_idx}] annotation '{txt_lbl}' intersects errorbar whisker segment")
                    break

            # Baseline intersection (only for rate/n annotations as specified)
            if "%" in at.get_text() or "n=" in at.get_text() or "leavers=" in at.get_text() or "l=" in at.get_text():
                for bl in baseline_lines:
                    d = bl.get_data()
                    pts = np.column_stack(d)
                    trans_pts = bl.get_transform().transform(pts)
                    if line_intersects_box(trans_pts[0], trans_pts[1], tb):
                        violations.append(f"{fid} [BASELINE_INTERSECTION]: ax[{ax_idx}] annotation '{txt_lbl}' intersects baseline line")
                        break

    return violations


@pytest.fixture
def loaded_tables():
    from scripts.run_phase3b_figures import load_phase3_tables
    from workforce_intel.config import PHASE3_TABLES_DIR
    return load_phase3_tables(PHASE3_TABLES_DIR)


def test_figure_layout_geometry(loaded_tables):
    """Assert all 14 figures meet layout geometry standards with zero overlap or clipping."""
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

    all_violations: List[str] = []

    for fid, func in fig_funcs:
        fig, plotted_df = func(loaded_tables)
        violations = collect_figure_layout_violations(fig, fid)
        if violations:
            all_violations.extend(violations)
        plt.close(fig)

    if all_violations:
        msg = f"Found {len(all_violations)} figure layout violations:\n" + "\n".join(all_violations)
        pytest.fail(msg)


def test_all_whiskers_inside_axes(loaded_tables):
    """Assert every whisker endpoint is inside the axes ylim/xlim."""
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

    clipping_violations = []
    for fid, func in fig_funcs:
        fig, _ = func(loaded_tables)
        for ax_idx, ax in enumerate(fig.axes):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for c in ax.containers:
                eb = getattr(c, "errorbar", None)
                if eb is None and isinstance(c, plt.matplotlib.container.ErrorbarContainer):
                    eb = c
                if eb is not None:
                    for item in eb.lines:
                        items = item if isinstance(item, tuple) else [item]
                        for sub in items:
                            if hasattr(sub, "get_segments"):
                                for seg in sub.get_segments():
                                    for pt in seg:
                                        if pt[0] < min(xlim) - 1e-3 or pt[0] > max(xlim) + 1e-3:
                                            clipping_violations.append(f"{fid} ax[{ax_idx}] whisker x={pt[0]:.2f} outside xlim {xlim}")
                                        if pt[1] < min(ylim) - 1e-3 or pt[1] > max(ylim) + 1e-3:
                                            clipping_violations.append(f"{fid} ax[{ax_idx}] whisker y={pt[1]:.2f} outside ylim {ylim}")
        plt.close(fig)

    assert not clipping_violations, "Found clipped whiskers:\n" + "\n".join(clipping_violations)


def test_baseline_values_in_colour_key_match_lines(loaded_tables):
    """Assert every baseline value stated in a figure's colour key equals an axhline or axvline value drawn in that figure."""
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
        fig, _ = func(loaded_tables)
        all_texts = [t.get_text() for t in fig.texts] + [ax.title.get_text() for ax in fig.axes]
        combined_text = "\n".join(all_texts)

        # Collect all axhline (y) and axvline (x) line values drawn in the figure
        drawn_lines = []
        for ax in fig.axes:
            for line in ax.lines:
                xdata, ydata = line.get_xdata(), line.get_ydata()
                if len(ydata) >= 2 and abs(ydata[0] - ydata[1]) < 1e-5:
                    drawn_lines.append(round(float(ydata[0]), 1))
                if len(xdata) >= 2 and abs(xdata[0] - xdata[1]) < 1e-5:
                    drawn_lines.append(round(float(xdata[0]), 1))

        if "dashed line:" in combined_text or "baseline" in combined_text.lower():
            key_matches = re.findall(r"baseline\s+(\d+\.?\d*)%", combined_text)
            for m in key_matches:
                val = round(float(m), 1)
                assert val in drawn_lines, (
                    f"{fid}: baseline value {val}% in key does not match any axhline/axvline in drawn lines {drawn_lines}"
                )
        plt.close(fig)


# =============================================================================
# MUTATION TESTS
# =============================================================================

def test_mutation_whisker_through_label(loaded_tables):
    """Mutation (a): in-bar label deliberately placed across a whisker."""
    fig, _ = viz.plot_fig01_department_overview(loaded_tables)
    ax = fig.axes[0]
    # Bar 0 is Sales, rate ~20.6%, CI ~17-24%. Place annotation at y=20 (right through whisker)
    ax.annotate("MUTATION_LABEL", xy=(0, 20.0), ha="center", va="center")
    violations = collect_figure_layout_violations(fig, "MUTATION-A")
    plt.close(fig)
    caught = [v for v in violations if "WHISKER_INTERSECTION" in v]
    assert len(caught) > 0, "Mutation (a) not caught!"
    print("\nMutation (a) caught:", caught[0])


def test_mutation_legend_over_whisker(loaded_tables):
    """Mutation (b): legend deliberately placed over a whisker."""
    fig, _ = viz.plot_fig01_department_overview(loaded_tables)
    ax = fig.axes[0]
    # Place legend directly over bar 0 whisker in data coordinates
    ax.legend(["Test"], loc="center", bbox_to_anchor=(0, 20), bbox_transform=ax.transData)
    violations = collect_figure_layout_violations(fig, "MUTATION-B")
    plt.close(fig)
    caught = [v for v in violations if "LEGEND_WHISKER_OVERLAP" in v]
    assert len(caught) > 0, "Mutation (b) not caught!"
    print("\nMutation (b) caught:", caught[0])


def test_mutation_text_outside_axes(loaded_tables):
    """Mutation (c): text deliberately placed outside axes."""
    fig, _ = viz.plot_fig01_department_overview(loaded_tables)
    ax = fig.axes[0]
    # Annotation at x=-2.0 is outside axes bounds
    ax.annotate("MUTATION_OUTSIDE", xy=(-2.0, 15.0), ha="center", va="center")
    violations = collect_figure_layout_violations(fig, "MUTATION-C")
    plt.close(fig)
    caught = [v for v in violations if "OUTSIDE_AXES" in v]
    assert len(caught) > 0, "Mutation (c) not caught!"
    print("\nMutation (c) caught:", caught[0])


def test_mutation_whisker_clipped(loaded_tables):
    """Mutation (d): whisker end outside axes ylim."""
    fig, _ = viz.plot_fig01_department_overview(loaded_tables)
    ax = fig.axes[0]
    # Set ylim to 10 (bar 0 whisker reaches >20)
    ax.set_ylim(0, 10)
    violations = collect_figure_layout_violations(fig, "MUTATION-D")
    plt.close(fig)
    caught = [v for v in violations if "WHISKER_CLIPPED" in v]
    assert len(caught) > 0, "Mutation (d) not caught!"
    print("\nMutation (d) caught:", caught[0])


def test_mutation_invisible_text(loaded_tables):
    """Mutation (e): light text with no dark bar patch behind it."""
    fig, _ = viz.plot_fig01_department_overview(loaded_tables)
    ax = fig.axes[0]
    # White text placed in empty area of axes (e.g. y=28, above all bars)
    ax.annotate("MUTATION_WHITE", xy=(1, 28.0), ha="center", va="center", color="white")
    violations = collect_figure_layout_violations(fig, "MUTATION-E")
    plt.close(fig)
    caught = [v for v in violations if "INVISIBLE_TEXT" in v]
    assert len(caught) > 0, "Mutation (e) not caught!"
    print("\nMutation (e) caught:", caught[0])
