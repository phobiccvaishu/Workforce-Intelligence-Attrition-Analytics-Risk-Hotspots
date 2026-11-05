"""Layout and typography geometry tests for report figures.

Validates that all 14 figures satisfy presentation invariants:
(a) every text bounding box is within the figure boundary;
(b) no two text bounding boxes overlap;
(c) legend bounding boxes overlap no text artists;
(d) rate/sample-size annotations intersect neither errorbar segments nor baseline lines.
Also validates that every baseline value stated in a colour key equals an axhline/axvline value drawn in that figure.
"""

import re
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
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
    # Allow 1.5px tolerance for border rounding
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
    for ax_idx, ax in enumerate(fig.axes):
        # Baseline dashed lines
        baseline_lines = [l for l in ax.lines if l.get_linestyle() == "--"]

        # Collect errorbar segments in display coordinates
        err_segments: List[np.ndarray] = []
        for c in ax.containers:
            eb = getattr(c, "errorbar", None)
            target = eb if eb is not None else (c if hasattr(c, "lines") and len(getattr(c, "lines", [])) > 1 else None)
            if target is not None:
                for line in target.lines[1:]:
                    if hasattr(line, "get_segments"):
                        for seg in line.get_segments():
                            err_segments.append(ax.transData.transform(seg))
                    elif hasattr(line, "get_data"):
                        d = line.get_data()
                        if len(d[0]) == 2:
                            pts = np.column_stack(d)
                            err_segments.append(ax.transData.transform(pts))

        for at in ax.texts:
            if at.get_visible() and ("%" in at.get_text() or "n=" in at.get_text() or "leavers=" in at.get_text() or "l=" in at.get_text()):
                tb = at.get_window_extent(renderer)
                txt_lbl = at.get_text().strip().replace("\n", " ")[:20]

                # Intersect errorbar segments
                for seg in err_segments:
                    if line_intersects_box(seg[0], seg[1], tb):
                        violations.append(f"{fid} [WHISKER_INTERSECTION]: ax[{ax_idx}] annotation '{txt_lbl}' intersects errorbar whisker segment")
                        break

                # Intersect baseline lines
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
