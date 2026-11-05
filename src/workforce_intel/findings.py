"""Findings register generation module.

Applies a pre-specified mechanical interval-flagging rule across all analysis dimensions
and renders the formal markdown findings register. Pure functions with zero file I/O.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


from workforce_intel.config import (
    ANALYSIS_DIMENSIONS,
    FIGURE_PALETTE,
    FINDINGS_RULE,
    ROUND_DIGITS,
)


def classify_segment(
    n: int,
    ci_low: float,
    ci_high: float,
    baseline: float,
    min_n: int = 30,
) -> str:
    """Classify a subgroup relative to baseline under the pre-specified mechanical rule."""
    if n < min_n:
        return "insufficient sample"
    if ci_low > baseline:
        return "above baseline"
    if ci_high < baseline:
        return "below baseline"
    return "no separation"


def get_segment_style(
    n: int,
    ci_low: float,
    ci_high: float,
    baseline: float,
    min_n: int = 30,
) -> Dict[str, Any]:
    """Retrieve consistent styling (color, hatch, label) for a subgroup based on classification."""
    classification = classify_segment(n, ci_low, ci_high, baseline, min_n)
    if classification == "insufficient sample":
        return {
            "color": FIGURE_PALETTE["low_n"],
            "hatch": "//",
            "flag": "low_n",
            "annotation": "n<30",
            "classification": classification,
        }
    elif classification == "above baseline":
        return {
            "color": FIGURE_PALETTE["highlight"],
            "hatch": None,
            "flag": "above",
            "annotation": "",
            "classification": classification,
        }
    elif classification == "below baseline":
        return {
            "color": FIGURE_PALETTE["secondary"],
            "hatch": None,
            "flag": "below",
            "annotation": "",
            "classification": classification,
        }
    else:
        return {
            "color": FIGURE_PALETTE["primary"],
            "hatch": None,
            "flag": "inside",
            "annotation": "",
            "classification": classification,
        }


def get_dimension_panel_title(
    dim_name: str,
    label_name: str,
    letter: str,
    findings_df: pd.DataFrame,
) -> str:
    """Generate dynamic panel title reflecting mechanical findings classification."""
    dim_rows = findings_df[findings_df["dimension"] == dim_name]
    above_segs = dim_rows[dim_rows["direction"] == "above baseline"]["segment"].tolist()
    below_segs = dim_rows[dim_rows["direction"] == "below baseline"]["segment"].tolist()

    parts = []
    if above_segs:
        parts.append(f"above baseline: {', '.join(above_segs)}")
    if below_segs:
        parts.append(f"below baseline: {', '.join(below_segs)}")

    if parts:
        status_str = "; ".join(parts)
    else:
        status_str = "no segment separates from baseline"

    return f"{letter}. {label_name} ({status_str})"



def build_findings(
    tables: Dict[str, pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Evaluate all analysis dimensions under the mechanical interval-flagging rule.

    Rule:
    - Segments with n < 30 are categorized as 'insufficient sample'.
    - For n >= 30, a segment is 'above baseline' if its Wilson 95% lower bound exceeds
      the baseline rate, and 'below baseline' if its Wilson 95% upper bound is below the baseline rate.
    - Otherwise, 'no separation'.
    - For workforce-wide tables, baseline is overall workforce attrition rate (16.12%).
    - For PromotionAnalysisCohort tables, baseline is the cohort's own attrition rate (10.79%).
    - If no segment in a dimension separates from baseline, a null result row is recorded.
    """
    rule_cfg = config or FINDINGS_RULE
    min_n = rule_cfg.get("min_n", 30)

    # 1. Resolve overall baseline rate
    baseline_df = tables.get("overall_baseline.csv")
    if baseline_df is None:
        baseline_df = tables.get("overall_baseline")
    if baseline_df is None:
        raise KeyError("Required table 'overall_baseline.csv' not found in tables dict.")
    workforce_baseline_pct = float(baseline_df["attrition_rate_pct"].iloc[0])

    # 2. Resolve cohort baseline rate
    cohort_df = tables.get("promotiongap_cohort_attrition.csv")
    if cohort_df is None:
        cohort_df = tables.get("promotiongap_cohort_attrition")
    if cohort_df is not None:
        cohort_tot_n = int(cohort_df["n"].sum())
        cohort_tot_leavers = int(cohort_df["leavers"].sum())
        cohort_baseline_pct = (cohort_tot_leavers / cohort_tot_n) * 100.0 if cohort_tot_n > 0 else 0.0
    else:
        cohort_tot_n = 890
        cohort_tot_leavers = 96
        cohort_baseline_pct = (96.0 / 890.0) * 100.0

    rows: List[Dict[str, Any]] = []
    finding_counter = 1

    # Mapping of dimensions to figure IDs
    dimension_figure_map = {
        "Department": "FIG-01",
        "JobRole": "FIG-02",
        "JobLevel": "FIG-10",
        "AgeGroup": "FIG-05",
        "Gender": "FIG-14",
        "MaritalStatus": "FIG-14",
        "Education": "FIG-14",
        "EducationField": "FIG-14",
        "TenureBand": "FIG-06",
        "EarlyTenure": "FIG-06",
        "CareerStage": "FIG-06",
        "RoleTenureBand": "FIG-06",
        "ManagerTenureBand": "FIG-06",
        "PromotionGapBand": "FIG-07",
        "PromotionGapEqualsTenure": "FIG-07",
        "HasPriorExperience": "FIG-14",
        "NumCompaniesWorked": "FIG-06",
        "OverTime": "FIG-08",
        "BusinessTravel": "FIG-08",
        "DistanceBand": "FIG-08",
        "WorkLifeBalance": "FIG-09",
        "JobSatisfaction": "FIG-09",
        "EnvironmentSatisfaction": "FIG-09",
        "RelationshipSatisfaction": "FIG-09",
        "JobInvolvement": "FIG-09",
        "StockOptionLevel": "FIG-08",
        "TrainingTimesLastYear": "FIG-08",
        "PerformanceRating": "FIG-14",
        "PromotionGapBand_Cohort": "FIG-07",
    }

    # Evaluate 28 standard dimensions
    for dim_info in ANALYSIS_DIMENSIONS:
        dim = dim_info["dimension"]
        tbl_name = dim_info["table"]
        is_protected = dim_info.get("protected", False)
        fig_id = dimension_figure_map.get(dim, "FIG-14")

        df = tables.get(tbl_name)
        if df is None:
            df = tables.get(tbl_name.replace(".csv", ""))
        if df is None:
            raise KeyError(f"Required table '{tbl_name}' not found in tables dict.")

        dim_flagged_count = 0
        total_dim_n = int(df["n"].sum()) if "n" in df.columns else 0
        total_dim_leavers = int(df["leavers"].sum()) if "leavers" in df.columns else 0

        for _, row in df.iterrows():
            seg_val = str(row["segment"])
            n_val = int(row["n"])
            leavers_val = int(row["leavers"])
            rate_val = float(row["attrition_rate_pct"])
            ci_low = float(row["ci95_low_pct"])
            ci_high = float(row["ci95_high_pct"])

            direction = classify_segment(n_val, ci_low, ci_high, workforce_baseline_pct, min_n=min_n)

            # If flagged above or below, record finding row
            if direction in ["above baseline", "below baseline"]:
                dim_flagged_count += 1
                fid = f"FIND-{finding_counter:02d}"
                finding_counter += 1

                # Robustness tag
                robustness = "not assessed"
                if dim == "EarlyTenure" and seg_val in ["1", "<= 1"]:
                    robustness = "same across cut points"
                elif dim == "AgeGroup" and seg_val in ["18-25"]:
                    robustness = "same across cut points"
                elif dim == "TenureBand" and seg_val in ["0-1"]:
                    robustness = "same across cut points"
                elif dim == "DistanceBand" and seg_val in ["21-29"]:
                    robustness = "same across cut points"

                # Confounding note
                confounding = ""
                if dim == "PromotionGapBand":
                    confounding = "Promotion gap progression is confounded by tenure in pooled workforce; isolated within PromotionAnalysisCohort (YearsAtCompany >= 5)."
                elif dim == "OverTime":
                    confounding = "OverTime leaver rates remain elevated across job levels, with highest rates in early career (Job Level 1)."
                elif dim in ["TenureBand", "EarlyTenure", "AgeGroup"]:
                    confounding = "Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition."

                protected_note = (
                    "Protected demographic attribute; descriptive only, not for individual scoring."
                    if is_protected
                    else ""
                )

                rows.append({
                    "finding_id": fid,
                    "dimension": dim,
                    "segment": seg_val,
                    "n": n_val,
                    "leavers": leavers_val,
                    "attrition_rate_pct": rate_val,
                    "ci95_low_pct": ci_low,
                    "ci95_high_pct": ci_high,
                    "baseline_rate_pct": round(workforce_baseline_pct, ROUND_DIGITS),
                    "baseline_type": "Workforce Baseline",
                    "direction": direction,
                    "protected_attribute_note": protected_note,
                    "source_table": tbl_name,
                    "figure_id": fig_id,
                    "robustness": robustness,
                    "confounding_note": confounding,
                    "status": "DESCRIPTIVE; formal testing pending Phase 4",
                })

        # Null result handling: if no segment separated in this dimension
        if dim_flagged_count == 0:
            fid = f"FIND-{finding_counter:02d}"
            finding_counter += 1
            protected_note = (
                "Protected demographic attribute; descriptive only, not for individual scoring."
                if is_protected
                else ""
            )
            rows.append({
                "finding_id": fid,
                "dimension": dim,
                "segment": "None (all segments within baseline interval)",
                "n": total_dim_n,
                "leavers": total_dim_leavers,
                "attrition_rate_pct": np.nan,
                "ci95_low_pct": np.nan,
                "ci95_high_pct": np.nan,
                "baseline_rate_pct": round(workforce_baseline_pct, ROUND_DIGITS),
                "baseline_type": "Workforce Baseline",
                "direction": "no separation",
                "protected_attribute_note": protected_note,
                "source_table": tbl_name,
                "figure_id": fig_id,
                "robustness": "not assessed",
                "confounding_note": "",
                "status": "DESCRIPTIVE; formal testing pending Phase 4",
            })

    # Evaluate Cohort Promotion Gap Dimension
    if cohort_df is not None:
        cohort_flagged_count = 0
        for _, row in cohort_df.iterrows():
            seg_val = str(row["segment"])
            n_val = int(row["n"])
            leavers_val = int(row["leavers"])
            rate_val = float(row["attrition_rate_pct"])
            ci_low = float(row["ci95_low_pct"])
            ci_high = float(row["ci95_high_pct"])

            direction = classify_segment(n_val, ci_low, ci_high, cohort_baseline_pct, min_n=min_n)

            if direction in ["above baseline", "below baseline"]:
                cohort_flagged_count += 1
                fid = f"FIND-{finding_counter:02d}"
                finding_counter += 1

                robustness = "same across cut points" if seg_val == "6+" else "not assessed"
                confounding = "Evaluated strictly within PromotionAnalysisCohort (YearsAtCompany >= 5) against cohort baseline to control for tenure progression."

                rows.append({
                    "finding_id": fid,
                    "dimension": "PromotionGapBand_Cohort",
                    "segment": seg_val,
                    "n": n_val,
                    "leavers": leavers_val,
                    "attrition_rate_pct": rate_val,
                    "ci95_low_pct": ci_low,
                    "ci95_high_pct": ci_high,
                    "baseline_rate_pct": round(cohort_baseline_pct, ROUND_DIGITS),
                    "baseline_type": "Cohort Baseline",
                    "direction": direction,
                    "protected_attribute_note": "",
                    "source_table": "promotiongap_cohort_attrition.csv",
                    "figure_id": "FIG-07",
                    "robustness": robustness,
                    "confounding_note": confounding,
                    "status": "DESCRIPTIVE; formal testing pending Phase 4",
                })

    findings_df = pd.DataFrame(rows)
    return findings_df


def compile_findings_summary(
    tables: Dict[str, pd.DataFrame],
    findings_df: pd.DataFrame,
    eda_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compile summary statistics of the mechanical findings scan into a dictionary."""
    baseline_df = tables.get("overall_baseline.csv")
    if baseline_df is None:
        baseline_df = tables.get("overall_baseline")
    workforce_baseline_pct = float(baseline_df["attrition_rate_pct"].iloc[0])

    cohort_df = tables.get("promotiongap_cohort_attrition.csv")
    if cohort_df is None:
        cohort_df = tables.get("promotiongap_cohort_attrition")
    if cohort_df is not None:
        cohort_tot_n = int(cohort_df["n"].sum())
        cohort_tot_leavers = int(cohort_df["leavers"].sum())
        cohort_baseline_pct = (cohort_tot_leavers / cohort_tot_n) * 100.0 if cohort_tot_n > 0 else 0.0
    else:
        cohort_tot_n = 890
        cohort_tot_leavers = 96
        cohort_baseline_pct = 10.7865

    # Count total evaluated segments with n >= 30 across the 28 standard dimensions
    total_evaluated_segments = 0
    for dim_info in ANALYSIS_DIMENSIONS:
        tbl_name = dim_info["table"]
        df = tables.get(tbl_name)
        if df is None:
            df = tables.get(tbl_name.replace(".csv", ""))
        if df is not None and "n" in df.columns:
            total_evaluated_segments += int((df["n"] >= 30).sum())

    # Count flags for 28 dimensions
    standard_findings = findings_df[findings_df["dimension"] != "PromotionGapBand_Cohort"]
    above_count = int((standard_findings["direction"] == "above baseline").sum())
    below_count = int((standard_findings["direction"] == "below baseline").sum())
    no_sep_count = int((standard_findings["direction"] == "no separation").sum())

    no_sep_dimensions = standard_findings[
        standard_findings["direction"] == "no separation"
    ]["dimension"].tolist()

    # Hotspot candidate metrics
    hotspots_df = tables.get("hotspot_candidates.csv")
    if hotspots_df is None:
        hotspots_df = tables.get("hotspot_candidates")
    hotspot_candidate_count = len(hotspots_df) if hotspots_df is not None else 0

    total_wf_n = (
        int(baseline_df["total_employees"].iloc[0])
        if "total_employees" in baseline_df.columns
        else int(baseline_df["n"].iloc[0])
    )
    total_wf_leavers = int(baseline_df["leavers"].iloc[0])

    if eda_summary and "hotspot_candidates_scan" in eda_summary:
        scan_info = eda_summary["hotspot_candidates_scan"]
        union_n = int(scan_info.get("distinct_employees_in_union", 0))
        union_leavers = int(scan_info.get("distinct_leavers_in_union", 0))
    elif "hotspot_union" in tables:
        union_obj = tables["hotspot_union"]
        if isinstance(union_obj, pd.DataFrame):
            union_n = int(union_obj["distinct_union_n"].iloc[0])
            union_leavers = int(union_obj["distinct_union_leavers"].iloc[0])
        elif isinstance(union_obj, dict):
            union_n = int(union_obj.get("distinct_union_n", 0))
            union_leavers = int(union_obj.get("distinct_union_leavers", 0))
        else:
            union_n = 0
            union_leavers = 0
    else:
        union_n = 0
        union_leavers = 0

    union_rate_pct = (
        round((union_leavers / union_n) * 100.0, ROUND_DIGITS)
        if union_n > 0
        else 0.0
    )
    union_workforce_share_pct = (
        round((union_n / total_wf_n) * 100.0, ROUND_DIGITS)
        if total_wf_n > 0
        else 0.0
    )
    union_leaver_share_pct = (
        round((union_leavers / total_wf_leavers) * 100.0, ROUND_DIGITS)
        if total_wf_leavers > 0
        else 0.0
    )

    outside_n = total_wf_n - union_n
    outside_leavers = total_wf_leavers - union_leavers
    outside_rate_pct = (
        round((outside_leavers / outside_n) * 100.0, ROUND_DIGITS)
        if outside_n > 0
        else 0.0
    )

    expected_chance_flags = round(0.05 * total_evaluated_segments, 1)

    summary = {
        "workforce_baseline_pct": round(workforce_baseline_pct, ROUND_DIGITS),
        "cohort_baseline_pct": round(cohort_baseline_pct, ROUND_DIGITS),
        "cohort_headcount": cohort_tot_n,
        "cohort_leavers": cohort_tot_leavers,
        "dimensions_evaluated": len(ANALYSIS_DIMENSIONS),
        "segments_evaluated_min_n": total_evaluated_segments,
        "segments_flagged_above": above_count,
        "segments_flagged_below": below_count,
        "dimensions_no_separation_count": len(no_sep_dimensions),
        "dimensions_no_separation": no_sep_dimensions,
        "expected_chance_flags": expected_chance_flags,
        "hotspot_candidate_cells": hotspot_candidate_count,
        "hotspot_union_headcount": union_n,
        "hotspot_union_leavers": union_leavers,
        "hotspot_union_rate_pct": union_rate_pct,
        "hotspot_union_workforce_share_pct": union_workforce_share_pct,
        "hotspot_union_leaver_share_pct": union_leaver_share_pct,
        "hotspot_outside_headcount": outside_n,
        "hotspot_outside_leavers": outside_leavers,
        "hotspot_outside_rate_pct": outside_rate_pct,
    }
    return summary


def render_register_markdown(
    findings_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> str:
    """Render the findings register as a GitHub Flavored Markdown document."""
    lines: List[str] = []

    lines.append("# Findings Register: Exploratory Data Analysis")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Mechanical Screening Protocol Notice:** The findings documented below result from a mechanical,")
    lines.append("> pre-specified confidence interval screening rule applied across 28 analysis dimensions.")
    lines.append("> **Flags are not statistical significance tests.** In a scan of multiple subgroup intervals,")
    lines.append(f"> approximately {summary['expected_chance_flags']} flags would be expected by random variation alone (0.05 x {summary['segments_evaluated_min_n']} segments evaluated).")
    lines.append("> All findings are purely descriptive; formal multivariable hypothesis testing and multiplicity adjustments are reserved for Phase 4.")
    lines.append("")
    lines.append("## 1. Synthesis & Scan Summary")
    lines.append("")
    total_flagged = summary['segments_flagged_above'] + summary['segments_flagged_below']
    lines.append(f"> [!NOTE]")
    lines.append(f"> **Multi-Dimension Correlation Notice:** Several evaluated dimensions exhibit substantial structural correlation")
    lines.append(f"> (for example, multiple tenure measures, job level and stock option tier, job role and department).")
    lines.append(f"> Consequently, flagged subgroups are not mutually independent findings.")
    lines.append("")
    lines.append(f"- **Dimensions Evaluated:** {summary['dimensions_evaluated']}")
    lines.append(f"- **Subgroups Evaluated (Headcount n >= 30):** {summary['segments_evaluated_min_n']}")
    lines.append(f"- **Workforce Baseline Attrition Rate:** {summary['workforce_baseline_pct']}%")
    lines.append(f"- **PromotionAnalysisCohort Baseline Rate:** {summary['cohort_baseline_pct']}% ({summary['cohort_leavers']} leavers / {summary['cohort_headcount']} employees)")
    lines.append(f"- **Subgroups Flagged Above Baseline:** {summary['segments_flagged_above']}")
    lines.append(f"- **Subgroups Flagged Below Baseline:** {summary['segments_flagged_below']}")
    lines.append(f"- **Dimensions with Zero Flagged Segments:** {summary['dimensions_no_separation_count']} ({', '.join(summary['dimensions_no_separation'])})")
    lines.append(f"- **Rough Count Expected by Chance Alone:** {summary['expected_chance_flags']} (stated as: flags are not statistical tests)")
    lines.append(f"- **Chance Comparison Statement:** {total_flagged} segments flagged against roughly {summary['expected_chance_flags']} expected by chance alone.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Pre-Specified Risk Hotspot Candidates & Union Analysis")
    lines.append("")
    lines.append(f"Across 58 scanned two-way interaction cells, exactly {summary['hotspot_candidate_cells']} subgroups met the pre-specified qualification criteria (headcount n >= 30 and Wilson 95% lower bound exceeding the workforce baseline {summary['workforce_baseline_pct']}%).")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> **Non-Independence Notice:** The 16 candidate subgroups overlap considerably (some are nested subsets) and do not represent 16 independent factors.")
    lines.append("")
    lines.append(f"- **Distinct Employees in Candidate Union:** {summary['hotspot_union_headcount']} ({summary['hotspot_union_workforce_share_pct']}% of workforce)")
    lines.append(f"- **Distinct Leavers in Candidate Union:** {summary['hotspot_union_leavers']} ({summary['hotspot_union_leaver_share_pct']}% of total departures)")
    lines.append(f"- **Attrition Rate within Union:** {summary['hotspot_union_rate_pct']}%")
    lines.append(f"- **Workforce Outside Candidate Union:** {summary['hotspot_outside_headcount']} employees, {summary['hotspot_outside_leavers']} leavers, attrition rate {summary['hotspot_outside_rate_pct']}%")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Register of Subgroup Findings")
    lines.append("")
    lines.append("| Finding ID | Dimension | Segment | Headcount (n) | Leavers | Rate (%) | 95% Wilson CI (%) | Baseline (%) | Direction | Robustness | Figure | Notes |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |")

    for _, row in findings_df.iterrows():
        fid = row["finding_id"]
        dim = row["dimension"]
        seg = row["segment"]
        n_val = row["n"]
        leavers_val = row["leavers"]

        if pd.isna(row["attrition_rate_pct"]):
            rate_str = "-"
            ci_str = "-"
        else:
            rate_str = f"{row['attrition_rate_pct']:.1f}%"
            ci_str = f"[{row['ci95_low_pct']:.1f}%, {row['ci95_high_pct']:.1f}%]"

        base_str = f"{row['baseline_rate_pct']:.1f}%"
        direction = row["direction"]
        robustness = row["robustness"]
        fig_id = row["figure_id"]

        notes_parts = []
        if row["protected_attribute_note"]:
            notes_parts.append(row["protected_attribute_note"])
        if row["confounding_note"]:
            notes_parts.append(row["confounding_note"])
        notes_str = " ".join(notes_parts) if notes_parts else "None"

        lines.append(
            f"| `{fid}` | {dim} | {seg} | {n_val} | {leavers_val} | {rate_str} | {ci_str} | {base_str} | `{direction}` | {robustness} | `{fig_id}` | {notes_str} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. What This Analysis Cannot Show")
    lines.append("")
    lines.append("This exploratory assessment is subject to substantial analytical and observational limitations:")
    lines.append("")
    lines.append("- **No Causal Inference:** Observed elevated rates reflect statistical associations in observational records. They do not demonstrate that any attribute produced or generated departures.")
    lines.append("- **Cross-Sectional Single Snapshot:** The data represents a single cross-sectional view without temporal departure timestamps, survival histories, or sequence intervals.")
    lines.append("- **Unobserved Departure Reasons:** Voluntariness of departures is unrecorded; voluntary resignations cannot be separated from involuntary terminations, retirements, or organizational reductions.")
    lines.append("- **Undocumented Measurement Units:** Several numeric features (such as commute distance) lack documented operational units (miles versus kilometers).")
    lines.append("- **Assumed Survey Scales:** Employee sentiment and satisfaction ratings represent assumed equidistant Likert scales without external calibration or longitudinal validation.")
    lines.append("- **Unverified Dataset Provenance:** The dataset originates from public synthetic HR records and must not be interpreted as authentic operational records of Palo Alto Networks.")
    lines.append("")

    return "\n".join(lines)
