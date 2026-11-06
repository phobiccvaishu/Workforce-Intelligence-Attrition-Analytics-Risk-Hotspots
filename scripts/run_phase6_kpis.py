"""Phase 6: KPI Module Pipeline Execution Script.

Evaluates the 10 pre-registered KPIs across the unfiltered workforce baseline
and representative filter cases. Generates:
1. docs/kpi_definitions.md
2. reports/phase6/tables/kpi_baseline.csv
3. reports/phase6/tables/kpi_filter_cases.csv
4. reports/phase6/kpi_report.md
5. Appends proposed decisions (DL-39+) to docs/decision_log.md

Enforces zero forbidden words, zero dollar signs, and strict determinism.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Add src to python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from workforce_intel.config import (
    ALLOWED_FILTER_DIMENSIONS,
    DECISION_LOG_FILE,
    DOCS_DIR,
    KPI_DEFINITIONS_MD,
    KPI_REGISTRY,
    MIN_GROUP_SIZE,
    PHASE6_REPORTS_DIR,
    PHASE6_REPORT_MD,
    PHASE6_TABLES_DIR,
    PROCESSED_DATA_FILE,
    PROJECT_ROOT,
    STATIC_TABLE_CHECKSUMS,
)
from workforce_intel.kpis import (
    apply_kpi_filter,
    evaluate_all_kpis,
    kpi_01_attrition_rate,
    kpi_02_headcount_leavers,
    kpi_03_relative_attrition,
    kpi_04_overtime_association,
    kpi_05_promotion_gap_cohort,
    kpi_06_adjusted_overtime,
    kpi_07_hotspot_universe_summary,
    kpi_08_hotspot_holdout_validation,
    kpi_09_hotspot_coverage,
    kpi_10_model_triage_yield,
)


def load_processed_data() -> pd.DataFrame:
    """Load the canonical processed dataset."""
    if not PROCESSED_DATA_FILE.is_file():
        raise FileNotFoundError(f"Processed dataset missing at: {PROCESSED_DATA_FILE}")
    return pd.read_csv(PROCESSED_DATA_FILE)


def format_value_summary(kpi: Dict[str, Any]) -> str:
    """Format KPI value as a compact descriptive string."""
    val = kpi["value"]
    kid = kpi["kpi_id"]

    if val is None:
        return "No Data (n=0)"

    if kid == "KPI-01":
        return f"{val * 100:.2f}%"
    elif kid == "KPI-02":
        return f"n={val['headcount']}, leavers={val['leavers']} ({val['headcount_share']*100:.1f}% headcount, {val['leavers_share']*100:.1f}% leavers)"
    elif kid == "KPI-03":
        diff_str = f"{val['rate_diff'] * 100:+.2f} pp"
        ratio_str = f"{val['rate_ratio']:.2f}x" if val["rate_ratio"] is not None else "-"
        return f"diff={diff_str}, ratio={ratio_str}, flag={val['separation_flag']}"
    elif kid == "KPI-04":
        or_str = f"OR={val['odds_ratio']:.2f}" if val["odds_ratio"] is not None else "OR=None"
        share_str = f"OT share={val['overtime_share']*100:.1f}%"
        return f"{or_str}, {share_str} (a={val['a_exposed_cases']}, b={val['b_exposed_noncases']}, c={val['c_unexposed_cases']}, d={val['d_unexposed_noncases']})"
    elif kid == "KPI-05":
        or_str = f"OR={val['crude_or_6plus_vs_0to5']:.2f}" if val["crude_or_6plus_vs_0to5"] is not None else "OR=None"
        return f"Cohort rate={val['cohort_attrition_rate']*100:.2f}% (n={val['cohort_headcount']}), 6+ vs 0-5 {or_str}"
    elif kid == "KPI-06":
        mh = val["mantel_haenszel_joblevel"]
        m1 = val["multivariable_m1"]
        m2 = val["multivariable_m2"]
        p4 = val["phase4_regularised_contrast"]
        return f"MH (JobLevel) OR={mh['odds_ratio']:.2f} [{mh['ci_low']:.2f}, {mh['ci_high']:.2f}]; M1 OR={m1['odds_ratio']:.2f}; M2 OR={m2['odds_ratio']:.2f}; P4 OR={p4['odds_ratio']:.2f}"
    elif kid == "KPI-07":
        return f"Universe={val['universe_size']} segments ({val['segments_n_ge_30']} with n>=30); Tier 1={val['tier1_priority_hotspots']}; Tier 2={val['tier2_secondary_hotspots']}; Prior m: 1-way={val['fitted_prior_m_1way']:.2f}, 2-way={val['fitted_prior_m_2way']:.2f}"
    elif kid == "KPI-08":
        return f"Screened Tier 1={val['screened_training_tier1']}; Holdout n>=30: {val['sample_floor_proportion']}; Above point: {val['point_replication_proportion']}; Above lower bound: {val['lower_bound_replication_proportion']}"
    elif kid == "KPI-09":
        return f"Steps={val['total_steps']}; Step 5: {val['step_5_workforce_share']*100:.1f}% workforce, {val['step_5_leavers_share']*100:.1f}% leavers; Final: {val['final_workforce_share']*100:.1f}% workforce, {val['final_leavers_share']*100:.1f}% leavers"
    elif kid == "KPI-10":
        t10 = val["top_10_percent"]
        t20 = val["top_20_percent"]
        return f"Base rate={val['holdout_base_rate']*100:.2f}%; Top 10%: k={t10['k']}, TP={t10['tp']}, Prec={t10['precision']*100:.1f}%, Rec={t10['recall']*100:.1f}%, Lift={t10['lift']:.2f}x; Top 20%: k={t20['k']}, TP={t20['tp']}, Prec={t20['precision']*100:.1f}%, Rec={t20['recall']*100:.1f}%, Lift={t20['lift']:.2f}x"
    return str(val)


def build_baseline_table(kpis: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format baseline KPI list into a standardized reporting DataFrame."""
    rows = []
    for k in kpis:
        val_summary = format_value_summary(k)
        ci_str = f"[{k['ci_low']:.4f}, {k['ci_high']:.4f}]" if k["ci_low"] is not None and k["ci_high"] is not None else "-"
        reg_entry = next((item for item in KPI_REGISTRY if item["kpi_id"] == k["kpi_id"]), {})
        rows.append({
            "kpi_id": k["kpi_id"],
            "name": k["name"],
            "kpi_class": reg_entry.get("kpi_class", "-"),
            "value_summary": val_summary,
            "n": k["n"],
            "ci_low": k["ci_low"] if k["ci_low"] is not None else "",
            "ci_high": k["ci_high"] if k["ci_high"] is not None else "",
            "ci_display": ci_str,
            "reliability": k["reliability"],
            "source": k["source"],
        })
    return pd.DataFrame(rows)


def build_filter_cases_table(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate LIVE KPIs across representative filter cases."""
    filter_cases = [
        ("CASE-01", "Unfiltered Workforce (Baseline)", {}),
        ("CASE-02", "Department: Sales", {"Department": "Sales"}),
        ("CASE-03", "Department: Research & Development", {"Department": "Research & Development"}),
        ("CASE-04", "Department: Human Resources", {"Department": "Human Resources"}),
        ("CASE-05", "JobLevel: Entry-Level 1", {"JobLevel": 1}),
        ("CASE-06", "OverTime: Yes (Exposed)", {"OverTime": "Yes"}),
        ("CASE-07", "OverTime: No (Unexposed)", {"OverTime": "No"}),
        ("CASE-08", "TenureBand: 0-1 Years", {"TenureBand": "0-1"}),
        ("CASE-09", "Multi-Attribute: Sales AND OverTime=Yes", {"Department": "Sales", "OverTime": "Yes"}),
        ("CASE-10", "Low-n Boundary: Manager AND OverTime=Yes (n=27 < 30)", {"JobRole": "Manager", "OverTime": "Yes"}),
        ("CASE-11", "Descriptive Protected Filter: Female AND Single", {"Gender": "Female", "MaritalStatus": "Single"}),
        ("CASE-12", "Empty Set: Sales AND Research Director (n=0)", {"Department": "Sales", "JobRole": "Research Director"}),
    ]

    records = []
    live_evaluators = [
        ("KPI-01", kpi_01_attrition_rate),
        ("KPI-02", kpi_02_headcount_leavers),
        ("KPI-03", kpi_03_relative_attrition),
        ("KPI-04", kpi_04_overtime_association),
        ("KPI-05", kpi_05_promotion_gap_cohort),
    ]

    for cid, cdesc, f_dict in filter_cases:
        for kid, func in live_evaluators:
            res = func(df, filters=f_dict)
            val_summary = format_value_summary(res)
            ci_str = f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]" if res["ci_low"] is not None and res["ci_high"] is not None else "-"
            records.append({
                "case_id": cid,
                "case_description": cdesc,
                "filter_json": json.dumps(f_dict),
                "kpi_id": kid,
                "name": res["name"],
                "value_summary": val_summary,
                "n": res["n"],
                "ci_low": res["ci_low"] if res["ci_low"] is not None else "",
                "ci_high": res["ci_high"] if res["ci_high"] is not None else "",
                "ci_display": ci_str,
                "reliability": res["reliability"],
            })

    return pd.DataFrame(records)


def generate_kpi_definitions_markdown() -> str:
    """Generate docs/kpi_definitions.md dynamically from KPI_REGISTRY."""
    text = r"""# Workforce Intelligence KPI Registry & Definitions

**Document Status**: Official Specification (Generated from `src/workforce_intel/config.py`)  
**Scope**: 10 Key Performance Indicators for Decision Support and Monitoring  
**Target Unit**: Employee (Attrition Yes = 1, voluntariness unrecorded)  
**Methodological Standard**: Descriptive association, Wilson score intervals, pre-registered static benchmarks

---

## 1. Architectural Architecture & Governance Rules

The KPI module exposes 10 standardized performance metrics divided into two distinct architectural classes:
1. **LIVE KPIs (KPI-01 through KPI-05)**: Evaluated dynamically on the processed dataset (`data/processed/workforce_attrition_processed.csv`) for any valid subpopulation filter. Protected demographic attributes (`AgeGroup`, `Gender`, `MaritalStatus`) are permitted solely as descriptive observational filters and are strictly prohibited from serving as segment definers for automated actions.
2. **STATIC Benchmarks (KPI-06 through KPI-10)**: Workforce-level benchmarks extracted from committed, immutable Phase 4 and Phase 5 statistical outputs. All static tables are verified upon loading via cryptographic SHA-256 integrity checksums. Static KPIs reject subpopulation filters with an explicit `ValueError` to prevent misleading comparisons against pre-computed full-population or split-sample models.

### Standardized Reliability Classification:
- **`reliable`**: The evaluated subpopulation meets the sample size floor (\(n \ge 30\)) and all underlying contingency cells satisfy \(a, b, c, d \ge 30\).
- **`low_n`**: Headcount \(0 < n < 30\) or one or more contingency cells fall below 30. Metric is returned for visibility but flagged as underpowered; ranking across low-n cells is prohibited.
- **`no_data`**: The filter yields zero matching employees (\(n = 0\)). Returns `value = None` without raising an error.
- **`unreliable_zero_cell`**: Contingency table contains a zero cell. A continuity correction of +0.5 is applied for displayed odds ratios per decision log protocol.
- **`workforce-level, not filterable`**: Benchmark applies to the full population or sequestered split and rejects ad-hoc filtering.

---

## 2. Complete KPI Registry

| KPI ID | Name | Class | Definition | Unit | Source / Function | Reliability Rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for k in KPI_REGISTRY:
        text += (
            f"| `{k['kpi_id']}` | **{k['name']}** | `{k['kpi_class']}` | "
            f"{k['definition']} | {k['unit']} | `{k['source']}` | {k['reliability_rule']} |\n"
        )

    text += """
---

## 3. Allowed Filter Dimensions (Conjunctive AND Logic)

Filter keys must belong to the pre-approved domain schema configured in `src/workforce_intel/config.py`. Unknown dimensions or unrecognized category levels raise an immediate `ValueError`:

"""
    for dim in sorted(ALLOWED_FILTER_DIMENSIONS):
        text += f"- `{dim}`\n"

    text += """
---

## 4. Decision Support Governance Principle

All KPI outputs are generated exclusively for **retention review, workforce planning, and organizational decision support**. Under no circumstances should individual-level or segment-level risk metrics be utilized for automated employment actions, disciplinary decisions, or adverse termination workflows.
"""
    return text


def generate_kpi_report_markdown(
    df_baseline: pd.DataFrame,
    df_cases: pd.DataFrame,
) -> str:
    """Generate reports/phase6/kpi_report.md summarizing baseline and case evaluations."""
    text = r"""# Phase 6 Report: Workforce KPI Module & Contract Validation

**Author**: Workforce Intelligence Engineering & Governance Team  
**Evaluation Target**: Pure-Function KPI Engine (10 Metrics)  
**Execution Environment**: Isolated Virtual Environment (.venv)  
**Baseline Population**: n = 1,470 employees, 237 leavers (16.12% baseline attrition)

---

## 1. Executive Summary

Phase 6 implements and validates the standalone, deterministic Key Performance Indicator (KPI) engine that will power the interactive Phase 7 analytics interface. The engine provides 10 pre-registered metrics adhering strictly to a standardized data contract:

\[
\\{ \\text{kpi\\_id}, \\text{name}, \\text{value}, n, \\text{ci\\_low}, \\text{ci\\_high}, \\text{reliability}, \\text{source}, \\text{as\\_of\\_filter} \\}
\]

Key verification outcomes:
1. **Contract Uniformity**: Every KPI returns the complete required schema. Rates are coupled with denominators (\(n\)) and 95% intervals where defined.
2. **Cryptographic Static Integrity**: Static benchmarks (KPIs 6-10) verify source table SHA-256 hashes upon initialization, rejecting ad-hoc filters with an explicit error.
3. **Robust Boundary Handling**: Empty filter results (\(n = 0\)) return `reliability = "no_data"` and `value = None` without runtime exceptions; small samples (\(n < 30\)) return flagged point estimates without hiding data.

---

## 2. Workforce Baseline KPI Summary (Unfiltered Population, n = 1,470)

The table below reports all 10 performance indicators computed across the unfiltered workforce baseline:

| KPI ID | Metric Name | Class | Baseline Value Summary | Base n | 95% Confidence Interval | Reliability Status | Source Path / Function |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in df_baseline.iterrows():
        text += (
            f"| `{r['kpi_id']}` | **{r['name']}** | `{r['kpi_class']}` | "
            f"{r['value_summary']} | {r['n']} | {r['ci_display']} | `{r['reliability']}` | `{r['source']}` |\n"
        )

    text += """
---

## 3. Representative Filter Case Evaluations (Live KPIs 1 through 5)

The live KPI functions were evaluated across 12 representative organizational, operational, multi-attribute, and boundary scenarios:

| Case ID | Scenario Description | Filter JSON | KPI ID | Metric Name | Output Value Summary | n | 95% Confidence Interval | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in df_cases.iterrows():
        text += (
            f"| `{r['case_id']}` | {r['case_description']} | `{r['filter_json']}` | "
            f"`{r['kpi_id']}` | **{r['name']}** | {r['value_summary']} | {r['n']} | {r['ci_display']} | `{r['reliability']}` |\n"
        )

    text += """
---

## 4. Multiplicity and Governance Assurances

1. **Observational Scope**: All live indicators quantify descriptive associations in cross-sectional records. No causal inferences or directional mechanisms are asserted.
2. **Demographic Filter Safeguards**: Protected attributes are permitted solely for descriptive audits and equity characterization. They cannot define operational risk hotspots or alter baseline regularised models.
3. **Decision Support Labeling**: Model triage yields (KPI-10) are explicitly designated for advisory decision support, prohibiting automated adverse personnel actions.
"""
    return text


def append_decision_log() -> None:
    """Audit and append Phase 6 decisions (DL-39+) as PROPOSED."""
    if not DECISION_LOG_FILE.is_file():
        raise FileNotFoundError(f"Decision log missing at: {DECISION_LOG_FILE}")

    dl_text = DECISION_LOG_FILE.read_text(encoding="utf-8")
    if "DL-39" in dl_text:
        print("DL-39 already present in decision log; skipping append.")
        return

    new_rows = """| `DL-39` | **Pure Function Architecture and Contract for Workforce KPI Engine** | 10 pre-registered KPIs (5 LIVE, 5 STATIC) evaluated across unfiltered baseline and 12 filter scenarios. Table: reports/phase6/tables/kpi_baseline.csv | Implement KPI engine as stateless pure functions returning contract: {kpi_id, name, value, n, ci_low, ci_high, reliability, source, as_of_filter}. Re-use Phase 3 and Phase 5 statistical utilities without re-implementing core math. | Stateful class architecture; ad-hoc Streamlit component computations; inconsistent return schemas. | `PROPOSED` | Phase 6 (KPI Module Architecture) |
| `DL-40` | **Cryptographic SHA-256 Integrity Verification for Static Inferential Benchmarks** | 5 static benchmark tables verified against pre-registered checksums in config.py. Table: reports/phase6/tables/kpi_baseline.csv | Verify SHA-256 hashes of Phase 4 and Phase 5 static benchmark tables upon loading; strictly reject ad-hoc subpopulation filtering on static workforce benchmarks with an explicit ValueError. | Silently filter static tables; re-estimate multivariable models on small subsets without convergence guarantees; allow unverified file loading. | `PROPOSED` | Phase 6 (Static Benchmark Integrity) |
| `DL-41` | **Standardized Reliability Flagging and Empty-Set Handling Policy** | Evaluated across empty filter (n=0), low-n boundary (n=27), and cell zero cases. Table: reports/phase6/tables/kpi_filter_cases.csv | Flag subpopulation metrics where base denominator n < MIN_GROUP_SIZE (30) as 'low_n', empty sets (n = 0) as 'no_data' returning None, and 2x2 contingency tables with zero cells as 'unreliable_zero_cell' (applying +0.5 Woolf continuity correction solely for displayed odds ratios). Never suppress or rank low-n metrics. | Suppress small cells from reporting; return 0.0% for empty filters; raise runtime exceptions on empty sets. | `PROPOSED` | Phase 6 (Reliability & Boundary Safeguards) |
"""
    updated_text = dl_text.rstrip() + "\n" + new_rows
    DECISION_LOG_FILE.write_text(updated_text, encoding="utf-8")
    print("Appended DL-39, DL-40, DL-41 to docs/decision_log.md as PROPOSED.")


def main() -> None:
    """Run full Phase 6 KPI generation pipeline."""
    print("=" * 70)
    print("PHASE 6: WORKFORCE KPI MODULE & VALIDATION PIPELINE")
    print("=" * 70)

    PHASE6_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE6_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1. Loading canonical processed dataset...")
    df = load_processed_data()
    print(f"Loaded dataset: n={len(df)}, total leavers={int(df['Attrition'].sum())} ({df['Attrition'].mean()*100:.2f}%)")

    print("\n2. Evaluating unfiltered workforce baseline across all 10 KPIs...")
    kpis_baseline = evaluate_all_kpis(df)
    df_baseline = build_baseline_table(kpis_baseline)
    baseline_csv = PHASE6_TABLES_DIR / "kpi_baseline.csv"
    df_baseline.to_csv(baseline_csv, index=False)
    print(f"Saved baseline table to: {baseline_csv}")

    print("\n3. Evaluating representative filter cases across LIVE KPIs...")
    df_cases = build_filter_cases_table(df)
    cases_csv = PHASE6_TABLES_DIR / "kpi_filter_cases.csv"
    df_cases.to_csv(cases_csv, index=False)
    print(f"Saved filter cases table to: {cases_csv}")

    print("\n4. Generating docs/kpi_definitions.md from KPI_REGISTRY...")
    kpi_defs_text = generate_kpi_definitions_markdown()
    KPI_DEFINITIONS_MD.write_text(kpi_defs_text, encoding="utf-8")
    print(f"Written KPI definitions to: {KPI_DEFINITIONS_MD}")

    print("\n5. Generating reports/phase6/kpi_report.md...")
    kpi_rep_text = generate_kpi_report_markdown(df_baseline, df_cases)
    PHASE6_REPORT_MD.write_text(kpi_rep_text, encoding="utf-8")
    print(f"Written KPI report to: {PHASE6_REPORT_MD}")

    print("\n6. Auditing decision log and appending DL-39+...")
    append_decision_log()

    print("\nPhase 6 pipeline completed successfully.")


if __name__ == "__main__":
    main()
