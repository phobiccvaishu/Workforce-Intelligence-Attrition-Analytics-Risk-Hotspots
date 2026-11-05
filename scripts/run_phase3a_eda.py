"""Execution pipeline for Phase 3A: Descriptive EDA Engine.

Produces all pre-specified rate tables, exact binomial funnel limits, concentration tables,
cross-tabulations, sensitivity checks, numerical distributions, hotspot candidate scanning,
analysis plan documentation, and comprehensive markdown report.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

from workforce_intel.config import (
    ALL_NUMERIC_COLUMNS,
    ANALYSIS_PLAN_MD,
    ANALYSIS_REGISTRY,
    CONTINUOUS_COLUMNS,
    FUNNEL_LEVELS,
    HOTSPOT_MIN_N,
    HOTSPOT_TOP_K,
    MIN_GROUP_SIZE,
    PHASE3_LOG_FILE,
    PHASE3_REPORT_MD,
    PHASE3_REPORTS_DIR,
    PHASE3_SUMMARY_JSON,
    PHASE3_TABLES_DIR,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
    ROUND_DIGITS,
    SENSITIVITY_SPECS,
    TARGET_COLUMN,
    TWO_WAY_SPECS,
    get_environment_metadata,
)
from workforce_intel.eda import (
    concentration_table,
    funnel_table,
    hotspot_candidates,
    monthly_income_by_joblevel,
    numeric_summary_by_outcome,
    overall_summary,
    rate_table,
    sensitivity_table,
    two_way_table,
)
from workforce_intel.io import load_processed
from workforce_intel.reporting import build_analysis_plan, build_phase3a_report


def setup_logging() -> logging.Logger:
    """Configure console and file logging."""
    PHASE3_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("phase3a_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(PHASE3_LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def compute_sha256(filepath: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def update_decision_log(logger: logging.Logger) -> None:
    """Append proposed decisions DL-27 to DL-31 to docs/decision_log.md if not already present."""
    decision_log_path = Path("docs/decision_log.md")
    if not decision_log_path.exists():
        logger.warning("docs/decision_log.md not found. Skipping decision log update.")
        return

    content = decision_log_path.read_text(encoding="utf-8")
    if "DL-27" in content:
        logger.info("Decisions DL-27+ already present in docs/decision_log.md.")
        return

    new_rows = """| `DL-27` | **Descriptive Focus & Absence of Causal Inference in Phase 3A** | 1,470 total records evaluated across 9 pre-specified analysis stages. Table: reports/phase3/tables/overall_baseline.csv | Strictly report observed rates, counts, and statistical intervals without directional claims. | Assert predictive factor statuses; perform unadjusted regression ranking. | `PROPOSED` | Phase 3 (Exploratory Data Analysis) |
| `DL-28` | **Wilson Score Confidence Intervals for Subgroup Rates** | Overall baseline 16.12% has 95% Wilson CI [14.33%, 18.09%]. Table: reports/phase3/tables/overall_baseline.csv | Report 95% Wilson score confidence intervals for all subgroup rates to handle near-boundary and small-sample estimates accurately. | Normal approximation (Wald) intervals; unadjusted point estimates. | `PROPOSED` | Phase 3 (Statistical Methodology) |
| `DL-29` | **Exact Binomial Funnel Limits for Organizational Outliers** | 12 organizational subgroups evaluated against baseline 16.12% (Department: 3, JobRole: 9). Table: reports/phase3/tables/funnel_limits.csv | Benchmark Department and JobRole attrition rates using exact binomial control limits at 95% and 99.8% to distinguish sample-size variation from outlying rates. | Arbitrary percentage cutoffs; standard deviation intervals assuming normal distribution. | `PROPOSED` | Phase 3 (Organizational Analytics) |
| `DL-30` | **Hotspot Candidate Screening Protocol** | 15 candidate subgroups selected from 58 scanned two-way interaction cells with n >= 30. Table: reports/phase3/tables/hotspot_candidates.csv | Require interaction cells to have n >= 30 and Wilson lower bound exceeding overall baseline (16.12%) to qualify as candidates for Phase 4 testing. | Rank all raw interaction rates without sample size floor or confidence interval threshold. | `PROPOSED` | Phase 3 / Phase 4 (Hotspot Protocol) |
| `DL-31` | **Dual Reporting of Promotion Gap (Pooled vs PromotionAnalysisCohort)** | 890 cohort records compared with 1,470 pooled records across 4 promotion gap bands. Table: reports/phase3/tables/promotiongap_side_by_side.csv | Report PromotionGapBand both pooled across all 1,470 employees and restricted to the eligible cohort (n = 890, YearsAtCompany >= 5) to isolate career stagnation. | Report pooled rates only; disregard tenure threshold. | `PROPOSED` | Phase 3 (Career & Mobility Analytics) |
"""
    # Insert before the last lines or append
    content = content.rstrip() + "\n" + new_rows
    decision_log_path.write_text(content, encoding="utf-8")
    logger.info("Appended proposed decisions DL-27 to DL-31 to docs/decision_log.md.")


def main() -> None:
    """Run the complete Phase 3A exploratory data analysis pipeline."""
    logger = setup_logging()
    logger.info("Starting Phase 3A Exploratory Data Analysis Engine...")

    # 1. Load Processed Dataset
    if not PROCESSED_DATA_FILE.exists():
        logger.error(f"Processed dataset not found at {PROCESSED_DATA_FILE}")
        sys.exit(1)

    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)
    processed_sha256 = compute_sha256(PROCESSED_DATA_FILE)
    logger.info(
        f"Loaded processed dataset ({len(df):,} rows x {len(df.columns)} columns). SHA-256: {processed_sha256}"
    )

    tables_dict: Dict[str, pd.DataFrame] = {}

    # 2. Stage B: Overall Baseline
    logger.info("Computing Stage B: Overall Baseline...")
    overall_df = overall_summary(df, target_col=TARGET_COLUMN)
    tables_dict["overall_baseline"] = overall_df
    baseline_rate_pct = float(overall_df["attrition_rate_pct"].iloc[0])
    p0 = baseline_rate_pct / 100.0

    # 3. Stage C: Organizational Hierarchy
    logger.info("Computing Stage C: Organizational Hierarchy (Department, JobRole, JobLevel, Concentration, Funnel)...")
    tables_dict["department_attrition"] = rate_table(df, "Department", target_col=TARGET_COLUMN)
    tables_dict["jobrole_attrition"] = rate_table(df, "JobRole", target_col=TARGET_COLUMN)
    tables_dict["joblevel_attrition"] = rate_table(df, "JobLevel", target_col=TARGET_COLUMN)
    tables_dict["department_x_jobrole"] = two_way_table(df, "Department", "JobRole", target_col=TARGET_COLUMN)
    tables_dict["leaver_concentration"] = concentration_table(df, ["JobRole", "Department"], target_col=TARGET_COLUMN)
    tables_dict["funnel_limits"] = funnel_table(df, ["JobRole", "Department"], p0=p0, levels=FUNNEL_LEVELS, target_col=TARGET_COLUMN)

    # 4. Stage D: Demographics
    logger.info("Computing Stage D: Demographics (AgeGroup, Gender, MaritalStatus, Education, EducationField)...")
    tables_dict["agegroup_attrition"] = rate_table(df, "AgeGroup", target_col=TARGET_COLUMN)
    tables_dict["gender_attrition"] = rate_table(df, "Gender", target_col=TARGET_COLUMN)
    tables_dict["maritalstatus_attrition"] = rate_table(df, "MaritalStatus", target_col=TARGET_COLUMN)
    tables_dict["education_attrition"] = rate_table(df, "Education", target_col=TARGET_COLUMN)
    tables_dict["educationfield_attrition"] = rate_table(df, "EducationField", target_col=TARGET_COLUMN)

    # 5. Stage E: Tenure & Career Progression
    logger.info("Computing Stage E: Tenure & Career Progression...")
    tables_dict["tenureband_attrition"] = rate_table(df, "TenureBand", target_col=TARGET_COLUMN)
    tables_dict["earlytenure_attrition"] = rate_table(df, "EarlyTenure", target_col=TARGET_COLUMN)
    tables_dict["careerstage_attrition"] = rate_table(df, "CareerStage", target_col=TARGET_COLUMN)
    tables_dict["roletenureband_attrition"] = rate_table(df, "RoleTenureBand", target_col=TARGET_COLUMN)
    tables_dict["managertenureband_attrition"] = rate_table(df, "ManagerTenureBand", target_col=TARGET_COLUMN)
    tables_dict["promotiongap_pooled_attrition"] = rate_table(df, "PromotionGapBand", target_col=TARGET_COLUMN)

    # Cohort analysis: PromotionAnalysisCohort == 1
    cohort_df = df[df["PromotionAnalysisCohort"] == 1].copy()
    tables_dict["promotiongap_cohort_attrition"] = rate_table(cohort_df, "PromotionGapBand", target_col=TARGET_COLUMN)

    # Side-by-side comparison table
    pooled_rt = tables_dict["promotiongap_pooled_attrition"].copy().set_index("segment")
    cohort_rt = tables_dict["promotiongap_cohort_attrition"].copy().set_index("segment")
    side_by_side_rows = []
    for band in pooled_rt.index:
        side_by_side_rows.append(
            {
                "PromotionGapBand": band,
                "pooled_n": int(pooled_rt.loc[band, "n"]),
                "pooled_leavers": int(pooled_rt.loc[band, "leavers"]),
                "pooled_attrition_rate_pct": pooled_rt.loc[band, "attrition_rate_pct"],
                "cohort_n": int(cohort_rt.loc[band, "n"]) if band in cohort_rt.index else 0,
                "cohort_leavers": int(cohort_rt.loc[band, "leavers"]) if band in cohort_rt.index else 0,
                "cohort_attrition_rate_pct": cohort_rt.loc[band, "attrition_rate_pct"] if band in cohort_rt.index else np.nan,
            }
        )
    tables_dict["promotiongap_side_by_side"] = pd.DataFrame(side_by_side_rows)
    tables_dict["tenureband_x_promotiongap"] = two_way_table(df, "TenureBand", "PromotionGapBand", target_col=TARGET_COLUMN)
    tables_dict["promotiongap_equals_tenure"] = rate_table(df, "PromotionGapEqualsTenure", target_col=TARGET_COLUMN)
    tables_dict["prior_experience_attrition"] = rate_table(df, "HasPriorExperience", target_col=TARGET_COLUMN)
    tables_dict["numcompanies_attrition"] = rate_table(df, "NumCompaniesWorked", target_col=TARGET_COLUMN)

    # 6. Stage F: Workload, Mobility & Environment
    logger.info("Computing Stage F: Workload, Mobility & Environment...")
    tables_dict["overtime_attrition"] = rate_table(df, "OverTime", target_col=TARGET_COLUMN)
    tables_dict["businesstravel_attrition"] = rate_table(df, "BusinessTravel", target_col=TARGET_COLUMN)
    tables_dict["distanceband_attrition"] = rate_table(df, "DistanceBand", target_col=TARGET_COLUMN)
    tables_dict["worklifebalance_attrition"] = rate_table(df, "WorkLifeBalance", target_col=TARGET_COLUMN)
    tables_dict["jobsatisfaction_attrition"] = rate_table(df, "JobSatisfaction", target_col=TARGET_COLUMN)
    tables_dict["environmentsatisfaction_attrition"] = rate_table(df, "EnvironmentSatisfaction", target_col=TARGET_COLUMN)
    tables_dict["relationshipsatisfaction_attrition"] = rate_table(df, "RelationshipSatisfaction", target_col=TARGET_COLUMN)
    tables_dict["jobinvolvement_attrition"] = rate_table(df, "JobInvolvement", target_col=TARGET_COLUMN)

    # 7. Stage G: Compensation & Numerical Summaries
    logger.info("Computing Stage G: Compensation & Numerical Distributions...")
    tables_dict["stockoption_attrition"] = rate_table(df, "StockOptionLevel", target_col=TARGET_COLUMN)
    tables_dict["trainingtimes_attrition"] = rate_table(df, "TrainingTimesLastYear", target_col=TARGET_COLUMN)
    tables_dict["performancerating_attrition"] = rate_table(df, "PerformanceRating", target_col=TARGET_COLUMN)
    tables_dict["monthlyincome_by_joblevel"] = monthly_income_by_joblevel(df, target_col=TARGET_COLUMN)

    numeric_cols_for_summary = [
        "Age",
        "MonthlyIncome",
        "DistanceFromHome",
        "PercentSalaryHike",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]
    tables_dict["numeric_distributions_by_outcome"] = numeric_summary_by_outcome(
        df, numeric_cols_for_summary, target_col=TARGET_COLUMN
    )

    # 8. Stage H: Two-Way Interactions & Hotspot Scanning
    logger.info("Computing Stage H: Two-Way Interactions & Risk Hotspot Scanning...")
    two_way_tables_map: Dict[str, pd.DataFrame] = {}
    for d1, d2 in TWO_WAY_SPECS:
        key = f"two_way_{d1.lower()}_x_{d2.lower()}"
        tw_df = two_way_table(df, d1, d2, target_col=TARGET_COLUMN)
        tables_dict[key] = tw_df
        two_way_tables_map[key] = tw_df

    candidates_df, total_scanned_cells = hotspot_candidates(
        two_way_tables_map,
        overall_rate_pct=baseline_rate_pct,
        min_n=HOTSPOT_MIN_N,
        top_k=HOTSPOT_TOP_K,
    )
    tables_dict["hotspot_candidates"] = candidates_df
    logger.info(
        f"Scanned {total_scanned_cells} interaction cells. Identified {len(candidates_df)} hotspot candidates with n >= {HOTSPOT_MIN_N} and Wilson LB > {baseline_rate_pct:.2f}%."
    )

    # 9. Stage I: Sensitivity Analysis
    logger.info("Computing Stage I: Sensitivity Analysis across alternative cuts...")
    tables_dict["sensitivity_analysis"] = sensitivity_table(
        df, SENSITIVITY_SPECS, target_col=TARGET_COLUMN, min_n=MIN_GROUP_SIZE
    )

    # 10. Save all CSV tables deterministically
    logger.info(f"Saving {len(tables_dict)} CSV tables to {PHASE3_TABLES_DIR}...")
    for name, table_df in tables_dict.items():
        csv_path = PHASE3_TABLES_DIR / f"{name}.csv"
        # Export with LF line endings and fixed precision
        table_df.to_csv(csv_path, index=False, lineterminator="\n", float_format=f"%.{ROUND_DIGITS}f")

    # 11. Compile Summary JSON
    logger.info("Compiling Phase 3A Summary JSON...")
    funnel_status_95 = tables_dict["funnel_limits"]["status_95"].value_counts().to_dict()
    funnel_status_998 = tables_dict["funnel_limits"]["status_99.8"].value_counts().to_dict()

    summary_data = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_file": PROCESSED_DATA_FILE.name,
        "dataset_sha256": processed_sha256,
        "environment": get_environment_metadata(),
        "dimensions": {
            "rows": len(df),
            "columns": len(df.columns),
        },
        "overall_metrics": {
            "total_employees": int(overall_df["total_employees"].iloc[0]),
            "leavers": int(overall_df["leavers"].iloc[0]),
            "retained": int(overall_df["retained"].iloc[0]),
            "attrition_rate_pct": float(overall_df["attrition_rate_pct"].iloc[0]),
            "ci95_low_pct": float(overall_df["ci95_low_pct"].iloc[0]),
            "ci95_high_pct": float(overall_df["ci95_high_pct"].iloc[0]),
            "imbalance_ratio": float(overall_df["imbalance_ratio"].iloc[0]),
        },
        "hotspot_candidates_scan": {
            "total_cells_scanned": total_scanned_cells,
            "min_n_threshold": HOTSPOT_MIN_N,
            "candidate_count": len(candidates_df),
            "candidates": candidates_df.to_dict(orient="records"),
        },
        "funnel_summary": {
            "status_95": funnel_status_95,
            "status_998": funnel_status_998,
        },
        "table_count": len(tables_dict),
    }

    with open(PHASE3_SUMMARY_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved summary JSON to {PHASE3_SUMMARY_JSON}")

    # 12. Build and save analysis plan & markdown report
    logger.info(f"Generating Analysis Plan at {ANALYSIS_PLAN_MD}...")
    plan_md = build_analysis_plan()
    ANALYSIS_PLAN_MD.write_text(plan_md, encoding="utf-8")

    logger.info(f"Generating EDA Tables Report at {PHASE3_REPORT_MD}...")
    report_md = build_phase3a_report(tables_dict, summary_data)
    PHASE3_REPORT_MD.write_text(report_md, encoding="utf-8")

    # 13. Update decision log
    update_decision_log(logger)

    logger.info("Phase 3A Exploratory Data Analysis Pipeline completed successfully.")


if __name__ == "__main__":
    main()
