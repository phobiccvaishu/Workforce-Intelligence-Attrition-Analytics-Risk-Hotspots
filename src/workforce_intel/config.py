"""Configuration module for Workforce Intelligence analytics project.

Defines project paths, column schema expectations, documented scale definitions,
and report styling palettes. Contains no magic numbers.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# Paths (Resolved relative to project root)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILENAME = "Palo Alto Networks.csv"
RAW_DATA_FILE = RAW_DATA_DIR / RAW_DATA_FILENAME
CHECKSUMS_FILE = RAW_DATA_DIR / "CHECKSUMS.txt"

INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_FILENAME = "workforce_attrition_processed.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / PROCESSED_DATA_FILENAME
PROCESSED_SCHEMA_FILENAME = "schema.json"
PROCESSED_SCHEMA_FILE = PROCESSED_DATA_DIR / PROCESSED_SCHEMA_FILENAME

DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DICTIONARY_FILE = DOCS_DIR / "data_dictionary.md"
DECISION_LOG_FILE = DOCS_DIR / "decision_log.md"

REPORTS_DIR = PROJECT_ROOT / "reports"
PHASE1_REPORTS_DIR = REPORTS_DIR / "phase1"
PHASE1_TABLES_DIR = PHASE1_REPORTS_DIR / "tables"
PHASE1_FIGURES_DIR = PHASE1_REPORTS_DIR / "figures"
PHASE1_REPORT_MD = PHASE1_REPORTS_DIR / "data_validation_report.md"
PHASE1_SUMMARY_JSON = PHASE1_REPORTS_DIR / "validation_summary.json"
PHASE1_LOG_FILE = PHASE1_REPORTS_DIR / "run.log"

PHASE2_REPORTS_DIR = REPORTS_DIR / "phase2"
PHASE2_TABLES_DIR = PHASE2_REPORTS_DIR / "tables"
PHASE2_REPORT_MD = PHASE2_REPORTS_DIR / "cleaning_report.md"
PHASE2_SUMMARY_JSON = PHASE2_REPORTS_DIR / "preprocessing_summary.json"
PHASE2_LOG_FILE = PHASE2_REPORTS_DIR / "run.log"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"

# -----------------------------------------------------------------------------
# Schema & Column Expectations (Exact names and order: 31 columns)
# -----------------------------------------------------------------------------
EXPECTED_COLUMNS: List[str] = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Target column
TARGET_COLUMN = "Attrition"

# Nominal columns (unordered categories)
NOMINAL_COLUMNS: List[str] = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Ordinal columns with documented expected scale ranges.
# NOTE: These ordinal ranges are ASSUMED scales based on standard public HR benchmark
# datasets. The validator reports observed ranges alongside these assumed scales.
ORDINAL_COLUMNS_DOCUMENTED_RANGES: Dict[str, Tuple[int, int]] = {
    "Education": (1, 5),
    "EnvironmentSatisfaction": (1, 4),
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "StockOptionLevel": (0, 3),
    "WorkLifeBalance": (1, 4),
}

# Assumed ordinal label mappings for documentation/validation
# NOTE: ASSUMPTION: Confirm with dataset provider.
ORDINAL_LABEL_ASSUMPTIONS: Dict[str, Dict[int, str]] = {
    "Education": {
        1: "Below College",
        2: "College",
        3: "Bachelor",
        4: "Master",
        5: "Doctor",
    },
    "EnvironmentSatisfaction": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High",
    },
    "JobInvolvement": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High",
    },
    "JobLevel": {
        1: "Entry Level",
        2: "Junior / Associate",
        3: "Mid / Senior",
        4: "Lead / Manager",
        5: "Director / Executive",
    },
    "JobSatisfaction": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High",
    },
    "PerformanceRating": {
        1: "Low",
        2: "Good",
        3: "Excellent",
        4: "Outstanding",
    },
    "RelationshipSatisfaction": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High",
    },
    "StockOptionLevel": {
        0: "Level 0 (None)",
        1: "Level 1",
        2: "Level 2",
        3: "Level 3",
    },
    "WorkLifeBalance": {
        1: "Bad",
        2: "Good",
        3: "Better",
        4: "Best",
    },
}

# Discrete count columns
DISCRETE_COUNT_COLUMNS: List[str] = [
    "NumCompaniesWorked",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "TotalWorkingYears",
]

# Continuous numeric columns
CONTINUOUS_COLUMNS: List[str] = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "PercentSalaryHike",
]

ALL_NUMERIC_COLUMNS: List[str] = (
    list(ORDINAL_COLUMNS_DOCUMENTED_RANGES.keys())
    + DISCRETE_COUNT_COLUMNS
    + CONTINUOUS_COLUMNS
)

# -----------------------------------------------------------------------------
# Static Audit Rules & Thresholds
# -----------------------------------------------------------------------------
SMALL_SAMPLE_THRESHOLD: int = 30
IQR_MULTIPLIER_OUTLIER: float = 1.5
IQR_MULTIPLIER_EXTREME: float = 3.0

# -----------------------------------------------------------------------------
# Report Palette & Styling (Brand-inspired palette only; no corporate logos)
# -----------------------------------------------------------------------------
PALETTE = {
    "primary": "#0A1F33",       # Deep navy
    "accent": "#FA582D",        # Vibrant orange
    "accent_soft": "#FF7650",   # Soft coral
    "slate_dark": "#2E3842",    # Dark slate
    "slate_med": "#5A6872",     # Medium grey
    "slate_light": "#DDE1E6",   # Light slate
    "background": "#F4F6F8",    # Off-white / light slate canvas
    "white": "#FFFFFF",
    "stayed": "#0A1F33",        # Navy for Attrition = 0
    "left": "#FA582D",          # Orange for Attrition = 1
}

# Provenance disclaimer mandated by project rules
PROVENANCE_DISCLAIMER = (
    "DISCLAIMER: The dataset provenance is UNVERIFIED. This data is analyzed as a "
    "public/synthetic HR dataset. Findings must not be described as real Palo Alto "
    "Networks employment records or personnel data."
)

# -----------------------------------------------------------------------------
# Phase 2: Derived Feature Configuration & Cut-Points
# -----------------------------------------------------------------------------
SURROGATE_KEY_COLUMN: str = "RowID"

EARLY_TENURE_MAX_YEARS: int = 1
MIN_TENURE_FOR_PROMOTION_ANALYSIS: int = 5
MIN_GROUP_SIZE: int = 30

AGE_BINS: List[int] = [17, 25, 35, 45, 60]
AGE_LABELS: List[str] = ["18-25", "26-35", "36-45", "46-60"]

TENURE_BINS: List[int] = [-1, 1, 3, 5, 10, 100]
TENURE_LABELS: List[str] = ["0-1", "2-3", "4-5", "6-10", "11+"]

CAREER_STAGE_BINS: List[int] = [-1, 5, 10, 20, 100]
CAREER_STAGE_LABELS: List[str] = ["0-5", "6-10", "11-20", "21+"]

ROLE_TENURE_BINS: List[int] = [-1, 1, 3, 7, 100]
ROLE_TENURE_LABELS: List[str] = ["0-1", "2-3", "4-7", "8+"]

MANAGER_TENURE_BINS: List[int] = [-1, 1, 3, 7, 100]
MANAGER_TENURE_LABELS: List[str] = ["0-1", "2-3", "4-7", "8+"]

PROMOTION_GAP_BINS: List[int] = [-1, 0, 2, 5, 100]
PROMOTION_GAP_LABELS: List[str] = ["0", "1-2", "3-5", "6+"]

DISTANCE_BINS: List[int] = [0, 5, 10, 20, 29]
DISTANCE_LABELS: List[str] = ["1-5", "6-10", "11-20", "21-29"]

DERIVED_COLUMNS: List[str] = [
    "AgeGroup",
    "TenureBand",
    "EarlyTenure",
    "CareerStage",
    "RoleTenureBand",
    "ManagerTenureBand",
    "PromotionGapBand",
    "DistanceBand",
    "PriorExperienceYears",
    "HasPriorExperience",
    "RoleTenureShare",
    "PromotionGapShare",
    "PromotionGapEqualsTenure",
    "PromotionAnalysisCohort",
    "OutstandingRating",
]

PROCESSED_COLUMNS: List[str] = [SURROGATE_KEY_COLUMN] + EXPECTED_COLUMNS + DERIVED_COLUMNS

PROTECTED_ATTRIBUTE_COLUMNS: List[str] = ["Age", "Gender", "MaritalStatus"]
NOT_USED_ANALYTICAL_COLUMNS: List[str] = ["DailyRate", "HourlyRate", "MonthlyRate"]
NULLABLE_DERIVED_COLUMNS: List[str] = ["RoleTenureShare", "PromotionGapShare"]


def get_environment_metadata() -> Dict[str, Any]:
    """Return runtime environment metadata for audit and reproducibility tracking."""
    import platform
    import numpy as np
    import pandas as pd
    import scipy

    try:
        import pyarrow
        pyarrow_installed = True
        pyarrow_version = pyarrow.__version__
    except ImportError:
        pyarrow_installed = False
        pyarrow_version = None

    return {
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pyarrow_installed": pyarrow_installed,
        "pyarrow_version": pyarrow_version,
        "pandas_string_storage": getattr(pd.options.mode, "string_storage", "default"),
    }


# -----------------------------------------------------------------------------
# Phase 3A Configuration & Analysis Specifications
# -----------------------------------------------------------------------------
PHASE3_REPORTS_DIR: Path = REPORTS_DIR / "phase3"
PHASE3_TABLES_DIR: Path = PHASE3_REPORTS_DIR / "tables"
PHASE3_REPORT_MD: Path = PHASE3_REPORTS_DIR / "eda_tables_report.md"
PHASE3_SUMMARY_JSON: Path = PHASE3_REPORTS_DIR / "eda_summary.json"
PHASE3_LOG_FILE: Path = PHASE3_REPORTS_DIR / "run.log"
ANALYSIS_PLAN_MD: Path = DOCS_DIR / "analysis_plan.md"

ROUND_DIGITS: int = 4
REPORT_RATE_DECIMALS: int = 1
FUNNEL_LEVELS: Tuple[float, float] = (0.95, 0.998)
HOTSPOT_MIN_N: int = 30
HOTSPOT_TOP_K: int = 15

TWO_WAY_SPECS: List[Tuple[str, str]] = [
    ("OverTime", "Department"),
    ("OverTime", "JobRole"),
    ("OverTime", "JobLevel"),
    ("OverTime", "AgeGroup"),
    ("OverTime", "TenureBand"),
    ("OverTime", "BusinessTravel"),
]

SENSITIVITY_SPECS: List[Dict[str, Any]] = [
    {
        "family": "EarlyTenure",
        "spec_name": "YearsAtCompany <= 1 (Default)",
        "type": "threshold",
        "column": "YearsAtCompany",
        "threshold": 1,
    },
    {
        "family": "EarlyTenure",
        "spec_name": "YearsAtCompany <= 2",
        "type": "threshold",
        "column": "YearsAtCompany",
        "threshold": 2,
    },
    {
        "family": "EarlyTenure",
        "spec_name": "YearsAtCompany <= 3",
        "type": "threshold",
        "column": "YearsAtCompany",
        "threshold": 3,
    },
    {
        "family": "AgeGroup",
        "spec_name": "4 Bands (Default: 18-25, 26-35, 36-45, 46-60)",
        "type": "bins",
        "column": "Age",
        "bins": [17, 25, 35, 45, 60],
        "labels": ["18-25", "26-35", "36-45", "46-60"],
    },
    {
        "family": "AgeGroup",
        "spec_name": "5 Bands (18-24, 25-34, 35-44, 45-54, 55-60)",
        "type": "bins",
        "column": "Age",
        "bins": [17, 24, 34, 44, 54, 60],
        "labels": ["18-24", "25-34", "35-44", "45-54", "55-60"],
    },
    {
        "family": "AgeGroup",
        "spec_name": "3 Bands (18-30, 31-45, 46-60)",
        "type": "bins",
        "column": "Age",
        "bins": [17, 30, 45, 60],
        "labels": ["18-30", "31-45", "46-60"],
    },
    {
        "family": "TenureBand",
        "spec_name": "5 Bands (Default: 0-1, 2-3, 4-5, 6-10, 11+)",
        "type": "bins",
        "column": "YearsAtCompany",
        "bins": [-1, 1, 3, 5, 10, 100],
        "labels": ["0-1", "2-3", "4-5", "6-10", "11+"],
    },
    {
        "family": "TenureBand",
        "spec_name": "4 Bands (0-2, 3-5, 6-10, 11+)",
        "type": "bins",
        "column": "YearsAtCompany",
        "bins": [-1, 2, 5, 10, 100],
        "labels": ["0-2", "3-5", "6-10", "11+"],
    },
    {
        "family": "DistanceBand",
        "spec_name": "Fixed Bins (Default: 1-5, 6-10, 11-20, 21-29)",
        "type": "bins",
        "column": "DistanceFromHome",
        "bins": [0, 5, 10, 20, 29],
        "labels": ["1-5", "6-10", "11-20", "21-29"],
    },
    {
        "family": "DistanceBand",
        "spec_name": "Empirical Quartiles",
        "type": "quantiles",
        "column": "DistanceFromHome",
        "q": 4,
    },
    {
        "family": "PromotionGapBand_Cohort",
        "spec_name": "Default Bins (0, 1-2, 3-5, 6+)",
        "type": "bins",
        "column": "YearsSinceLastPromotion",
        "filter_col": "PromotionAnalysisCohort",
        "filter_val": 1,
        "bins": [-1, 0, 2, 5, 100],
        "labels": ["0", "1-2", "3-5", "6+"],
    },
    {
        "family": "PromotionGapBand_Cohort",
        "spec_name": "Alternative Bins (0-1, 2-3, 4-6, 7+)",
        "type": "bins",
        "column": "YearsSinceLastPromotion",
        "filter_col": "PromotionAnalysisCohort",
        "filter_val": 1,
        "bins": [-1, 1, 3, 6, 100],
        "labels": ["0-1", "2-3", "4-6", "7+"],
    },
]

ANALYSIS_REGISTRY: List[Dict[str, Any]] = [
    {
        "stage": "Stage B",
        "name": "Overall Baseline",
        "type": "overall",
        "description": "Baseline workforce size, total leavers, overall attrition rate, and 95% Wilson confidence interval.",
        "table_name": "overall_baseline.csv",
    },
    {
        "stage": "Stage C",
        "name": "Department Breakdown",
        "type": "rate_table",
        "dimension": "Department",
        "description": "Attrition rate and leaver distribution by business unit.",
        "table_name": "department_attrition.csv",
    },
    {
        "stage": "Stage C",
        "name": "JobRole Breakdown",
        "type": "rate_table",
        "dimension": "JobRole",
        "description": "Attrition rate and leaver distribution across all 9 job roles.",
        "table_name": "jobrole_attrition.csv",
    },
    {
        "stage": "Stage C",
        "name": "Department x JobRole Hierarchy",
        "type": "two_way",
        "dim1": "Department",
        "dim2": "JobRole",
        "description": "Rate breakdown across populated organizational hierarchy cells.",
        "table_name": "department_x_jobrole.csv",
    },
    {
        "stage": "Stage C",
        "name": "JobLevel Breakdown",
        "type": "rate_table",
        "dimension": "JobLevel",
        "description": "Attrition rate by hierarchical job seniority level (1 to 5).",
        "table_name": "joblevel_attrition.csv",
    },
    {
        "stage": "Stage C",
        "name": "Leaver Concentration (Roles & Departments)",
        "type": "concentration",
        "dimensions": ["JobRole", "Department"],
        "description": "Cumulative leaver volume rankings illustrating organizational concentration.",
        "table_name": "leaver_concentration.csv",
    },
    {
        "stage": "Stage C",
        "name": "Funnel Control Limits (JobRole & Department)",
        "type": "funnel",
        "dimensions": ["JobRole", "Department"],
        "description": "Exact binomial control limits (95% and 99.8%) around overall baseline rate.",
        "table_name": "funnel_limits.csv",
    },
    {
        "stage": "Stage D",
        "name": "AgeGroup Breakdown",
        "type": "rate_table",
        "dimension": "AgeGroup",
        "description": "Attrition rate across age cohorts (descriptive evaluation).",
        "table_name": "agegroup_attrition.csv",
        "protected": True,
    },
    {
        "stage": "Stage D",
        "name": "Gender Breakdown",
        "type": "rate_table",
        "dimension": "Gender",
        "description": "Attrition rate by gender (descriptive evaluation).",
        "table_name": "gender_attrition.csv",
        "protected": True,
    },
    {
        "stage": "Stage D",
        "name": "MaritalStatus Breakdown",
        "type": "rate_table",
        "dimension": "MaritalStatus",
        "description": "Attrition rate by marital status (descriptive evaluation).",
        "table_name": "maritalstatus_attrition.csv",
        "protected": True,
    },
    {
        "stage": "Stage D",
        "name": "Education Breakdown",
        "type": "rate_table",
        "dimension": "Education",
        "description": "Attrition rate across educational attainment levels (1 to 5).",
        "table_name": "education_attrition.csv",
    },
    {
        "stage": "Stage D",
        "name": "EducationField Breakdown",
        "type": "rate_table",
        "dimension": "EducationField",
        "description": "Attrition rate by field of study, highlighting small-sample categories.",
        "table_name": "educationfield_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "TenureBand Breakdown",
        "type": "rate_table",
        "dimension": "TenureBand",
        "description": "Attrition rate across company tenure brackets.",
        "table_name": "tenureband_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "EarlyTenure Breakdown",
        "type": "rate_table",
        "dimension": "EarlyTenure",
        "description": "Attrition rate comparison for new joiners (<= 1 year tenure) versus tenured staff.",
        "table_name": "earlytenure_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "CareerStage Breakdown",
        "type": "rate_table",
        "dimension": "CareerStage",
        "description": "Attrition rate across total career experience brackets.",
        "table_name": "careerstage_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "RoleTenureBand Breakdown",
        "type": "rate_table",
        "dimension": "RoleTenureBand",
        "description": "Attrition rate across years in current role brackets.",
        "table_name": "roletenureband_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "ManagerTenureBand Breakdown",
        "type": "rate_table",
        "dimension": "ManagerTenureBand",
        "description": "Attrition rate across years with current manager brackets.",
        "table_name": "managertenureband_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "PromotionGapBand Pooled Breakdown",
        "type": "rate_table",
        "dimension": "PromotionGapBand",
        "description": "Attrition rate across promotion interval bands across all employees.",
        "table_name": "promotiongap_pooled_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "PromotionGapBand Cohort Breakdown",
        "type": "rate_table_cohort",
        "dimension": "PromotionGapBand",
        "filter_col": "PromotionAnalysisCohort",
        "filter_val": 1,
        "description": "Attrition rate across promotion interval bands restricted to eligible cohort (tenure >= 5).",
        "table_name": "promotiongap_cohort_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "PromotionGap Pooled vs Cohort Comparison",
        "type": "side_by_side",
        "description": "Side-by-side comparison of pooled versus cohort-restricted promotion gap rates.",
        "table_name": "promotiongap_side_by_side.csv",
    },
    {
        "stage": "Stage E",
        "name": "TenureBand x PromotionGapBand Matrix",
        "type": "two_way",
        "dim1": "TenureBand",
        "dim2": "PromotionGapBand",
        "description": "Cross-classification illustrating the intersection of organizational tenure and promotion delay.",
        "table_name": "tenureband_x_promotiongap.csv",
    },
    {
        "stage": "Stage E",
        "name": "PromotionGapEqualsTenure Breakdown",
        "type": "rate_table",
        "dimension": "PromotionGapEqualsTenure",
        "description": "Attrition rate for employees whose entire tenure matches their promotion interval.",
        "table_name": "promotiongap_equals_tenure.csv",
    },
    {
        "stage": "Stage E",
        "name": "HasPriorExperience Breakdown",
        "type": "rate_table",
        "dimension": "HasPriorExperience",
        "description": "Attrition rate comparing experienced joiners versus company-only careers.",
        "table_name": "prior_experience_attrition.csv",
    },
    {
        "stage": "Stage E",
        "name": "NumCompaniesWorked Breakdown",
        "type": "rate_table",
        "dimension": "NumCompaniesWorked",
        "description": "Attrition rate by reported number of previous employers, noting zero-count ambiguity.",
        "table_name": "numcompanies_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "OverTime Breakdown",
        "type": "rate_table",
        "dimension": "OverTime",
        "description": "Attrition rate comparison between overtime and non-overtime cohorts.",
        "table_name": "overtime_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "BusinessTravel Breakdown",
        "type": "rate_table",
        "dimension": "BusinessTravel",
        "description": "Attrition rate across business travel frequencies.",
        "table_name": "businesstravel_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "DistanceBand Breakdown",
        "type": "rate_table",
        "dimension": "DistanceBand",
        "description": "Attrition rate across commute distance brackets.",
        "table_name": "distanceband_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "WorkLifeBalance Breakdown",
        "type": "rate_table",
        "dimension": "WorkLifeBalance",
        "description": "Attrition rate across survey work-life balance scores (1 to 4).",
        "table_name": "worklifebalance_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "JobSatisfaction Breakdown",
        "type": "rate_table",
        "dimension": "JobSatisfaction",
        "description": "Attrition rate across job satisfaction scores (1 to 4).",
        "table_name": "jobsatisfaction_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "EnvironmentSatisfaction Breakdown",
        "type": "rate_table",
        "dimension": "EnvironmentSatisfaction",
        "description": "Attrition rate across work environment satisfaction scores (1 to 4).",
        "table_name": "environmentsatisfaction_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "RelationshipSatisfaction Breakdown",
        "type": "rate_table",
        "dimension": "RelationshipSatisfaction",
        "description": "Attrition rate across peer relationship satisfaction scores (1 to 4).",
        "table_name": "relationshipsatisfaction_attrition.csv",
    },
    {
        "stage": "Stage F",
        "name": "JobInvolvement Breakdown",
        "type": "rate_table",
        "dimension": "JobInvolvement",
        "description": "Attrition rate across job involvement scores (1 to 4).",
        "table_name": "jobinvolvement_attrition.csv",
    },
    {
        "stage": "Stage G",
        "name": "StockOptionLevel Breakdown",
        "type": "rate_table",
        "dimension": "StockOptionLevel",
        "description": "Attrition rate across equity grant tiers (0 to 3).",
        "table_name": "stockoption_attrition.csv",
    },
    {
        "stage": "Stage G",
        "name": "TrainingTimesLastYear Breakdown",
        "type": "rate_table",
        "dimension": "TrainingTimesLastYear",
        "description": "Attrition rate across annual training session frequencies (0 to 6).",
        "table_name": "trainingtimes_attrition.csv",
    },
    {
        "stage": "Stage G",
        "name": "PerformanceRating Breakdown",
        "type": "rate_table",
        "dimension": "PerformanceRating",
        "description": "Attrition rate between observed performance tiers 3 and 4.",
        "table_name": "performancerating_attrition.csv",
    },
    {
        "stage": "Stage G",
        "name": "MonthlyIncome Summary by JobLevel & Outcome",
        "type": "stratified_numeric",
        "stratify_col": "JobLevel",
        "metric_col": "MonthlyIncome",
        "description": "Compensation medians and quartiles stratified by job level and departure outcome.",
        "table_name": "monthlyincome_by_joblevel.csv",
    },
    {
        "stage": "Stage G",
        "name": "Numeric Summaries by Outcome",
        "type": "numeric_summary",
        "columns": [
            "Age",
            "MonthlyIncome",
            "DistanceFromHome",
            "PercentSalaryHike",
            "TotalWorkingYears",
            "YearsAtCompany",
            "YearsInCurrentRole",
            "YearsSinceLastPromotion",
            "YearsWithCurrManager",
        ],
        "description": "Distributional summary statistics (mean, std, median, quartiles) for continuous metrics by outcome.",
        "table_name": "numeric_distributions_by_outcome.csv",
    },
    {
        "stage": "Stage H",
        "name": "Hotspot Candidates Scan",
        "type": "hotspot_scan",
        "description": "Screening of all two-way interaction cells with n >= 30 where Wilson lower bound exceeds overall baseline.",
        "table_name": "hotspot_candidates.csv",
    },
    {
        "stage": "Stage I",
        "name": "Sensitivity Analysis",
        "type": "sensitivity",
        "description": "Evaluation of alternative bin boundary definitions and threshold assumptions.",
        "table_name": "sensitivity_analysis.csv",
    },
]
