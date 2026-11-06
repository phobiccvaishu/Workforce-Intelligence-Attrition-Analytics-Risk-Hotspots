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

# -----------------------------------------------------------------------------
# Phase 3B: Visualization & Findings Configuration
# -----------------------------------------------------------------------------

PHASE3_FIGURES_DIR: Path = PHASE3_REPORTS_DIR / "figures"
PHASE3_MANIFEST_CSV: Path = PHASE3_REPORTS_DIR / "figure_manifest.csv"
FINDINGS_REGISTER_MD: Path = DOCS_DIR / "findings_register.md"
FINDINGS_SUMMARY_JSON: Path = PHASE3_REPORTS_DIR / "findings_summary.json"
NOTEBOOK_FIGURES_IPYNB: Path = NOTEBOOKS_DIR / "03_eda_figures.ipynb"
EXECUTED_FIGURES_IPYNB: Path = PHASE3_REPORTS_DIR / "03_eda_figures.executed.ipynb"

FIGURE_DPI = 150
FIGURE_PALETTE = {
    "primary": "#1E3A8A",      # Navy for neutral/standard
    "highlight": "#EA580C",    # Orange for flagged/highlight
    "muted": "#9CA3AF",        # Muted grey for low sample size (n < 30)
    "neutral": "#1E3A8A",
    "low_n": "#9CA3AF",
    "baseline": "#DC2626",     # Red dashed line for baseline
    "secondary": "#2563EB",    # Royal blue
    "stayed": "#1E3A8A",
    "left": "#EA580C",
}

FIGURE_FOOTNOTE_STANDARD = "Public/synthetic HR dataset (provenance unverified). Associations, not causes."
FIGURE_FOOTNOTE_ORDINAL = "Public/synthetic HR dataset (provenance unverified). Associations, not causes. Ordinal labels are assumptions."
FIGURE_FOOTNOTE_PROTECTED = "Public/synthetic HR dataset (provenance unverified). Associations, not causes. Descriptive only; not for individual scoring."

FIGURE_SPECS = {
    "FIG-01": {
        "figure_id": "FIG-01",
        "file": "fig01_department_overview.png",
        "title": "Attrition Rate and Representation by Department",
        "subtitle": "Whiskers are 95% Wilson intervals; dashed line is workforce baseline",
        "source_tables": ["department_attrition.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Two-panel chart showing attrition rate with 95% Wilson CI by department, and comparison of workforce share vs leaver share.",
    },
    "FIG-02": {
        "figure_id": "FIG-02",
        "file": "fig02_jobrole_attrition.png",
        "title": "Attrition Rate by Job Role (Ranked)",
        "subtitle": "Whiskers are 95% Wilson intervals; dashed line is workforce baseline",
        "source_tables": ["jobrole_attrition.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Horizontal bar chart of attrition rates across 9 job roles ranked from highest to lowest.",
    },
    "FIG-03": {
        "figure_id": "FIG-03",
        "file": "fig03_funnel_plot.png",
        "title": "Exact Binomial Funnel Plot for Job Roles",
        "subtitle": "Exact binomial control limits (95% and 99.8%) around workforce baseline",
        "source_tables": ["funnel_limits.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Funnel plot showing job role attrition rates plotted against headcount with 95% and 99.8% binomial limits.",
    },
    "FIG-04": {
        "figure_id": "FIG-04",
        "file": "fig04_leaver_concentration.png",
        "title": "Cumulative Leaver Concentration Pareto by Job Role",
        "subtitle": "Cumulative share of departures across job roles against 80% reference threshold",
        "source_tables": ["leaver_concentration.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Pareto chart illustrating cumulative headcount and leaver concentration across job roles.",
    },
    "FIG-05": {
        "figure_id": "FIG-05",
        "file": "fig05_age_sensitivity.png",
        "title": "Attrition Rate by Age Cohort across Alternative Bins",
        "subtitle": "Whiskers are 95% Wilson intervals; dashed line is workforce baseline",
        "source_tables": ["agegroup_attrition.csv", "sensitivity_analysis.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_PROTECTED,
        "alt_text": "Three-panel comparison of age band attrition rates under 4-band, 5-band, and 3-band specifications.",
    },
    "FIG-06": {
        "figure_id": "FIG-06",
        "file": "fig06_tenure_sensitivity.png",
        "title": "Attrition Rate by Tenure Band and Early Tenure Thresholds",
        "subtitle": "Whiskers are 95% Wilson intervals; dashed line is workforce baseline",
        "source_tables": ["tenureband_attrition.csv", "sensitivity_analysis.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Two-panel chart showing tenure band attrition rates and sensitivity across early tenure thresholds.",
    },
    "FIG-07": {
        "figure_id": "FIG-07",
        "file": "fig07_promotion_gap_cohort.png",
        "title": "Promotion Gap Attrition: Pooled vs PromotionAnalysisCohort",
        "subtitle": "Whiskers are 95% Wilson intervals; panel B and C benchmarked against cohort baseline",
        "source_tables": ["promotiongap_side_by_side.csv", "sensitivity_analysis.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Three-panel chart displaying promotion gap attrition rates pooled across all employees, within the cohort, and across alternative bins.",
    },
    "FIG-08": {
        "figure_id": "FIG-08",
        "file": "fig08_workload_mobility.png",
        "title": "Attrition Rate by OverTime, Business Travel, and Distance to Office",
        "subtitle": "Whiskers are 95% Wilson intervals; dashed line is workforce baseline",
        "source_tables": ["overtime_attrition.csv", "businesstravel_attrition.csv", "distanceband_attrition.csv", "sensitivity_analysis.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Multi-panel chart showing attrition rates across overtime status, business travel frequency, and commute distance bands.",
    },
    "FIG-09": {
        "figure_id": "FIG-09",
        "file": "fig09_satisfaction_small_multiples.png",
        "title": "Attrition Rate across Employee Survey & Sentiment Dimensions",
        "subtitle": "Whiskers are 95% Wilson intervals; dashed line is workforce baseline; 4-point rating scales",
        "source_tables": ["worklifebalance_attrition.csv", "jobsatisfaction_attrition.csv", "environmentsatisfaction_attrition.csv", "relationshipsatisfaction_attrition.csv", "jobinvolvement_attrition.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_ORDINAL,
        "alt_text": "Small multiples showing attrition rates across 5 survey metrics: Work-Life Balance, Job Satisfaction, Environment Satisfaction, Relationship Satisfaction, and Job Involvement.",
    },
    "FIG-10": {
        "figure_id": "FIG-10",
        "file": "fig10_overtime_interactions.png",
        "title": "OverTime Attrition Interactions with Career & Demographic Segments",
        "subtitle": "Cell attrition rates annotated with headcount; cells with n < 30 hatched grey",
        "source_tables": ["two_way_overtime_x_joblevel.csv", "two_way_overtime_x_agegroup.csv", "two_way_overtime_x_tenureband.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_PROTECTED,
        "alt_text": "Three heatmap panels displaying attrition rates for OverTime crossed with JobLevel, AgeGroup, and TenureBand.",
    },
    "FIG-11": {
        "figure_id": "FIG-11",
        "file": "fig11_hotspot_candidates.png",
        "title": "Screened Risk Hotspot Candidates",
        "subtitle": "Screening rule: two-way interaction cells with n >= 30 where Wilson lower bound exceeds overall baseline;\ncandidates are selected because their lower bound exceeds the baseline; rates are expected to be optimistic",
        "source_tables": ["hotspot_candidates.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Dot plot of 16 qualified hotspot candidate subgroups with 95% Wilson intervals and sample sizes.",
    },
    "FIG-12": {
        "figure_id": "FIG-12",
        "file": "fig12_hotspot_overlap.png",
        "title": "Hotspot Candidate Pairwise Overlap and Workforce Union Summary",
        "subtitle": "Pairwise intersection matrix and distinct employee union summary",
        "source_tables": ["hotspot_overlap.csv", "hotspot_candidates.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_STANDARD,
        "alt_text": "Pairwise overlap matrix heatmap and summary bars showing headcount and leavers inside vs outside candidate union.",
    },
    "FIG-13": {
        "figure_id": "FIG-13",
        "file": "fig13_numeric_distributions.png",
        "title": "Continuous Metric Distributions by Outcome (Median and Interquartile Range)",
        "subtitle": "Points show median; whiskers show interquartile range (25th to 75th percentiles)",
        "source_tables": ["numeric_distributions_by_outcome.csv"],
        "footnote": FIGURE_FOOTNOTE_PROTECTED,
        "alt_text": "Distributional comparison showing median and IQR for continuous features stratified by retention outcome.",
    },
    "FIG-14": {
        "figure_id": "FIG-14",
        "file": "fig14_demographics_background.png",
        "title": "Attrition Rate across Demographic and Professional Background Dimensions",
        "subtitle": "Whiskers are 95% Wilson intervals; panel titles generated from mechanical findings",
        "source_tables": ["gender_attrition.csv", "maritalstatus_attrition.csv", "education_attrition.csv", "educationfield_attrition.csv", "prior_experience_attrition.csv", "performancerating_attrition.csv", "overall_baseline.csv"],
        "footnote": FIGURE_FOOTNOTE_PROTECTED,
        "alt_text": "Six-panel display of attrition rates across Gender, MaritalStatus, Education, EducationField, HasPriorExperience, and PerformanceRating.",
    },
}
FINDINGS_RULE = {
    "min_n": 30,
    "alpha": 0.05,
    "baseline_table": "overall_baseline.csv",
    "cohort_baseline_table": "promotiongap_cohort_attrition.csv",
}

ANALYSIS_DIMENSIONS = [
    {"dimension": "Department", "table": "department_attrition.csv", "protected": False},
    {"dimension": "JobRole", "table": "jobrole_attrition.csv", "protected": False},
    {"dimension": "JobLevel", "table": "joblevel_attrition.csv", "protected": False},
    {"dimension": "AgeGroup", "table": "agegroup_attrition.csv", "protected": True},
    {"dimension": "Gender", "table": "gender_attrition.csv", "protected": True},
    {"dimension": "MaritalStatus", "table": "maritalstatus_attrition.csv", "protected": True},
    {"dimension": "Education", "table": "education_attrition.csv", "protected": False},
    {"dimension": "EducationField", "table": "educationfield_attrition.csv", "protected": False},
    {"dimension": "TenureBand", "table": "tenureband_attrition.csv", "protected": False},
    {"dimension": "EarlyTenure", "table": "earlytenure_attrition.csv", "protected": False},
    {"dimension": "CareerStage", "table": "careerstage_attrition.csv", "protected": False},
    {"dimension": "RoleTenureBand", "table": "roletenureband_attrition.csv", "protected": False},
    {"dimension": "ManagerTenureBand", "table": "managertenureband_attrition.csv", "protected": False},
    {"dimension": "PromotionGapBand", "table": "promotiongap_pooled_attrition.csv", "protected": False},
    {"dimension": "PromotionGapEqualsTenure", "table": "promotiongap_equals_tenure.csv", "protected": False},
    {"dimension": "HasPriorExperience", "table": "prior_experience_attrition.csv", "protected": False},
    {"dimension": "NumCompaniesWorked", "table": "numcompanies_attrition.csv", "protected": False},
    {"dimension": "OverTime", "table": "overtime_attrition.csv", "protected": False},
    {"dimension": "BusinessTravel", "table": "businesstravel_attrition.csv", "protected": False},
    {"dimension": "DistanceBand", "table": "distanceband_attrition.csv", "protected": False},
    {"dimension": "WorkLifeBalance", "table": "worklifebalance_attrition.csv", "protected": False},
    {"dimension": "JobSatisfaction", "table": "jobsatisfaction_attrition.csv", "protected": False},
    {"dimension": "EnvironmentSatisfaction", "table": "environmentsatisfaction_attrition.csv", "protected": False},
    {"dimension": "RelationshipSatisfaction", "table": "relationshipsatisfaction_attrition.csv", "protected": False},
    {"dimension": "JobInvolvement", "table": "jobinvolvement_attrition.csv", "protected": False},
    {"dimension": "StockOptionLevel", "table": "stockoption_attrition.csv", "protected": False},
    {"dimension": "TrainingTimesLastYear", "table": "trainingtimes_attrition.csv", "protected": False},
    {"dimension": "PerformanceRating", "table": "performancerating_attrition.csv", "protected": False},
]

PHASE4_REPORTS_DIR = REPORTS_DIR / "phase4"
PHASE4_TABLES_DIR = PHASE4_REPORTS_DIR / "tables"

# -----------------------------------------------------------------------------
# Phase 5: Inferential Analysis Configuration & Parameters
# -----------------------------------------------------------------------------
PHASE5_REPORTS_DIR = REPORTS_DIR / "phase5"
PHASE5_TABLES_DIR = PHASE5_REPORTS_DIR / "tables"
PHASE5_REPORT_MD = PHASE5_REPORTS_DIR / "inference_report.md"
PHASE5_SUMMARY_JSON = PHASE5_REPORTS_DIR / "inference_summary.json"
INFERENCE_PLAN_MD = DOCS_DIR / "inference_plan.md"

# Pre-specified analysis families
CATEGORICAL_ORDINAL_DIMENSIONS: List[str] = [d["dimension"] for d in ANALYSIS_DIMENSIONS]

ORDERED_DIMENSIONS_SCORES: Dict[str, Dict[any, int]] = {
    "AgeGroup": {"18-25": 1, "26-35": 2, "36-45": 3, "46-60": 4},
    "TenureBand": {"0-1": 1, "2-3": 2, "4-5": 3, "6-10": 4, "11+": 5},
    "CareerStage": {"0-5": 1, "6-10": 2, "11-20": 3, "21+": 4},
    "RoleTenureBand": {"0-1": 1, "2-3": 2, "4-7": 3, "8+": 4},
    "ManagerTenureBand": {"0-1": 1, "2-3": 2, "4-7": 3, "8+": 4},
    "PromotionGapBand": {"0": 1, "1-2": 2, "3-5": 3, "6+": 4},
    "DistanceBand": {"1-5": 1, "6-10": 2, "11-20": 3, "21-29": 4},
    "JobLevel": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "Education": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "EnvironmentSatisfaction": {1: 1, 2: 2, 3: 3, 4: 4},
    "JobSatisfaction": {1: 1, 2: 2, 3: 3, 4: 4},
    "RelationshipSatisfaction": {1: 1, 2: 2, 3: 3, 4: 4},
    "WorkLifeBalance": {1: 1, 2: 2, 3: 3, 4: 4},
    "JobInvolvement": {1: 1, 2: 2, 3: 3, 4: 4},
    "StockOptionLevel": {0: 0, 1: 1, 2: 2, 3: 3},
    "TrainingTimesLastYear": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
}

ORDERED_DIMENSIONS: List[str] = list(ORDERED_DIMENSIONS_SCORES.keys())

NUMERIC_MEASURES: List[str] = [
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

# Adjusted logistic regression covariate sets
M1_FEATURES: List[str] = [
    "OverTime",
    "BusinessTravel",
    "JobLevel",
    "StockOptionLevel",
    "DistanceFromHome",
    "YearsAtCompany",
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "WorkLifeBalance",
    "JobInvolvement",
]

M2_CONTROLS: List[str] = ["Age", "Gender", "MaritalStatus"]
M2_FEATURES: List[str] = M1_FEATURES + M2_CONTROLS

LOGISTIC_EXCLUDED_COVARIATES: List[str] = [
    "MonthlyIncome",
    "DailyRate",
    "HourlyRate",
    "MonthlyRate",
    "TotalWorkingYears",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
LOGISTIC_EXCLUSION_REASON: str = (
    "Excluded due to severe collinearity with included tenure, compensation tier (JobLevel), and baseline pay structures."
)

# Hotspot index universe definition
HOTSPOT_ONE_WAY_COLS: List[str] = [
    "Department",
    "JobRole",
    "JobLevel",
    "TenureBand",
    "OverTime",
    "BusinessTravel",
    "DistanceBand",
    "StockOptionLevel",
    "WorkLifeBalance",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "TrainingTimesLastYear",
]

HOTSPOT_TWO_WAY_PAIRS: List[Tuple[str, str]] = [
    ("OverTime", "Department"),
    ("OverTime", "JobRole"),
    ("OverTime", "JobLevel"),
    ("OverTime", "TenureBand"),
    ("OverTime", "BusinessTravel"),
]

HOTSPOT_EXCLUDED_DIMENSIONS: List[str] = [
    "Age",
    "AgeGroup",
    "Gender",
    "MaritalStatus",
    "Education",
    "EducationField",
    "PromotionGapBand",
]

# Statistical thresholds and hyperparameters
SEED: int = 42
RANDOM_SEED: int = SEED
MIN_GROUP_SIZE: int = 30
ALPHA_LEVEL: float = 0.05
N_BOOTSTRAP_INFERENCE: int = 2000
N_PERMUTATIONS_INFERENCE: int = 10000
PRIOR_STRENGTH_SENSITIVITY: List[int] = [10, 30, 100]
MAX_GREEDY_STEPS: int = 10


