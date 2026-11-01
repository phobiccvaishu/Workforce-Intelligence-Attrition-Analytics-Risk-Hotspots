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
