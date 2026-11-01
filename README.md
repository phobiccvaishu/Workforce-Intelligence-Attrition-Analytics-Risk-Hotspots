# Workforce Attrition Patterns and Risk Hotspot Analysis

An enterprise-grade, reproducible HR analytics and workforce intelligence system.

> [!IMPORTANT]
> **PROVENANCE DISCLAIMER:**  
> The dataset provenance is **UNVERIFIED**. This data is analyzed strictly as a public/synthetic HR benchmark dataset. Findings, models, and conclusions must never be described as real Palo Alto Networks employment records or proprietary corporate data. Corporate logos are not used anywhere in this project.

---

## 1. Project Overview & Architecture

This project investigates career stages, departments, roles, compensation structures, and workload conditions associated with employee turnover (attrition). Across project phases, the initiative delivers:
1. **Phase 1 (Current):** Project initialization, raw data preservation, structural contract verification, and rigorous mathematical data validation.
2. **Phase 2:** Reproducible data cleaning, feature engineering, and stratified Exploratory Data Analysis (EDA).
3. **Phase 3:** Risk hotspot identification, statistical testing, and predictive attrition modeling.
4. **Phase 4:** Interactive Streamlit executive dashboard, academic research paper, and executive decision briefings.

---

## 2. Directory Structure

```text
├── .gitignore                      # Git ignore rules (tracks data/raw, ignores cache/temp)
├── pytest.ini                      # Pytest runner configuration
├── requirements.txt                # Pinned dependencies for Phase 1 (and commented future phases)
├── README.md                       # Master project overview and execution instructions
├── data/
│   ├── raw/                        # Immutable raw dataset + cryptographic checksums
│   │   ├── Palo Alto Networks.csv  (Exact original copy, marked read-only)
│   │   └── CHECKSUMS.txt           (SHA-256: 389735e8... size: 209,376 bytes)
│   ├── interim/                    # Reserved for Phase 2 intermediate outputs
│   └── processed/                  # Reserved for Phase 2 cleaned datasets
├── docs/
│   ├── data_dictionary.md          # 31-column data dictionary, assumed scales, and notes
│   └── decision_log.md             # Detected data issues & proposed handling (all PROPOSED)
├── notebooks/
│   └── 01_data_inspection.ipynb    # Thin inspection notebook calling modular validation logic
├── src/
│   └── workforce_intel/            # Core Python package
│       ├── __init__.py             # Package marker
│       ├── config.py               # Paths, column groups, assumed scales, report palette
│       ├── io.py                   # Safe UTF-8/BOM reader with data/raw write-lock
│       └── validation.py           # Pure validation functions (Sections 5.5 a-l)
├── scripts/
│   ├── run_phase1_validation.py    # Idempotent one-command validation pipeline runner
│   └── make_notebook.py            # Generator script for inspection notebook
├── reports/
│   ├── phase1/
│   │   ├── tables/                 # 17 detailed CSV audit tables
│   │   ├── figures/                # Static 150 DPI figures (target, grid, boxplots, heatmap)
│   │   ├── data_validation_report.md # Generated validation report
│   │   ├── validation_summary.json # Machine-readable summary & checkpoint comparisons
│   │   ├── 01_data_inspection.executed.ipynb # Headless-executed notebook record
│   │   └── run.log                 # Pipeline execution log
│   ├── paper/                      # Later phase research paper
│   └── executive_summary/          # Later phase executive deliverables
├── app/                            # Later phase Streamlit application
└── tests/
    ├── __init__.py
    ├── test_validation.py          # Unit tests for validation functions across edge cases
    └── test_data_contract.py       # Structural data contract tests on real preserved data
```

---

## 3. Environment Setup & Execution

### Prerequisites
- Python >= 3.10 (Tested on Python 3.14.0)

### Setup Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install Phase 1 dependencies
pip install -r requirements.txt

# (Optional) Register kernel for Jupyter
python -m ipykernel install --user --name workforce-intel --display-name "Python (workforce-intel)"
```

### Running the Validation Pipeline
The full validation pipeline is deterministic and idempotent. It re-runs all data contract checks, computes all statistics, updates the report, and saves tables and figures:

```bash
# Set PYTHONPATH to include src
$env:PYTHONPATH = "src"
python scripts/run_phase1_validation.py
```

### Running Tests
Execute the unit and contract test suites:

```bash
$env:PYTHONPATH = "src"
python -m pytest -q
```
*Expected result:* 21 passed.

### Executing the Inspection Notebook
The thin inspection notebook imports pure functions from `workforce_intel`:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_inspection.ipynb --output reports/phase1/01_data_inspection.executed.ipynb
```
> [!NOTE]
> The inspection notebook execution has been verified independently in the environment with all 14 cells generating complete outputs and 0 errors, saved at `reports/phase1/01_data_inspection.executed.ipynb`.

---

## 4. Phase 1 Verification Summary

All Section 9 reference checkpoints were evaluated against programmatic outputs:

| Checkpoint | Target | Computed | Status |
| :--- | :--- | :--- | :--- |
| **Rows** | `1470` | `1470` | **MATCH** |
| **Columns** | `31` | `31` | **MATCH** |
| **Text Columns** | `7` | `7` | **MATCH** |
| **Missing Cells** | `0` | `0` | **MATCH** |
| **Full-Row Duplicates** | `0` | `0` | **MATCH** |
| **Attrition = 1** | `237` (16.12%) | `237` (16.12%) | **MATCH** |
| **Attrition = 0** | `1233` (83.88%) | `1233` (83.88%) | **MATCH** |
| **C1–C4 Violations** | `0` | `0` | **MATCH** |
| **C5 Hard Violations** | `0` | `0` | **MATCH** |
| **C6 Contradictions** | `197` | `197` | **MATCH** |
| **YearsSinceLastPromotion = 0** | `581` | `581` | **MATCH** |
| **YearsAtCompany = 0** | `44` | `44` | **MATCH** |
| **TotalWorkingYears = 0** | `11` | `11` | **MATCH** |
| **Dept x JobRole Populated Cells** | `11 of 27` | `11 of 27` | **MATCH** |
| **PerformanceRating Levels** | `[3, 4]` | `[3, 4]` | **MATCH** |
| **MonthlyIncome 1.5x Outliers** | `114` | `114` | **MATCH** |
| **MonthlyIncome vs JobLevel Rho** | `>= 0.7` | `0.9204` | **MATCH** |

---

## 5. Phase Status

- **Phase 1: Project Initialization & Data Validation**: **COMPLETE** (Awaiting owner sign-off on decision log).
- **Phase 2: Data Cleaning & Stratified EDA**: Pending.
- **Phase 3: Hotspot Analytics & Predictive Modeling**: Pending.
- **Phase 4: Streamlit Dashboard & Reporting**: Pending.
