# Workforce Intelligence — Attrition Analytics & Risk Hotspots

An enterprise-grade, reproducible HR analytics, statistical inference, and workforce intelligence platform evaluating employee turnover dynamics, risk hotspots, and predictive triage with publication-grade deliverables.

> [!IMPORTANT]
> **PROVENANCE DISCLAIMER:**  
> The dataset provenance is **UNVERIFIED**. This data is analyzed strictly as an observational, synthetic-style HR benchmark dataset supplied under the file name "Palo Alto Networks.csv". Findings, models, and conclusions must never be described as real employee records or proprietary corporate data. Corporate logos are not used anywhere in this project. All findings represent statistical associations; individual employee scoring, risk ranking, and employment targeting are strictly prohibited.

---

## Setup

Ensure Python 3.10+ is installed (tested on Python 3.14.0 Windows x64).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 2. Install production dependencies
pip install -r requirements.txt
```

---

## Running the Application

Launch the interactive Streamlit executive dashboard:

```bash
# Windows (PowerShell):
$env:PYTHONPATH = "src;."
streamlit run app/Overview.py

# Windows (CMD):
set PYTHONPATH=src;.
streamlit run app/Overview.py

# Linux / macOS (bash):
PYTHONPATH="src:." streamlit run app/Overview.py
```

The application opens locally at **`http://localhost:8501`** and provides 5 multi-page modules:
- 📊 **Overview**: Baseline headcount, departure rates, and dynamic KPI evaluation with exact binomial Wilson confidence intervals.
- 🎯 **Risk Hotspots**: Empirical Bayes shrinkage, overlap-aware intersection graph, and greedy submodular coverage.
- ⚖️ **Adjusted Associations**: Mantel-Haenszel stratified odds ratios and multivariable logistic regression (Models M1 and M2) controlling for job hierarchy.
- 🤖 **Model Triage**: Precision-recall curves, risk tier triage yield, calibration assessments, and demographic fairness parity audits.
- 📜 **Decision Log**: Audited ledger of 44 technical architecture decisions (DL-01 through DL-44).

To run headless verification of all 5 dashboard pages and 12 pre-registered filter cases:
```bash
python scripts/run_phase7_app_checks.py
```

---

## Testing & Quality Assurance

The codebase includes an extensive automated test suite covering AST boundaries, data contracts, feature pipelines, statistical inference, figure geometries, and language hygiene.

```bash
# Run the full test suite (203 tests passing, 0 failures)
pytest -q

# Run end-to-end byte-reproducibility and determinism audit
python scripts/verify_full_pipeline_determinism.py
```

---

## Code-Generated Final Deliverables

All deliverables are generated directly from committed pipeline tables by code (zero hand-typed numbers, zero dollar signs, and strict language hygiene):

```bash
# Rebuild research paper, executive summary, and canonical figures
python scripts/build_final_docs.py
```

### Generated Artifacts (`reports/final/`)
- 📄 **Research Paper**: `reports/final/paper.md` (also exported to `.docx` and `.pdf` via Word COM automation)
- 📋 **Executive Summary**: `reports/final/executive_summary.md` (also exported to `.docx` and `.pdf`)
- 📈 **Canonical Figures**: `reports/final/figures/` (20 publication-quality PNG figures from Phases 3 and 4)
- 📑 **Submission Checklist**: `docs/final_submission_checklist.md` (deliverables manifest and governance checklist)

---

## Headline Findings

- **Overall Baseline**: 16.12% attrition rate (237 leavers out of 1,470 employees; 95% Wilson CI [14.33%, 18.09%]).
- **Early-Career Turnover Peak**: Personnel in their first year of tenure (0-1 years) exhibit an attrition rate of 34.88% (75/215, 95% CI [28.81%, 41.48%]), stabilizing to 11.56% by years 6-10.
- **Role Concentration**: Sales Representatives display the highest observed attrition at 39.76% (33/83), followed by Laboratory Technicians at 23.94% (62/259).
- **Overtime Association**: Overtime employees exhibit 30.53% attrition (127/416) vs 10.44% (110/1,054) for non-overtime staff. The association remains strong after controlling for job level (Mantel-Haenszel OR = 4.05 [3.01, 5.46]; multivariable M2 OR = 5.27 [3.77, 7.35]).
- **Promotion Gap**: Personnel with 6+ years since their last promotion (within the career cohort n=890) exhibit 15.35% attrition vs 8.13% for 0-2 years (Tenure-adjusted Mantel-Haenszel OR = 2.38 [1.44, 3.92]).
- **Combinatorial Hotspots**: A greedy combination of 5 non-redundant operational segments accounts for 67.51% of all departures while surveying only 30.68% of total workforce headcount.
- **Model Triage & Fairness**: Supervised triage models achieve a holdout PR-AUC of 0.5736 (top-10% precision 62.07%, recall 38.30%, lift 3.88x) with confirmed demographic parity across gender, age, and marital status.

---

## Project Structure

```text
├── data/
│   ├── raw/                        # Immutable raw dataset (SHA-256: 389735e8..., 209,376 bytes)
│   └── processed/                  # Engineered analysis datasets (1,470 x 39)
├── docs/
│   ├── data_dictionary.md          # 31-column data dictionary and scales
│   ├── decision_status.md          # Audited decision ledger (DL-01 to DL-44)
│   ├── findings_register.md        # Register of all statistical findings
│   ├── kpi_definitions.md          # Formal contracts for KPI-01 to KPI-08
│   └── final_submission_checklist.md # Deliverables manifest and validation checklist
├── src/workforce_intel/            # Core package modules
│   ├── config.py                   # Configuration, palette, static table checksums
│   ├── validation.py               # Data contract and validation checks
│   ├── cleaning.py                 # Reproducible preprocessing and feature engineering
│   ├── eda.py                      # Descriptive statistics and bivariate contingency tests
│   ├── viz.py                      # Figure layout and styling
│   ├── inference.py                # Wilson CIs, BH corrections, Mantel-Haenszel, M1/M2
│   ├── hotspot_index.py            # Empirical Bayes shrinkage and greedy coverage
│   ├── kpis.py                     # Hash-verified KPI computing engine
│   └── final_docs.py               # Document builder and Word COM export engine
├── app/                            # Multipage Streamlit Application
│   ├── Overview.py                 # Application entrypoint
│   ├── data_access.py              # Hash-verified static table loader
│   ├── components.py               # UI components and KPI metric cards
│   └── pages/                      # Hotspots, Adjusted Associations, Model Triage, Decision Log
├── scripts/
│   ├── build_final_docs.py         # Final paper and executive summary builder
│   ├── run_phase7_app_checks.py    # Headless Streamlit smoke verification
│   └── verify_full_pipeline_determinism.py # Determinism verification
├── reports/
│   ├── phase1/ .. phase6/          # Historical phase audit tables, figures, and JSON
│   └── final/                      # Final research paper, executive summary, and figures
└── tests/                          # 203 automated unit and integration tests (100% pass)
```

---

## Ethical & Responsible Use

This system is designed exclusively as an **aggregate organizational diagnostic** to guide company-wide workload balancing, onboarding support, and career progression frameworks.
- **No Individual Scoring**: Scores, odds ratios, and risk tiers must never be applied to individual employees for appraisal, compensation, or adverse personnel action.
- **No Demographic Targeting**: Age, gender, and marital status must never be utilized as operational targeting criteria or program eligibility factors.
- **Human Review**: All statistical associations serve as hypotheses for organizational inquiry, requiring contextual human review before operational changes.
