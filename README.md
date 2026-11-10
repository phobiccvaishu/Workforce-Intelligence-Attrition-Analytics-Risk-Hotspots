# Workforce Attrition Patterns and Risk Hotspot Analysis

An enterprise-grade, reproducible HR analytics, statistical inference, and workforce intelligence platform.

> [!IMPORTANT]
> **PROVENANCE DISCLAIMER:**  
> The dataset provenance is **UNVERIFIED**. This data is analyzed strictly as an observational, synthetic-style HR benchmark dataset supplied under the file name "Palo Alto Networks.csv". Findings, models, and conclusions must never be described as real employee records or proprietary corporate data. Corporate logos are not used anywhere in this project. All findings represent statistical associations; individual employee scoring and flight-risk profiling are strictly prohibited.

---

## 1. Project Overview & Current Status

**Current Status:** All phases (Phases 1 through 7 + Final Submission) are fully completed, verified, and audited.

1. **Phase 1: Validation & Integrity** — Raw data cryptographic write-lock, structural contract verification, and mathematical data validation.
2. **Phase 2: Preprocessing & Baseline** — Reproducible data cleaning, deterministic feature derivation, and contract enforcement.
3. **Phase 3: Exploratory Data Analysis & Hotspots** — 20 canonical publication-quality figures, bivariate analyses, Wilson score confidence intervals, and empirical Bayes hotspot indexing.
4. **Phase 4: Predictive Triage & Consistency Audit** — Regularized logistic regression, gradient boosting, PR-AUC / ROC-AUC / Brier evaluation, demographic fairness audits, and 57-finding cross-split consistency verification.
5. **Phase 5: Inferential Modeling** — Multiple testing controls (Benjamini-Hochberg FDR, family size 28), stratified Mantel-Haenszel odds ratios across JobLevel/Department/Tenure, multivariable logistic models (M1, M2), and greedy combinatorial hotspot coverage.
6. **Phase 6: Executive KPIs** — 8 pre-registered core operational KPIs with exact binomial Wilson confidence intervals and audited reliability contracts.
7. **Phase 7: Multipage Streamlit Dashboard** — Interactive presentation layer (Overview, Hotspots, Adjusted Associations, Model Triage, Decision Log) with strict AST-enforced statistical read-only boundaries and SHA-256 table integrity verification.
8. **Final Submission: Academic Research Paper & Executive Summary** — Fully code-generated research paper (`reports/final/paper.*`), executive summary (`reports/final/executive_summary.*`), and copied canonical figures (`reports/final/figures/`).

---

## 2. Directory Structure

```text
├── data/
│   ├── raw/                        # Immutable raw dataset + cryptographic checksums
│   │   ├── Palo Alto Networks.csv  (SHA-256: 389735e8..., 209,376 bytes, byte-exact)
│   │   └── CHECKSUMS.txt           (Audited raw checksum)
│   ├── interim/                    # Intermediate preprocessed artifacts
│   └── processed/                  # Cleaned, engineered analysis datasets (1,470 x 39)
├── docs/
│   ├── data_dictionary.md          # 31-column data dictionary, assumed scales, notes
│   ├── decision_log.md             # Detailed engineering decisions
│   ├── decision_status.md          # Audited technical decision status (DL-01 to DL-44)
│   ├── findings_register.md        # Audited register of all statistical findings
│   ├── kpi_definitions.md          # Formal definitions and contracts for KPI-01 to KPI-08
│   ├── model_card.md               # Model cards for supervised triage models
│   └── final_submission_checklist.md # Deliverables manifest and validation checklist
├── src/workforce_intel/            # Production Python package
│   ├── config.py                   # Paths, constants, palette, table checksum registry
│   ├── io.py                       # Safe UTF-8/BOM reader with data/raw write-lock
│   ├── validation.py               # Data contract and validation functions
│   ├── cleaning.py                 # Data cleaning and feature engineering
│   ├── eda.py                      # Descriptive statistics and contingency analysis
│   ├── viz.py                      # Figure rendering and publication layout
│   ├── inference.py                # Wilson CIs, BH corrections, Mantel-Haenszel, M1/M2
│   ├── hotspot_index.py            # Empirical Bayes shrinkage and greedy coverage
│   ├── kpis.py                     # Hash-verified KPI computing and caching
│   └── final_docs.py               # Document builder and Word COM export engine
├── app/                            # Multipage Streamlit Application
│   ├── Overview.py                 # Landing page & high-level workforce metrics
│   ├── data_loader.py              # SHA-256 hash-verified tabular data loader
│   ├── components.py               # Visual components and KPI display cards
│   └── pages/
│       ├── 1_Hotspots.py           # Hotspot universe, shrinkage, and greedy coverage
│       ├── 2_Adjusted_Associations.py # OverTime, Mantel-Haenszel, and M1/M2 models
│       ├── 3_Model_Triage.py       # Supervised triage, calibration, fairness audits
│       └── 4_Decision_Log.py       # Audited decision ledger (DL-01 to DL-44)
├── scripts/
│   ├── run_phase1_validation.py    # Phase 1 runner
│   ├── run_phase2_preprocessing.py # Phase 2 runner
│   ├── run_phase3a_eda.py          # Phase 3a runner
│   ├── run_phase3b_figures.py      # Phase 3b runner
│   ├── run_phase4_models.py        # Phase 4 runner
│   ├── run_phase5_inference.py     # Phase 5 runner
│   ├── run_phase6_kpis.py          # Phase 6 runner
│   ├── run_phase7_app_checks.py    # Headless Streamlit smoke verification
│   ├── build_final_docs.py         # Final paper and executive summary builder
│   └── verify_full_pipeline_determinism.py # End-to-end byte-reproducibility audit
├── reports/
│   ├── phase1/ .. phase6/          # Historical phase audit tables, JSON, and figures
│   └── final/                      # Final submission deliverables
│       ├── paper.md / .docx / .pdf # Comprehensive research paper
│       ├── executive_summary.md / .docx / .pdf # Plain-language executive summary
│       └── figures/                # 20 canonical figures from Phases 3 and 4
└── tests/                          # 200+ unit and integration tests (100% pass)
```

---

## 3. Quickstart & Environment Setup

### Prerequisites
- Python 3.10+ (Developed and tested on Python 3.14.0 Windows x64)
- Microsoft Word (Optional, required only for native `.docx` and `.pdf` COM compilation)

### Virtual Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

---

## 4. Running Tests & Pipeline Verification

```powershell
# Run the full test suite (all tests passing)
.venv\Scripts\pytest -q

# Run end-to-end pipeline determinism audit
.venv\Scripts\python scripts/verify_full_pipeline_determinism.py
```

---

## 5. Launching the Interactive Dashboard

Launch the Streamlit executive dashboard locally:
```powershell
# Launch dashboard on http://localhost:8501
$env:PYTHONPATH = "src;."
.venv\Scripts\streamlit run app/Overview.py
```

To run headless verification of all 5 dashboard pages and 12 pre-registered filter cases:
```powershell
.venv\Scripts\python scripts/run_phase7_app_checks.py
```

---

## 6. Rebuilding Final Deliverables

To re-generate the research paper and executive summary from committed pipeline tables:
```powershell
# Executes src/workforce_intel/final_docs.py
.venv\Scripts\python scripts/build_final_docs.py
```
This command reads all statistics directly from `reports/phase1..6/tables/`, enforces zero causal language and zero dollar signs, copies the canonical figures into `reports/final/figures/`, and exports Markdown, DOCX, and PDF formats.

---

## 7. Key Findings Summary

- **Workforce Baseline**: 16.12% attrition rate (237 leavers out of 1,470 employees; 95% Wilson CI [14.33%, 18.09%]).
- **Early-Tenure Risk**: Personnel with 0-1 years of tenure exhibit 34.88% attrition (75/215), compared to 11.56% for tenure 6-10 years.
- **Role Concentration**: Sales Representatives experience 39.76% attrition (33/83), followed by Lab Technicians at 23.94% (62/259).
- **Overtime Association**: Overtime employees exhibit 30.53% attrition (127/416) vs 10.44% (110/1,054) for non-overtime employees. The association remains strong after controlling for job level (Mantel-Haenszel OR = 4.05 [3.01, 5.46]; multivariable M2 OR = 5.27 [3.77, 7.35]).
- **Promotion Gap**: Employees with 6+ years since their last promotion (within the career cohort n=890) exhibit 15.35% attrition vs 8.13% for 0-2 years (Tenure-adjusted Mantel-Haenszel OR = 2.38 [1.44, 3.92]).
- **Hotspot Greedy Coverage**: A combination of 5 operational segments accounts for 67.51% of all leavers while surveying only 30.68% of the workforce.
