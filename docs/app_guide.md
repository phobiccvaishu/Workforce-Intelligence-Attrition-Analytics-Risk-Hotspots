# Workforce Intelligence Dashboard Application Guide

## 1. Overview and Architecture

The Workforce Intelligence Dashboard is a multipage Streamlit application designed as a pure presentation layer for workforce attrition analytics, hotspot identification, multivariable inference, model validation, and governance.

The application adheres strictly to the separation of concerns:
- **Presentation Layer Only**: The `app/` package contains zero statistical logic, zero modeling algorithms, and zero manual rate arithmetic.
- **Pure Function Analytics Engine**: All indicators are evaluated through pure functions in `workforce_intel.kpis`, which reuse vetted Phase 3 and Phase 5 routines.
- **Cryptographic Hash Verification**: All precomputed benchmark tables and figures are verified against known SHA-256 digests in `workforce_intel.config.STATIC_TABLE_CHECKSUMS` before being displayed.
- **Strict Aggregation Invariant**: No employee-level identifiers, individual risk scores, or row-level DataFrames are displayed or exported. All visualizations and downloadable tables represent aggregate group summaries.

---

## 2. Page Taxonomy and Structure

The application is structured into five native Streamlit pages:

### 1. Overview (`app/Overview.py`)
- **Nature**: Dynamic and interactive.
- **Capabilities**:
  - Live filterable indicators (KPI-01 through KPI-05).
  - Headcount, leavers, and attrition rate with 95% Wilson score confidence intervals.
  - Relative-to-baseline rate comparisons.
  - OverTime bivariate association odds ratios (Woolf 95% confidence intervals).
  - Promotion-analysis cohort evaluation (employees with tenure >= 5 years).
  - Active filter banner and reset controls.
  - Graceful handling of empty sets (n = 0) and small-sample boundaries (n < 30).
  - Aggregate table download with provenance comment headers.

### 2. Hotspots (`app/pages/1_Hotspots.py`)
- **Nature**: Static workforce-level evaluation (filters decouple with explicit notification).
- **Capabilities**:
  - KPI-07: Hotspot candidate universe summary (104 candidate segments).
  - Interactive, sortable Tier 1 hotspot rankings with empirical Bayes shrunken indices.
  - Plotly Wilson interval comparison chart across candidate segments.
  - KPI-08: Split-sample replication benchmarks (25 screened, 14 with holdout n >= 30, 12 above baseline by point estimate, 7 above baseline by Wilson lower bound).
  - KPI-09: Greedy marginal coverage sequence (8 steps covering 87.3% of leavers).
  - Empirical Bayes prior sensitivity analysis (m = 10, 30, 100).
  - Overlap non-independence and regression-to-the-mean warnings.

### 3. Adjusted Associations (`app/pages/2_Adjusted_Associations.py`)
- **Nature**: Static workforce-level evaluation.
- **Capabilities**:
  - KPI-06: OverTime association comparisons across crude (OR 3.82), stratified Mantel-Haenszel (JobLevel OR 4.05, Department OR 3.81, TenureBand OR 3.85), multivariable logistic regression (Model M1 OR 3.99, Model M2 OR 3.73), and penalised logistic contrasts (OR 3.66).
  - Tarone Breslow-Day homogeneity p-values and Robbins-Breslow-Greenland confidence intervals.
  - Zero-exposure Mantel-Haenszel exclusion documentation.
  - Multivariable parameter coefficients, standard errors, odds ratios, and variance inflation factors (VIF < 2.5).
  - Benjamini-Hochberg false discovery rate audit table across 28 categorical dimensions (23 cleared at q < 0.05).

### 4. Model Triage (`app/pages/3_Model_Triage.py`)
- **Nature**: Static workforce-level evaluation.
- **Capabilities**:
  - KPI-10: Out-of-sample triage yield at top-10% (Recall 38.3%, Lift 3.88x) and top-20% (Recall 57.4%, Lift 2.86x) review thresholds on held-out test data (n = 294, leavers = 47).
  - Benchmark model comparison table (Dummy, Logistic Regression, HistGradientBoosting, Calibrated LR).
  - Paired bootstrap model difference evaluations proving overlapping confidence intervals.
  - Probability calibration bin diagnostics with small-cell (n < 10) stability flags.
  - Feature Set governance policy documentation (Set B selected, Set A excluded).
  - Demographic fairness and selection parity audit across Gender, MaritalStatus, and AgeGroup.
  - Hash-verified static PNG figures (FIG-15 through FIG-20).

### 5. Governance and Limitations (`app/pages/4_Governance.py`)
- **Nature**: Static methodology documentation.
- **Capabilities**:
  - Complete Decision Status Registry (DL-01 through DL-44) dynamically ingested from `docs/decision_status.md`.
  - Findings consistency replication audit summary (57 exploratory findings: 21 replicated, 18 direction consistent, 11 unstable, 7 excluded from Set B).
  - Dataset provenance declaration: synthetic/public HR data with unrecorded voluntariness, cross-sectional structure, and absence of survival windows.
  - Master KPI Registry definitions (KPI-01 through KPI-10).
  - Formal reliability classification policies (DL-41).

---

## 3. Filter Management and Session State

Filters are configured via `app/filters.py` and persisted in `st.session_state`:
1. **Operational Filters**:
   - `Department` (Sales, Research & Development, Human Resources)
   - `JobRole` (9 distinct organizational job roles)
   - `JobLevel` (Levels 1 through 5)
   - `OverTime` (Yes, No)
   - `TenureBand` (0-1, 2-3, 4-5, 6-10, 11+)
   - `WorkLifeBalance` (Bad, Good, Better, Best)
2. **Descriptive Context Filters (Segregated under DL-44)**:
   - `AgeGroup` (18-25, 26-35, 36-45, 46-60)
   - `Gender` (Female, Male)
   - `MaritalStatus` (Single, Married, Divorced)
3. **Decoupling on Static Pages**:
   - Static pages (Hotspots, Adjusted Associations, Model Triage, Governance) display an explicit notification informing the user that slice filters are not applied to fixed workforce benchmarks.

---

## 4. Execution and Verification

### Launching the Dashboard Locally

Activate the virtual environment and launch Streamlit:

```bash
.venv/Scripts/streamlit run app/Overview.py
```

### Running Automated Test Suites

The application includes automated unit, AST, component, data access, and end-to-end AppTest test suites:

```bash
.venv/Scripts/pytest tests/test_app_ast.py tests/test_app_components.py tests/test_app_data_access.py tests/test_app_pages.py -v
```

### Headless Server Health Check

Run the headless smoke verification script to test server startup, health endpoint response, and error log absence:

```bash
.venv/Scripts/python scripts/run_phase7_app_checks.py
```
