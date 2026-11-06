# Model Card: Workforce Intelligence Attrition Risk Model (Phase 4)

**Model Release Date**: November 2026  
**Model Version**: 1.0.0 (Phase 4 Production Candidate)  
**Governance Status**: Approved for Decision Support Only  
**Recommended Operational Pipeline**: L2-Regularised Logistic Regression with Platt/Sigmoid Calibration (Feature Set B: Demographic-Free)

---

## 1. Model Details

### 1.1 Overview and Architecture
- **Model Family**: Generalized Linear Model (L2 Regularised Binary Logistic Regression).
- **Optimization Objective**: Log-loss with $L_2$ ridge penalty, tuned over regularization strength $C \in [0.001, 10.0]$ with Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 folds). Optimal hyperparameter: $C = 0.10$.
- **Probability Calibration**: Post-hoc Sigmoid (Platt) scaling fitted via nested cross-validation (`CalibratedClassifierCV(cv=5, method='sigmoid')`) on the training split only.
- **Preprocessing Pipeline**:
  - Imputation: Constant zero-fill (`SimpleImputer(strategy='constant', fill_value=0.0)`) for ratio features with zero-tenure denominators (`RoleTenureShare`, `PromotionGapShare`).
  - Standardisation: Standard normal scaling (`StandardScaler`) applied to all continuous, count, and ordinal variables.
  - Categorical Encoding: Unsupervised One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`) for nominal features.
- **Target Variable**: Binary employee attrition (`0 = Retained`, `1 = Departed`).
- **Implementation Stack**: Python 3.14+, `scikit-learn==1.9.1`, `pandas`, `numpy`, `scipy`.

### 1.2 Training & Evaluation Framework
- **Data Source**: Phase 2 Cleaned Dataset (`data/processed/workforce_attrition_processed.csv`, $n = 1,470$).
- **Partitioning**: Deterministic stratified 80/20 train/holdout split (`SEED = 42`).
  - Training Set: $n = 1,176$ (190 departures, 16.16% baseline rate).
  - Out-of-Sample Holdout Set: $n = 294$ (47 departures, 15.99% baseline rate).
  - The holdout partition remained strictly untouched throughout feature selection and hyperparameter tuning, evaluated exactly once.

---

## 2. Intended Use and Operational Guidelines

### 2.1 Intended Operational Use Cases
- **Prioritised Triage for HR Business Partners & Line Managers**: Rank-order organizational cohorts and individual employee profiles to direct proactive retention check-ins, stay interviews, workload audits, and career development pathway discussions.
- **High-Risk Cohort Identification**: Detect organizational segments facing elevated systemic departure risk (e.g., intensive overtime, early role stagnation, compensation compression) to guide macro-level talent policies.
- **Decision Support Mode**: The model is strictly advisory. All risk scores and percentiles provide descriptive context to assist human professional judgment.

### 2.2 Prohibited and Out-of-Scope Use Cases
- **NO Automated Adverse Employment Action**: The model must **NEVER** be used to trigger, recommend, or justify automated termination, dismissal, demotion, salary reductions, bonus withholdings, or disciplinary actions.
- **NO Performance Appraisal Scoring**: Predicted attrition risk must not factor into employee performance ratings, competency assessments, or compensation adjustment algorithms.
- **NO Hiring or Pre-Employment Screening**: The model is calibrated specifically for internal workforce retention dynamics and cannot be applied to job applicant pools.

---

## 3. Quantitative Performance Factors & Metrics

### 3.1 Primary Metric: Precision-Recall AUC (PR-AUC)
Due to substantial target class imbalance (16.0% holdout departure prevalence), PR-AUC serves as the primary metric:
- **Baseline Prior (Chance)**: 0.1599
- **Cross-Validation Mean (Set B, 15 folds)**: $0.6304 \pm 0.0815$
- **Holdout Point Estimate (Calibrated LR)**: $0.5690$
- **Holdout 95% Percentile Bootstrap CI (2,000 resamples)**: $[0.4258, 0.6986]$

### 3.2 Comprehensive Holdout Performance Summary ($n = 294$, 47 Departures)

Both calibrated and uncalibrated performance are presented to reconcile triage yields:

| Model Specification | Triage Tier | Cutoff ($k$) | True Positives (TP) | False Positives (FP) | Precision (Yield) | Recall | Lift vs Base (15.99%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Logistic Regression** | Top 10% | 29 | 18 | 11 | **62.07%** | **38.30%** | **3.88x** |
| **Calibrated Logistic Regression** | Top 20% | 59 | 27 | 32 | **45.76%** | **57.45%** | **2.86x** |
| *Uncalibrated Logistic Regression* | Top 10% | 29 | 19 | 10 | 65.52% | 40.43% | 4.10x |
| *Uncalibrated Logistic Regression* | Top 20% | 59 | 27 | 32 | 45.76% | 57.45% | 2.86x |

#### Metric Bootstrap Summary (Calibrated Logistic Regression - Set B)
- **PR-AUC**: 0.5690 ($95\% \text{ CI } [0.4258, 0.6986]$)
- **ROC-AUC**: 0.8099 ($95\% \text{ CI } [0.7353, 0.8774]$)
- **Brier Score**: 0.1006 ($95\% \text{ CI } [0.0764, 0.1253]$)
- **Recall @ Top 10%**: 0.3830 ($95\% \text{ CI } [0.2923, 0.5250]$)
- **Lift @ Top 10%**: 3.8826 ($95\% \text{ CI } [2.9632, 5.3224]$)
- **Recall @ Top 20%**: 0.5745 ($95\% \text{ CI } [0.4375, 0.6977]$)
- **Lift @ Top 20%**: 2.8626 ($95\% \text{ CI } [2.1801, 3.4768]$)

### 3.3 Feature Set Selection & Materiality Governance
- **Feature Set A**: Full set including demographic attributes (`Age`, `Gender`, `MaritalStatus`, `AgeGroup`).
- **Feature Set B**: Demographic attributes excluded.
- **Materiality Rule**: Feature Set A is adopted only if its CV PR-AUC exceeds Set B by $> 0.0300$ and bootstrap CIs do not overlap. This rule was specified pre-flight in `docs/phase4_spec.md`.
- **Audit Outcome**: Set A achieved mean CV PR-AUC of 0.6555 vs Set B at 0.6304 (gain of $+0.0251 \le 0.0300$). Across 15 paired folds, mean difference was $+0.0251$ (SD $0.0263$). Under the pre-flight governance protocol, **Feature Set B was selected and locked as the operational model**.

---

## 4. Key Predictive Drivers (Odds Ratios)

Coefficients are estimated from L2-regularised logistic regression ($C=0.10$) fitted on the training split ($n=1,176$). Standard errors and 95% confidence intervals are derived asymptotically from the penalized Fisher information Hessian $H = X^T W X + (1/C) I$:

| Feature | Feature Type | Scale / Unit | Coefficient ($\beta$) | SE | Odds Ratio | 95% Asymptotic CI |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **OverTime (`Yes`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.6928 | 0.225 | **1.999** | $[1.284, 3.110]$ |
| **OverTime (`No`)** | Binary Indicator | Presence vs Absence (1 vs 0) | -0.6928 | 0.225 | **0.500** | $[0.322, 0.777]$ |
| **BusinessTravel (`Travel_Frequently`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.5460 | 0.222 | **1.726** | $[1.118, 2.666]$ |
| **JobRole (`Laboratory Technician`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.5197 | 0.220 | **1.682** | $[1.093, 2.586]$ |
| **NumCompaniesWorked** | Continuous | Per 1 SD (2.50 employers) | +0.4022 | 0.103 | **1.495** | $[1.222, 1.829]$ |
| **JobRole (`Sales Representative`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.3175 | 0.259 | **1.374** | $[0.827, 2.285]$ |
| **EducationField (`Technical Degree`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.2743 | 0.232 | **1.316** | $[0.835, 2.074]$ |
| **JobRole (`Research Scientist`)** | Binary Indicator | Presence vs Absence (1 vs 0) | -0.2954 | 0.228 | **0.744** | $[0.476, 1.163]$ |
| **Department (`Research & Development`)**| Binary Indicator | Presence vs Absence (1 vs 0) | -0.3129 | 0.237 | **0.731** | $[0.460, 1.163]$ |
| **PriorExperienceYears** | Continuous | Per 1 SD (5.26 years) | -0.3175 | 0.210 | **0.728** | $[0.483, 1.098]$ |
| **JobSatisfaction** | Ordinal (1-4) | Per 1 SD (1.10 rating units) | -0.3444 | 0.091 | **0.709** | $[0.593, 0.847]$ |
| **JobInvolvement** | Ordinal (1-4) | Per 1 SD (0.71 rating units) | -0.3458 | 0.089 | **0.708** | $[0.594, 0.843]$ |
| **EnvironmentSatisfaction** | Ordinal (1-4) | Per 1 SD (1.09 rating units) | -0.3951 | 0.091 | **0.674** | $[0.564, 0.806]$ |
| **BusinessTravel (`Non-Travel`)** | Binary Indicator | Presence vs Absence (1 vs 0) | -0.4447 | 0.250 | **0.641** | $[0.393, 1.045]$ |
| **StockOptionLevel** | Ordinal (0-3) | Per 1 SD (0.85 grant tiers) | -0.4497 | 0.102 | **0.638** | $[0.522, 0.779]$ |

---

## 5. Demographic Fairness & Disparate Impact Audit

Even though demographic features were excluded from model training (Set B), the resulting model risk scores were audited across protected groups on the holdout split ($n=294$) at the operational 20% triage threshold ($k=59$):

### 5.1 Subgroup Audit Summary ($n = 294$, Holdout Split)

| Attribute | Subgroup | Sample Size ($n$) | Leavers | Flagged ($k$) | Selection Rate (95% Wilson CI) | Recall (95% Wilson CI) | Precision (95% Wilson CI) | Disparate Impact Ratio | Low-$n$ Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gender** | Female | 116 | 16 | 26 | 22.41% [15.8%, 30.8%] | 62.50% [38.6%, 81.5%] | 38.46% [22.4%, 57.5%] | 1.0000 | False |
| **Gender** | Male | 178 | 31 | 33 | 18.54% [13.5%, 24.9%] | 54.84% [37.8%, 70.8%] | 51.52% [35.2%, 67.5%] | 0.8271 | False |
| **MaritalStatus** | Single | 97 | 25 | 32 | 32.99% [24.4%, 42.8%] | 64.00% [44.5%, 79.8%] | 50.00% [33.6%, 66.4%] | 1.0000 | False |
| **MaritalStatus** | Married | 133 | 18 | 20 | 15.04% [10.0%, 22.1%] | 50.00% [29.0%, 71.0%] | 45.00% [25.8%, 65.8%] | 0.4558 | False |
| **MaritalStatus** | Divorced | 64 | 4 | 7 | 10.94% [5.4%, 20.9%] | 50.00% [15.0%, 85.0%] | 28.57% [8.2%, 64.1%] | 0.3315 | False |
| **AgeGroup** | 18-25 | 27 | 7 | 16 | 59.26% [40.7%, 75.5%] | 85.71% [48.7%, 97.4%] | 37.50% [18.5%, 61.4%] | 1.0000 | **True ($n<30$)** |
| **AgeGroup** | 26-35 | 121 | 20 | 20 | 16.53% [11.0%, 24.2%] | 55.00% [34.2%, 74.2%] | 55.00% [34.2%, 74.2%] | 0.2789 | False |
| **AgeGroup** | 36-45 | 98 | 11 | 15 | 15.31% [9.5%, 23.7%] | 63.64% [35.4%, 84.8%] | 46.67% [24.8%, 69.9%] | 0.2583 | False |
| **AgeGroup** | 46-60 | 48 | 9 | 8 | 16.67% [8.7%, 29.6%] | 33.33% [12.1%, 64.6%] | 37.50% [13.7%, 69.4%] | 0.2812 | False |

### 5.2 Fairness Findings & Governance Controls
- **Disparate Impact Notice**: Because operational factors (overtime, entry-level job role, lower tenure) correlate in practice with younger age and single marital status, the demographic-free model selects younger and single employees at higher rates.
- **Four-Fifths Rule Context**: Under EEOC guidelines, a selection rate below 80% of the highest group constitutes disparate impact in selection/hiring. However, for a **proactive retention triage program**, higher selection rates translate to **greater retention outreach and support**, not adverse selection.
- **Fairness Recommendations**:
  1. HR teams must ensure that retention outreach is framed constructively (career development, wellness, compensation reviews).
  2. Line managers must not treat triage lists as "flight risk blacklists".
  3. Proactive retention reviews should be conducted across all age and tenure brackets, supplementing model rankings with periodic universal check-ins.

---

## 6. Caveats and Limitations

1. **Synthetic Data Provenance**: The dataset originates from public synthetic HR records (Palo Alto Networks pseudonym). Findings reflect structural simulation parameters, not authentic corporate records.
2. **Cross-Sectional Observational Nature**: All model relationships represent statistical correlations at a single snapshot. They do **NOT** prove causality. Interventions must be tested through pilot trials.
3. **Unobserved Exit Types**: Voluntary departures are not distinguished from involuntary terminations or retirements.
4. **Self-Fulfilling Feedback Loops**: If line managers act punitively based on risk scores (e.g., withholding high-impact projects from flagged individuals), the model can induce the very turnover it was meant to prevent.
