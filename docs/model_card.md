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

| Evaluation Metric | Holdout Point Estimate | 95% Bootstrap CI | Operational Interpretation |
| :--- | :---: | :---: | :--- |
| **PR-AUC (Average Precision)** | **0.5690** | $[0.4258, 0.6986]$ | High discriminative precision across operational triage thresholds. |
| **ROC-AUC** | **0.8099** | $[0.7353, 0.8774]$ | Strong ranking separation between leavers and stayers. |
| **Brier Score** | **0.1006** | $[0.0764, 0.1253]$ | Well-calibrated probability predictions (squared error). |
| **Recall @ Top 10% Triage** | **0.3830** | $[0.2923, 0.5250]$ | Captures 38.3% of all departures by reviewing only top 10% ($n = 29$). |
| **Lift @ Top 10% Triage** | **3.88** | $[2.96, 5.32]$ | Attrition rate in top 10% is 3.88x higher than workforce baseline. |
| **Recall @ Top 20% Triage** | **0.5745** | $[0.4375, 0.6977]$ | Captures 57.5% of all departures by reviewing top 20% ($n = 59$). |
| **Lift @ Top 20% Triage** | **2.86** | $[2.18, 3.48]$ | Attrition rate in top 20% is 2.86x higher than workforce baseline. |

### 3.3 Feature Set Selection & Materiality Governance
- **Feature Set A**: Full set including demographic attributes (`Age`, `Gender`, `MaritalStatus`, `AgeGroup`).
- **Feature Set B**: Demographic attributes excluded.
- **Materiality Rule**: Feature Set A is adopted only if its CV PR-AUC exceeds Set B by $> 0.0300$ and bootstrap CIs do not overlap.
- **Audit Outcome**: Set A achieved CV PR-AUC of 0.6550 vs Set B at 0.6304 (gain of $+0.0246 \le 0.0300$). Under the governance protocol, **Feature Set B was selected and locked as the operational model**.

---

## 4. Key Predictive Drivers (Odds Ratios)

Top risk and protective drivers from the locked regularised logistic regression model (Set B):
- **OverTime (`Yes`)**: Odds Ratio $\approx 2.00$ ($95\% \text{ CI } [1.46, 2.74]$) — single largest operational risk factor.
- **BusinessTravel (`Travel_Frequently`)**: Odds Ratio $\approx 1.73$ ($95\% \text{ CI } [1.26, 2.37]$).
- **JobRole (`Laboratory Technician`)**: Odds Ratio $\approx 1.68$ ($95\% \text{ CI } [1.23, 2.30]$).
- **StockOptionLevel**: Odds Ratio $\approx 0.64$ ($95\% \text{ CI } [0.47, 0.88]$) — strong protective retention factor.
- **EnvironmentSatisfaction & JobInvolvement**: Odds Ratio $\approx 0.67 - 0.71$ — strong protective retention factors.

---

## 5. Demographic Fairness & Disparate Impact Audit

Even though demographic features were excluded from model training (Set B), the resulting model risk scores were audited across protected groups on the holdout split at the operational 20% triage threshold:

### 5.1 Subgroup Audit Summary ($n = 294$)

| Attribute | Subgroup | Sample Size | Leavers | Selected Count | Selection Rate | Leaver Recall | Disparate Impact Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gender** | Female | 114 | 16 | 27 | 23.7% | 62.5% | 1.00 (Reference) |
| **Gender** | Male | 180 | 31 | 32 | 17.8% | 54.8% | **0.75** |
| **MaritalStatus** | Single | 96 | 25 | 32 | 33.3% | 64.0% | 1.00 (Reference) |
| **MaritalStatus** | Married | 134 | 16 | 20 | 14.9% | 50.0% | **0.45** |
| **MaritalStatus** | Divorced | 64 | 6 | 7 | 10.9% | 50.0% | **0.33** |
| **AgeGroup** | 18-24 | 20 | 9 | 9 | 45.0% | 66.7% | 1.00 (Reference) |
| **AgeGroup** | 25-34 | 126 | 24 | 33 | 26.2% | 62.5% | 0.58 |
| **AgeGroup** | 35-44 | 93 | 9 | 12 | 12.9% | 44.4% | 0.29 |
| **AgeGroup** | 45-54 | 45 | 4 | 5 | 11.1% | 25.0% | 0.25 |
| **AgeGroup** | 55+ | 10 | 1 | 0 | 0.0% | 0.0% | 0.00 |

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
