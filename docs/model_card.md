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

Both calibrated and uncalibrated performance are presented to evaluate ranking and probability fidelity:

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

*Calibration Evaluation Note*: Applying 5-fold Sigmoid (Platt) calibration shifted the probability distribution spread but did not improve the Brier score (0.0995 uncalibrated vs 0.1006 calibrated) and made no measurable difference to ranking or true positive capture at the operational 20% triage cutoff (TP = 27 under both).
- **10-Bin Evaluation**: 6 of 10 bins in the uncalibrated model (bins 5, 6, 7, 8, 9, 10 with counts 8, 9, 9, 2, 2, 1) and 5 of 10 bins in the calibrated model (bins 6, 7, 8, 9, 10 with counts 8, 8, 0, 2, 1) contain fewer than 10 holdout cases.
- **5-Bin Evaluation**: 1 of 5 bins in the uncalibrated model (bin 5 has $n=3$) and 2 of 5 bins in the calibrated model (bin 4 has $n=8$, bin 5 has $n=3$) contain fewer than 10 holdout cases.
Claims of fine-grained probability calibration are unsupported by holdout sample sizes.

### 3.3 Feature Set Selection & Materiality Governance
- **Feature Set A**: Full set including demographic attributes (`Age`, `Gender`, `MaritalStatus`, `AgeGroup`).
- **Feature Set B**: Demographic attributes excluded.
- **Materiality Threshold & Timing Audit**: Feature Set A was specified to be adopted only if its CV PR-AUC exceeded Set B by $> 0.0300$ with non-overlapping bootstrap CIs. However, the 0.03 threshold first appeared in commit `111db00` alongside the initial modeling code and results, meaning its pre-flight timing cannot be independently verified from git history.
- **Audit Outcome**: In 15-fold cross-validation on the training set, Set A achieved mean CV PR-AUC of 0.6555 vs Set B at 0.6304 (gain of $+0.0251 \le 0.0300$, SD 0.0263). Set B was selected and locked on primary governance and ethical grounds (preventing disparate treatment). On the holdout split, Set A PR-AUC (0.5960, 95% CI: [0.4503, 0.7185]) and Set B PR-AUC (0.5736, 95% CI: [0.4315, 0.7037]) exhibit overlapping 95% bootstrap confidence intervals, and the paired holdout difference (+0.0224, 95% CI: [-0.0182, +0.0684]) spans zero, confirming that demographic inclusion yields no statistically distinguishable out-of-sample benefit.

### 3.4 Paired Model Differences Table

| Comparison | Metric | Point Diff | Bootstrap Mean Diff | 95% Bootstrap CI | Zero Included? | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Holdout: LR (Uncal) - HGB** | PR-AUC | **+0.0652** | +0.0638 | $[-0.0344, +0.1647]$ | Yes | Point diff = 0.5736 - 0.5084 |
| **Holdout: LR (Uncal) - HGB** | ROC-AUC | **+0.0388** | +0.0386 | $[-0.0112, +0.0911]$ | Yes | Point diff = 0.8130 - 0.7742 |
| **Holdout: LR (Uncal) - HGB** | Brier Score | **-0.0115** | -0.0115 | $[-0.0261, +0.0020]$ | Yes | Point diff = 0.0995 - 0.1110 |
| **Holdout: LR (Set A - Set B)** | PR-AUC | **+0.0224** | +0.0220 | $[-0.0182, +0.0684]$ | Yes | Set A (0.5960) vs Set B (0.5736) |
| **Holdout: LR (Set A - Set B)** | ROC-AUC | **+0.0097** | +0.0096 | $[-0.0115, +0.0315]$ | Yes | Set A (0.8227) vs Set B (0.8130) |
| **Holdout: LR (Set A - Set B)** | Brier Score | **-0.0031** | -0.0031 | $[-0.0098, +0.0034]$ | Yes | Set A (0.0964) vs Set B (0.0995) |
| **15-Fold CV: LR (Set A - Set B)** | PR-AUC | **+0.0251** | +0.0251 | $[-0.0076, +0.0578]$ | Yes | Nadeau-Bengio interval; fold spread: $[-0.0265, +0.0766]$ (SD=0.0263, not a CI) |

---

## 4. Key Predictive Drivers (Odds Ratios & Nominal Contrasts)

Coefficients are estimated from L2-regularised logistic regression ($C=0.10$) fitted on the training split ($n=1,176$). Standard errors and confidence intervals are computed both asymptotically from the penalized Fisher information Hessian $H = X^T W X + (1/C) I$ (approximate because the shrinkage penalty bias is ignored) and via 500 bootstrap refits.

> [!IMPORTANT]
> **Interpretation of Nominal Coefficients**: In the symmetric one-hot encoding used with L2 regularization, categorical coefficients represent deviations from the penalized grand mean across levels, NOT relative to an omitted reference level. Therefore, assessing the relative effect between categories requires evaluating pairwise contrast differences ($\beta_A - \beta_B$). Furthermore, features cannot be directly ranked by $|\beta|$ across mixed scales (standardized continuous per-1-SD vs discrete binary indicators).

### 4.1 Key Nominal Contrasts

| Dimension | Nominal Contrast | Coefficient Difference ($\Delta \beta$) | Contrast Odds Ratio | 95% Bootstrap CI | Asymptotic 95% CI | Operational Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **OverTime** | **Yes vs No** | **+1.3855** | **4.00** | **$[2.97, 5.51]$** | $[2.80, 5.71]$ | OverTime quadruples departure odds relative to non-overtime employees. |
| **BusinessTravel** | Travel_Frequently vs Non-Travel | +0.9911 | 2.69 | $[1.89, 3.70]$ | $[1.53, 4.75]$ | Frequent travel increases departure odds by 2.69x over non-travelers. |
| **BusinessTravel** | Travel_Frequently vs Travel_Rarely | +0.6476 | 1.91 | $[1.39, 2.68]$ | $[1.29, 2.84]$ | Frequent travel nearly doubles odds over rare travelers. |
| **BusinessTravel** | Travel_Rarely vs Non-Travel | +0.3436 | 1.41 | $[1.00, 1.98]$ | $[0.83, 2.38]$ | Rare travel carries modest elevation over non-travelers. |

### 4.2 Standardized Predictor Odds Ratios

| Feature | Feature Type | Scale / Unit | Coefficient ($\beta$) | SE (Asymp) | Odds Ratio | 95% Bootstrap CI | 95% Asymptotic CI |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **OverTime (`Yes`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.6925 | 0.226 | **2.00** | $[1.67, 2.36]$ | $[1.28, 3.11]$ |
| **OverTime (`No`)** | Binary Indicator | Presence vs Absence (1 vs 0) | -0.6930 | 0.225 | **0.50** | $[0.42, 0.60]$ | $[0.32, 0.78]$ |
| **BusinessTravel (`Travel_Frequently`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.5460 | 0.222 | **1.73** | $[1.38, 2.08]$ | $[1.12, 2.67]$ |
| **JobRole (`Laboratory Technician`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.5197 | 0.220 | **1.68** | $[1.35, 2.05]$ | $[1.09, 2.59]$ |
| **NumCompaniesWorked** | Continuous | Per 1 SD (2.49 employers) | +0.4019 | 0.103 | **1.49** | $[1.24, 1.77]$ | $[1.22, 1.83]$ |
| **JobRole (`Sales Representative`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.3180 | 0.259 | **1.37** | $[1.03, 1.82]$ | $[0.83, 2.29]$ |
| **EducationField (`Technical Degree`)** | Binary Indicator | Presence vs Absence (1 vs 0) | +0.2744 | 0.232 | **1.32** | $[1.03, 1.63]$ | $[0.83, 2.07]$ |
| **JobRole (`Research Scientist`)** | Binary Indicator | Presence vs Absence (1 vs 0) | -0.2953 | 0.228 | **0.74** | $[0.59, 0.94]$ | $[0.48, 1.16]$ |
| **Department (`Research & Development`)**| Binary Indicator | Presence vs Absence (1 vs 0) | -0.3128 | 0.237 | **0.73** | $[0.58, 0.92]$ | $[0.46, 1.16]$ |
| **PriorExperienceYears** | Continuous | Per 1 SD (6.26 years) | -0.3177 | 0.210 | **0.73** | $[0.51, 1.05]$ | $[0.48, 1.10]$ |
| **JobSatisfaction** | Ordinal (1-4) | Per 1 SD (1.11 rating units) | -0.3441 | 0.091 | **0.71** | $[0.60, 0.83]$ | $[0.59, 0.85]$ |
| **JobInvolvement** | Ordinal (1-4) | Per 1 SD (0.70 rating units) | -0.3455 | 0.089 | **0.71** | $[0.60, 0.82]$ | $[0.59, 0.84]$ |
| **EnvironmentSatisfaction** | Ordinal (1-4) | Per 1 SD (1.09 rating units) | -0.3944 | 0.091 | **0.67** | $[0.57, 0.80]$ | $[0.56, 0.81]$ |
| **BusinessTravel (`Non-Travel`)** | Binary Indicator | Presence vs Absence (1 vs 0) | -0.4451 | 0.250 | **0.64** | $[0.48, 0.86]$ | $[0.39, 1.05]$ |
| **StockOptionLevel** | Ordinal (0-3) | Per 1 SD (0.85 grant tiers) | -0.4497 | 0.102 | **0.64** | $[0.53, 0.77]$ | $[0.52, 0.78]$ |

### 4.3 Out-of-Sample Risk Hotspot Generalisation Audit

Risk hotspots are evaluated by strictly separating train-screened candidates from optimistic candidates:

| Candidate Group | Screened ($n$) | Holdout $n \ge 30$ | Above Baseline Point Estimate ($n \ge 30$) | Above Baseline CI Lower Bound ($n \ge 30$) | Headline Generalisation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train-Screened Candidates** | **24** | **13** | **11** | **8** | **8 / 13 (61.5%)** |
| *Optimistic Candidates (Full Data)* | 3 | 1 | 1 | 1 | 1 / 1 (100.0%) |
| **Total Candidates Evaluated** | 27 | 14 | 12 | 9 | 9 / 14 (64.3%) |

> [!NOTE]
> **Screening Rule Consistency & Governance**: In code, candidate screening strictly excluded protected demographic attributes `AgeGroup` and `Gender` (and interactions with age). However, `MaritalStatus=Single` was retained in the candidate pool because it was an established bivariate finding from Phase 3, even though `MaritalStatus` is designated as a protected demographic in `schema.json`. This inconsistency is documented here: `MaritalStatus=Single` generalises out-of-sample (holdout rate 25.8% [18.1%, 35.3%] vs 16.0% baseline), but its operational use as an intervention target is prohibited under governance rules.
> Optimistic rows (`OverTime=Yes & AgeGroup=18-25`, `OverTime=Yes & AgeGroup=26-35`, `OverTime=Yes & TenureBand=4-5`) used holdout data during Phase 3 selection and are strictly excluded from headline train-screened generalisation metrics.

---

## 5. Demographic Fairness & Disparate Impact Audit

Even though demographic features were excluded from model training (Set B), the resulting model risk scores were audited across protected groups on the holdout split ($n=294$) at the operational 20% triage threshold ($k=59$):

### 5.1 Subgroup Audit Summary ($n = 294$, Holdout Split)

| Attribute | Subgroup | Sample Size ($n$) | Leavers | Flagged ($k$) | Selection Rate (95% Wilson CI) | Ratio vs Highest | Ratio vs Largest | Recall (95% Wilson CI) | Precision (95% Wilson CI) | Observed Rate (95% Wilson CI) | Low-$n$ Flags ($n$ / leav / flag) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gender** | Female | 116 | 16 | 26 | 22.41% [15.8%, 30.8%] | 1.0000 | 1.2087 | 62.50% [38.6%, 81.5%] | 38.46% [22.4%, 57.5%] | 13.79% [8.7%, 21.1%] | False / **True** / **True** |
| **Gender** | Male | 178 | 31 | 33 | 18.54% [13.5%, 24.9%] | 0.8271 | 1.0000 | 54.84% [37.8%, 70.8%] | 51.52% [35.2%, 67.5%] | 17.42% [12.4%, 23.8%] | False / False / False |
| **MaritalStatus** | Single | 97 | 25 | 32 | 32.99% [24.4%, 42.8%] | 1.0000 | 2.1935 | 64.00% [44.5%, 79.8%] | 50.00% [33.6%, 66.4%] | 25.77% [18.1%, 35.3%] | False / **True** / False |
| **MaritalStatus** | Married | 133 | 18 | 20 | 15.04% [10.0%, 22.1%] | 0.4558 | 1.0000 | 50.00% [29.0%, 71.0%] | 45.00% [25.8%, 65.8%] | 13.53% [8.7%, 20.4%] | False / **True** / **True** |
| **MaritalStatus** | Divorced | 64 | 4 | 7 | 10.94% [5.4%, 20.9%] | 0.3315 | 0.7274 | 50.00% [15.0%, 85.0%] | 28.57% [8.2%, 64.1%] | 6.25% [2.5%, 15.0%] | False / **True** / **True** |
| **AgeGroup** | 18-25 | 27 | 7 | 16 | 59.26% [40.7%, 75.5%] | 1.0000 | 3.5850 | 85.71% [48.7%, 97.4%] | 37.50% [18.5%, 61.4%] | 25.93% [13.2%, 44.7%] | **True** / **True** / **True** |
| **AgeGroup** | 26-35 | 121 | 20 | 20 | 16.53% [11.0%, 24.2%] | 0.2789 | 1.0000 | 55.00% [34.2%, 74.2%] | 55.00% [34.2%, 74.2%] | 16.53% [11.0%, 24.2%] | False / **True** / **True** |
| **AgeGroup** | 36-45 | 98 | 11 | 15 | 15.31% [9.5%, 23.7%] | 0.2583 | 0.9262 | 63.64% [35.4%, 84.8%] | 46.67% [24.8%, 69.9%] | 11.22% [6.4%, 18.9%] | False / **True** / **True** |
| **AgeGroup** | 46-60 | 48 | 9 | 8 | 16.67% [8.7%, 29.6%] | 0.2812 | 1.0085 | 33.33% [12.1%, 64.6%] | 37.50% [13.7%, 69.4%] | 18.75% [10.2%, 31.9%] | False / **True** / **True** |

### 5.2 Fairness Findings & Governance Controls
Selection-rate ratios differ markedly across demographic groups: for MaritalStatus, Single employees are selected at 2.19x the rate of Married employees (the largest group), and Divorced employees at 0.73x; for AgeGroup, employees aged 18-25 are selected at 3.58x the rate of employees aged 26-35 (the largest group). Observed historical attrition rates also differ substantially across these same groups: Single employees exhibit an observed attrition rate of 25.77% [18.1%, 35.3%] versus 13.53% [8.7%, 20.4%] for Married employees, and employees aged 18-25 exhibit 25.93% [13.2%, 44.7%] versus 11.22% [6.4%, 18.9%] for employees aged 36-45. 

While Feature Set B intentionally excludes protected demographic attributes to prevent explicit disparate treatment, the multivariable model may use correlated features (such as job level, compensation levels, and tenure) that act as empirical proxies. For example, `MonthlyIncome` is present in Set B ($|\beta| = 0.1052$ per SD) and is strongly associated with Age ($r = +0.498, p < 10^{-15}$) and MaritalStatus (Single mean $\$5,889$ vs Married mean $\$6,794$). Similarly, `JobLevel=1` and `StockOptionLevel=0` correlate with younger single workers. These operational features act as partial statistical proxies for demographic groups. These disparities are actively monitored and governed through recurring audit reviews, rather than assumed to be resolved simply by omitting demographic variables. No legal conclusions regarding employment compliance or statutory standards are asserted.
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
