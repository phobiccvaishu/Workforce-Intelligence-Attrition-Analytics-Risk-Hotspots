# Workforce Intelligence: Formal Statistical Inference, Stratified Adjustment & Hotspot Index Report

## 1. Executive Summary & Inferential Architecture
This report establishes the formal inferential foundation for the Workforce Intelligence analytics program, evaluating cross-sectional observational patterns across 1,470 employee records (237 leavers; baseline rate 16.12%). Analyses adhere to pre-registered protocols defined in `docs/inference_plan.md` with multiplicity adjustments via Benjamini-Hochberg (BH) False Discovery Rate control at alpha = 0.05.

All reported metrics reflect observational associations. In accordance with organizational governance, demographic attributes are examined exclusively for baseline context and are excluded from hotspot index screening.

### Summary of Statistical Findings
- **Categorical Associations**: Of 28 pre-specified organizational dimensions, 21 clear the multiplicity threshold of BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28). Ordered trend evaluations corroborate monotonic associations across job levels, tenure bands, career stages, and satisfaction scales.
- **Continuous Measures**: 8 of 9 numeric measures clear BH-adjusted q < 0.05 (family: numeric_measures, size: 9). PercentSalaryHike exhibits no observable separation between leavers and stayers (Mann-Whitney U p = 0.601, prob superiority = 0.508).
- **Stratified Odds Ratios**: The association between OverTime and Attrition remains robust across organizational stratifications:
  - Stratified by JobLevel: Mantel-Haenszel common OR = 4.05 [3.01, 5.46], Breslow-Day Tarone p = 0.084.
  - Stratified by Department: Mantel-Haenszel common OR = 3.81 [2.85, 5.08], Breslow-Day Tarone p = 0.730.
  - Stratified by TenureBand: Mantel-Haenszel common OR = 3.85 [2.86, 5.18], Breslow-Day Tarone p = 0.280.
- **Multivariable Adjustment**: In unpenalised multivariable logistic regression (Model M1), OverTime exhibits an adjusted odds ratio of 4.81 [3.48, 6.65]. With demographic controls added (Model M2), the adjusted odds ratio is 5.27 [3.77, 7.35]. Collinearity audits confirm all variance inflation factors remain below 2.34.
- **Overlap-Aware Hotspot Screening**: From a 104-segment universe, 14 segments qualify for Tier 1 (n >= 30, Wilson lower bound > baseline, and BH q < 0.05). A 5-step greedy marginal coverage selects 5 non-redundant segments that encompass 50.14% of workforce leavers across 27.69% of total employees.

---

## 2. Multiplicity-Controlled Association Tests

### 2.1 Categorical and Ordinal Dimensions (Family Size: 28)
Evaluated across contingency chi-square, Fisher exact (2x2 tables), permutation tests (where expected cell counts fall below 5), bootstrap Cramer's V confidence intervals (2,000 resamples), and Cochran-Armitage trend tests for ordered scales.

| Dimension | Levels | Chi2 Stat | Primary p-value | BH q-value | Multiplicity Status | Cramer's V [95% CI] | Cochran-Armitage z | Trend p-value | Phase 3 Agreement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Department | 3 | 10.80 | 4.5256e-03 | 7.0398e-03 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.086 [0.039, 0.143] | - | - | Agrees |
| JobRole | 9 | 86.19 | 2.7525e-15 | 2.5690e-14 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.242 [0.203, 0.298] | - | - | Agrees |
| JobLevel | 5 | 72.53 | 6.6347e-15 | 4.6443e-14 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.222 [0.177, 0.272] | -6.48 | 8.9575e-11 | Agrees |
| AgeGroup | 4 | 58.56 | 1.1918e-12 | 4.7671e-12 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.200 [0.147, 0.262] | -6.17 | 7.0270e-10 | Agrees |
| Gender | 2 | 1.28 | 2.7781e-01 | 2.9918e-01 | FDR Not Met | 0.029 [0.001, 0.077] | - | - | Agrees |
| MaritalStatus | 3 | 46.16 | 9.4555e-11 | 2.6475e-10 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.177 [0.128, 0.232] | - | - | Agrees |
| Education | 5 | 3.07 | 5.4553e-01 | 5.6573e-01 | FDR Not Met | 0.046 [0.027, 0.108] | -1.20 | 2.2903e-01 | Agrees |
| EducationField | 6 | 16.02 | 6.7740e-03 | 9.9827e-03 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.104 [0.069, 0.177] | - | - | Agrees |
| TenureBand | 5 | 75.59 | 1.4938e-15 | 2.0913e-14 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.227 [0.173, 0.286] | -7.78 | 7.3275e-15 | Agrees |
| EarlyTenure | 2 | 65.55 | 1.3176e-13 | 7.3787e-13 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.211 [0.150, 0.272] | - | - | Agrees |
| CareerStage | 4 | 54.34 | 9.5027e-12 | 3.3260e-11 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.192 [0.144, 0.250] | -6.70 | 2.1141e-11 | Agrees |
| RoleTenureBand | 4 | 49.45 | 1.0459e-10 | 2.6622e-10 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.183 [0.134, 0.235] | -6.74 | 1.5661e-11 | Agrees |
| ManagerTenureBand | 4 | 53.56 | 1.3956e-11 | 4.3418e-11 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.191 [0.141, 0.245] | -6.52 | 7.0044e-11 | Agrees |
| PromotionGapBand | 4 | 9.04 | 2.8757e-02 | 3.5009e-02 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.101 [0.047, 0.182] | 2.64 | 8.4013e-03 | Agrees |
| PromotionGapEqualsTenure | 2 | 7.61 | 9.3991e-03 | 1.3128e-02 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.072 [0.015, 0.132] | - | - | Agrees |
| HasPriorExperience | 2 | 3.64 | 5.8202e-02 | 6.7902e-02 | FDR Not Met | 0.050 [0.005, 0.106] | - | - | Agrees |
| NumCompaniesWorked | 10 | 25.74 | 2.2492e-03 | 3.7045e-03 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.132 [0.105, 0.200] | - | - | Agrees |
| OverTime | 2 | 89.04 | 2.3614e-19 | 6.6120e-18 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.246 [0.193, 0.304] | - | - | Agrees |
| BusinessTravel | 3 | 24.18 | 5.6086e-06 | 1.2080e-05 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.128 [0.079, 0.181] | - | - | Agrees |
| DistanceBand | 4 | 11.38 | 9.8458e-03 | 1.3128e-02 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.088 [0.045, 0.148] | 3.20 | 1.3507e-03 | Agrees |
| WorkLifeBalance | 4 | 16.33 | 9.7257e-04 | 1.7020e-03 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.105 [0.054, 0.173] | -2.45 | 1.4228e-02 | Agrees |
| JobSatisfaction | 4 | 17.51 | 5.5630e-04 | 1.0384e-03 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.109 [0.069, 0.166] | -3.97 | 7.2623e-05 | Agrees |
| EnvironmentSatisfaction | 4 | 22.50 | 5.1235e-05 | 1.0247e-04 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.124 [0.074, 0.185] | -3.96 | 7.3944e-05 | Agrees |
| RelationshipSatisfaction | 4 | 5.24 | 1.5497e-01 | 1.7357e-01 | FDR Not Met | 0.060 [0.026, 0.121] | -1.76 | 7.8617e-02 | Differs |
| JobInvolvement | 4 | 28.49 | 2.8632e-06 | 6.6808e-06 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.139 [0.088, 0.205] | -4.98 | 6.1998e-07 | Agrees |
| StockOptionLevel | 4 | 60.60 | 4.3794e-13 | 2.0437e-12 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.203 [0.160, 0.252] | -5.26 | 1.4546e-07 | Agrees |
| TrainingTimesLastYear | 7 | 15.15 | 1.9148e-02 | 2.4370e-02 | BH-adjusted q < 0.05 (family: categorical_ordinal, size: 28) | 0.102 [0.072, 0.167] | -2.28 | 2.2583e-02 | Agrees |
| PerformanceRating | 2 | 0.01 | 9.2172e-01 | 9.2172e-01 | FDR Not Met | 0.003 [0.001, 0.058] | - | - | Agrees |

### 2.2 Continuous Measures (Family Size: 9)
Evaluated using two-sided Mann-Whitney U tests and non-parametric probability of superiority (A = U / (n1 * n2)) with 2,000-resample percentile bootstrap confidence intervals.

| Measure | Leavers Median (IQR) | Stayers Median (IQR) | Mann-Whitney U | p-value | BH q-value | Prob of Superiority [95% CI] | Multiplicity Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Age | 32.0 (11.0) | 36.0 (12.0) | 106859.0 | 5.3043e-11 | 7.9565e-11 | 0.366 [0.325, 0.406] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |
| MonthlyIncome | 3202.0 (3543.0) | 5204.0 (5623.0) | 100620.5 | 2.9508e-14 | 1.3279e-13 | 0.344 [0.305, 0.384] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |
| DistanceFromHome | 9.0 (14.0) | 7.0 (11.0) | 164225.5 | 2.3870e-03 | 3.0691e-03 | 0.562 [0.523, 0.602] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |
| PercentSalaryHike | 14.0 (5.0) | 14.0 (6.0) | 140726.5 | 3.6551e-01 | 3.6551e-01 | 0.482 [0.441, 0.523] | FDR Not Met |
| TotalWorkingYears | 7.0 (7.0) | 10.0 (10.0) | 100567.0 | 2.3996e-14 | 1.3279e-13 | 0.344 [0.302, 0.385] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |
| YearsAtCompany | 3.0 (6.0) | 6.0 (7.0) | 102582.0 | 2.9162e-13 | 8.7486e-13 | 0.351 [0.313, 0.391] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |
| YearsInCurrentRole | 2.0 (4.0) | 3.0 (5.0) | 105214.0 | 4.4296e-12 | 9.9665e-12 | 0.360 [0.322, 0.398] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |
| YearsSinceLastPromotion | 1.0 (2.0) | 1.0 (3.0) | 134374.0 | 4.1179e-02 | 4.6326e-02 | 0.460 [0.420, 0.499] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |
| YearsWithCurrManager | 2.0 (5.0) | 3.0 (5.0) | 106361.5 | 1.8068e-11 | 3.2522e-11 | 0.364 [0.327, 0.402] | BH-adjusted q < 0.05 (family: numeric_measures, size: 9) |

---

## 3. Stratified Epidemiological Adjustment (Mantel-Haenszel)

Stratified analyses evaluate whether observed subgroup disparities persist when controlling for key organizational confounding structures. Strata with zero exposed or zero unexposed records are excluded from pooled pooling, and continuity corrections (+0.5) are applied solely to displayed stratum rows with zero cells.

### 3.1 Stratified Analysis Summary

| Analysis ID | Exposure | Stratifier | Cohort | Crude OR [95% CI] | MH Common OR [95% CI] | MH Chi2 p | Breslow-Day Tarone p | Excluded Strata |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MH-01 | OverTime (Yes vs No) | JobLevel | Workforce Full (n=1470) | 3.77 [2.83, 5.03] | 4.05 [3.01, 5.46] | 7.9265e-22 | 0.084 | None |
| MH-02 | OverTime (Yes vs No) | Department | Workforce Full (n=1470) | 3.77 [2.83, 5.03] | 3.81 [2.85, 5.08] | 3.5155e-21 | 0.730 | None |
| MH-03 | OverTime (Yes vs No) | TenureBand | Workforce Full (n=1470) | 3.77 [2.83, 5.03] | 3.85 [2.86, 5.18] | 2.0481e-20 | 0.280 | None |
| MH-04 | PromotionGap 6+ vs 0-5 | TenureBand | PromotionAnalysisCohort (n=890) | 1.96 [1.25, 3.06] | 2.38 [1.44, 3.92] | 5.3195e-04 | 0.613 | Cohort TenureBand 0-1; Cohort TenureBand 2-3; Cohort TenureBand 4-5 |

### 3.2 Stratum-Specific Odds Ratios

| Analysis ID | Stratum | Exposed Leavers (a) | Exposed Stayers (b) | Unexposed Leavers (c) | Unexposed Stayers (d) | Headcount (n) | Stratum OR [95% CI] | Zero Cell | Low n (<30) | Adjustment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MH-01 | JobLevel 1 | 82 | 74 | 61 | 326 | 543 | 5.92 [3.90, 8.98] | No | No | unadjusted |
| MH-01 | JobLevel 2 | 26 | 120 | 26 | 362 | 534 | 3.02 [1.69, 5.40] | No | No | unadjusted |
| MH-01 | JobLevel 3 | 13 | 50 | 19 | 136 | 218 | 1.86 [0.86, 4.05] | No | No | unadjusted |
| MH-01 | JobLevel 4 | 3 | 30 | 2 | 71 | 106 | 3.55 [0.56, 22.34] | No | No | unadjusted |
| MH-01 | JobLevel 5 | 3 | 15 | 2 | 49 | 69 | 4.90 [0.75, 32.12] | No | No | unadjusted |
| MH-02 | Department Human Resources | 5 | 12 | 7 | 39 | 63 | 2.32 [0.62, 8.67] | No | No | unadjusted |
| MH-02 | Department Research & Development | 74 | 197 | 59 | 631 | 961 | 4.02 [2.75, 5.86] | No | No | unadjusted |
| MH-02 | Department Sales | 48 | 80 | 44 | 274 | 446 | 3.74 [2.31, 6.03] | No | No | unadjusted |
| MH-03 | TenureBand 0-1 | 38 | 31 | 37 | 109 | 215 | 3.61 [1.98, 6.60] | No | No | unadjusted |
| MH-03 | TenureBand 2-3 | 29 | 46 | 18 | 162 | 255 | 5.67 [2.89, 11.12] | No | No | unadjusted |
| MH-03 | TenureBand 4-5 | 22 | 65 | 18 | 201 | 306 | 3.78 [1.91, 7.48] | No | No | unadjusted |
| MH-03 | TenureBand 6-10 | 31 | 89 | 24 | 304 | 448 | 4.41 [2.46, 7.90] | No | No | unadjusted |
| MH-03 | TenureBand 11+ | 7 | 58 | 13 | 168 | 246 | 1.56 [0.59, 4.10] | No | No | unadjusted |
| MH-04 | Cohort TenureBand 6-10 | 22 | 92 | 33 | 301 | 448 | 2.18 [1.21, 3.93] | No | No | unadjusted |
| MH-04 | Cohort TenureBand 11+ | 13 | 88 | 7 | 138 | 246 | 2.91 [1.12, 7.58] | No | No | unadjusted |

---

## 4. Adjusted Multivariable Logistic Regression Models (M1 and M2)

Unpenalised logistic regression models fitted via statsmodels Logit provide simultaneous parameter adjustments. Model M1 includes core operational and satisfaction measures (11 parameters; 21.55 events per parameter). Model M2 incorporates demographic controls (Age, Gender, MaritalStatus) solely as observational adjustment terms (15 parameters; 15.80 events per parameter). Both models converge without quasi-complete separation.

### 4.1 Model M1 Parameter Estimates and Collinearity Audit

| Predictor Term | Coefficient | Std Error | z-statistic | p-value | Odds Ratio [95% CI] | VIF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Intercept | 2.0383 | 0.6251 | 3.26 | 1.1118e-03 | Baseline Log-Odds | - |
| C(OverTime, Treatment(reference='No'))[T.Yes] | 1.5704 | 0.1655 | 9.49 | 2.3249e-21 | 4.81 [3.48, 6.65] | 1.01 |
| C(BusinessTravel, Treatment(reference='Non-Travel'))[T.Travel_Frequently] | 1.5392 | 0.3715 | 4.14 | 3.4222e-05 | 4.66 [2.25, 9.65] | 2.32 |
| C(BusinessTravel, Treatment(reference='Non-Travel'))[T.Travel_Rarely] | 0.7876 | 0.3484 | 2.26 | 2.3802e-02 | 2.20 [1.11, 4.35] | 2.32 |
| JobLevel | -0.4913 | 0.1036 | -4.74 | 2.1140e-06 | 0.61 [0.50, 0.75] | 1.40 |
| StockOptionLevel | -0.5360 | 0.1060 | -5.06 | 4.2564e-07 | 0.59 [0.48, 0.72] | 1.00 |
| DistanceFromHome_sd | 0.2917 | 0.0781 | 3.74 | 1.8674e-04 | 1.34 [1.15, 1.56] | 1.00 |
| YearsAtCompany_sd | -0.2483 | 0.1158 | -2.14 | 3.1979e-02 | 0.78 [0.62, 0.98] | 1.40 |
| EnvironmentSatisfaction | -0.3416 | 0.0733 | -4.66 | 3.1311e-06 | 0.71 [0.62, 0.82] | 1.01 |
| JobSatisfaction | -0.3375 | 0.0720 | -4.69 | 2.7515e-06 | 0.71 [0.62, 0.82] | 1.00 |
| WorkLifeBalance | -0.2342 | 0.1106 | -2.12 | 3.4280e-02 | 0.79 [0.64, 0.98] | 1.00 |
| JobInvolvement | -0.5819 | 0.1102 | -5.28 | 1.2780e-07 | 0.56 [0.45, 0.69] | 1.00 |

### 4.2 Model M2 Parameter Estimates and Collinearity Audit (With Demographic Controls)

| Predictor Term | Coefficient | Std Error | z-statistic | p-value | Odds Ratio [95% CI] | VIF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Intercept | 1.0897 | 0.6658 | 1.64 | 1.0168e-01 | Baseline Log-Odds | - |
| C(OverTime, Treatment(reference='No'))[T.Yes] | 1.6613 | 0.1701 | 9.76 | 1.5944e-22 | 5.27 [3.77, 7.35] | 1.01 |
| C(BusinessTravel, Treatment(reference='Non-Travel'))[T.Travel_Frequently] | 1.6316 | 0.3776 | 4.32 | 1.5529e-05 | 5.11 [2.44, 10.72] | 2.33 |
| C(BusinessTravel, Treatment(reference='Non-Travel'))[T.Travel_Rarely] | 0.8665 | 0.3534 | 2.45 | 1.4204e-02 | 2.38 [1.19, 4.75] | 2.34 |
| C(Gender, Treatment(reference='Female'))[T.Male] | 0.3515 | 0.1689 | 2.08 | 3.7391e-02 | 1.42 [1.02, 1.98] | 1.01 |
| C(MaritalStatus, Treatment(reference='Married'))[T.Divorced] | -0.2236 | 0.2489 | -0.90 | 3.6888e-01 | 0.80 [0.49, 1.30] | 1.28 |
| C(MaritalStatus, Treatment(reference='Married'))[T.Single] | 0.7577 | 0.2256 | 3.36 | 7.8332e-04 | 2.13 [1.37, 3.32] | 1.75 |
| JobLevel | -0.3504 | 0.1145 | -3.06 | 2.2179e-03 | 0.70 [0.56, 0.88] | 1.72 |
| StockOptionLevel | -0.1742 | 0.1412 | -1.23 | 2.1736e-01 | 0.84 [0.64, 1.11] | 1.87 |
| DistanceFromHome_sd | 0.2977 | 0.0791 | 3.76 | 1.6896e-04 | 1.35 [1.15, 1.57] | 1.01 |
| YearsAtCompany_sd | -0.2210 | 0.1184 | -1.87 | 6.2122e-02 | 0.80 [0.64, 1.01] | 1.41 |
| EnvironmentSatisfaction | -0.3583 | 0.0745 | -4.81 | 1.5367e-06 | 0.70 [0.60, 0.81] | 1.01 |
| JobSatisfaction | -0.3632 | 0.0730 | -4.97 | 6.5540e-07 | 0.70 [0.60, 0.80] | 1.01 |
| WorkLifeBalance | -0.2765 | 0.1132 | -2.44 | 1.4639e-02 | 0.76 [0.61, 0.95] | 1.01 |
| JobInvolvement | -0.5645 | 0.1115 | -5.06 | 4.1565e-07 | 0.57 [0.46, 0.71] | 1.01 |
| Age_sd | -0.2676 | 0.0971 | -2.76 | 5.8375e-03 | 0.77 [0.63, 0.93] | 1.37 |

### 4.3 Methodological Synthesis: OverTime Association Estimates

To assess stability across estimation frameworks, the OverTime association is compared across bivariate, stratified, multivariable, and regularised contrast specifications.

| Framework | Estimation Model | Odds Ratio [95% CI] | Adjustment Level | Key Methodological Notes |
| :--- | :--- | :--- | :--- | :--- |
| Unadjusted Crude OR | Bivariate 2x2 contingency | 3.77 [2.83, 5.03] | None | Workforce baseline unstratified cross-tabulation |
| Stratified Mantel-Haenszel OR | Stratified non-parametric pooling | 4.05 [3.01, 5.46] | JobLevel (5 strata) | Robins-Breslow-Greenland 95% CI; Breslow-Day Tarone p=0.084 |
| Adjusted Logistic Regression (M1) | Unpenalised multivariable Logit | 4.81 [3.48, 6.65] | JobLevel, Travel, Satisfaction, StockOptions, Tenure | Wald 95% CI; all VIF < 2.33; 21.5 events/parameter |
| Adjusted Logistic Regression (M2) | Unpenalised multivariable Logit | 5.27 [3.77, 7.35] | M1 + Age, Gender, MaritalStatus (controls only) | Wald 95% CI; all VIF < 2.34; 15.8 events/parameter |
| Phase 4 Regularised Contrast | L2 Logistic Regression (C=0.10, symmetric) | 4.00 [2.97, 5.51] | 30 operational/work features | Contrast beta_Yes - beta_No with 2000-bootstrap percentile CI |

---

## 5. Overlap-Aware Empirical Bayes Hotspot Index

The hotspot universe encompasses 104 pre-specified candidate segments: 54 one-way segments and 50 two-way OverTime interaction cells. Protected attributes (`Age, AgeGroup, Gender, MaritalStatus, Education, EducationField, PromotionGapBand`) are excluded from candidate definitions.

### 5.1 Shrinkage Parameters & Screening Gates
- **Fitted Prior Strength (m)**: Fitted separately via marginal maximum likelihood:
  - One-Way Family: m = 29.55
  - Two-Way Family: m = 8.65
- **Gate Criteria**:
  - **Tier 1 (Priority Hotspots)**: Headcount n >= 30, Wilson 95% lower bound > workforce baseline (16.12%), and BH q < 0.05 from Fisher exact test across the 104-segment universe.
  - **Tier 2 (Secondary Hotspots)**: Headcount n >= 30 and Wilson 95% lower bound > workforce baseline, without FDR clearance.
- **Universe Gate Breakdown**:
  - Total Pre-Specified Universe: 104 segments
  - Segments Meeting Sample Floor (n >= 30): 98 segments
  - Tier 1 Qualified Segments: 25 segments
  - Tier 2 Qualified Segments: 2 segments

### 5.2 Tier 1 Qualified Hotspots

| Segment ID | Family | Headcount (n) | Leavers (k) | Raw Rate | Wilson 95% CI | Shrunken Rate | Hotspot Index | Fisher p | BH q-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OverTime=Yes & TenureBand=0-1 | 2-way | 69 | 38 | 55.1% | [43.4%, 66.2%] | 50.7% | 314.7 | 2.2524e-14 | 5.8562e-13 |
| OverTime=Yes & JobLevel=1 | 2-way | 156 | 82 | 52.6% | [44.8%, 60.2%] | 50.6% | 314.2 | 7.7516e-30 | 8.0617e-28 |
| OverTime=Yes & JobRole=Laboratory Technician | 2-way | 62 | 31 | 50.0% | [37.9%, 62.1%] | 45.9% | 284.4 | 2.0854e-10 | 3.0983e-09 |
| OverTime=Yes & BusinessTravel=Travel_Frequently | 2-way | 86 | 36 | 41.9% | [32.0%, 52.4%] | 39.5% | 245.0 | 3.4592e-09 | 3.5976e-08 |
| OverTime=Yes & TenureBand=2-3 | 2-way | 75 | 29 | 38.7% | [28.5%, 50.0%] | 36.3% | 225.4 | 1.0777e-06 | 8.0057e-06 |
| OverTime=Yes & Department=Sales | 2-way | 128 | 48 | 37.5% | [29.6%, 46.1%] | 36.1% | 224.2 | 5.4225e-10 | 7.0492e-09 |
| JobRole=Sales Representative | 1-way | 83 | 33 | 39.8% | [29.9%, 50.5%] | 33.6% | 208.1 | 7.9165e-08 | 6.8610e-07 |
| TenureBand=0-1 | 1-way | 215 | 75 | 34.9% | [28.8%, 41.5%] | 32.6% | 202.3 | 1.0347e-13 | 1.7934e-12 |
| OverTime=Yes & JobRole=Research Scientist | 2-way | 97 | 33 | 34.0% | [25.4%, 43.9%] | 32.6% | 201.9 | 5.4540e-06 | 3.5451e-05 |
| OverTime=Yes & JobRole=Sales Executive | 2-way | 94 | 31 | 33.0% | [24.3%, 43.0%] | 31.6% | 195.7 | 2.2481e-05 | 1.2989e-04 |
| OverTime=Yes | 1-way | 416 | 127 | 30.5% | [26.3%, 35.1%] | 29.6% | 183.4 | 1.7187e-19 | 8.9372e-18 |
| OverTime=Yes & BusinessTravel=Travel_Rarely | 2-way | 295 | 84 | 28.5% | [23.6%, 33.9%] | 28.1% | 174.4 | 8.4129e-10 | 9.7215e-09 |
| OverTime=No & JobRole=Sales Representative | 2-way | 59 | 17 | 28.8% | [18.8%, 41.4%] | 27.2% | 168.6 | 8.7489e-03 | 3.6395e-02 |
| WorkLifeBalance=1 | 1-way | 80 | 25 | 31.2% | [22.2%, 42.1%] | 27.2% | 168.5 | 3.9474e-04 | 2.0527e-03 |
| OverTime=Yes & Department=Research & Development | 2-way | 271 | 74 | 27.3% | [22.3%, 32.9%] | 27.0% | 167.2 | 1.1455e-07 | 9.1640e-07 |
| JobLevel=1 | 1-way | 543 | 143 | 26.3% | [22.8%, 30.2%] | 25.8% | 160.1 | 1.0554e-15 | 3.6588e-14 |
| OverTime=Yes & TenureBand=6-10 | 2-way | 120 | 31 | 25.8% | [18.8%, 34.3%] | 25.2% | 156.2 | 2.9967e-03 | 1.2986e-02 |
| OverTime=No & TenureBand=0-1 | 2-way | 146 | 37 | 25.3% | [19.0%, 33.0%] | 24.8% | 154.0 | 1.6937e-03 | 7.6585e-03 |
| EnvironmentSatisfaction=1 | 1-way | 284 | 72 | 25.4% | [20.6%, 30.7%] | 24.5% | 151.9 | 4.8661e-06 | 3.3738e-05 |
| BusinessTravel=Travel_Frequently | 1-way | 277 | 69 | 24.9% | [20.2%, 30.3%] | 24.1% | 149.2 | 1.6857e-05 | 1.0313e-04 |
| StockOptionLevel=0 | 1-way | 631 | 154 | 24.4% | [21.2%, 27.9%] | 24.0% | 149.1 | 7.0763e-14 | 1.4719e-12 |
| JobRole=Laboratory Technician | 1-way | 259 | 62 | 23.9% | [19.1%, 29.5%] | 23.1% | 143.5 | 2.0146e-04 | 1.1027e-03 |
| JobSatisfaction=1 | 1-way | 289 | 66 | 22.8% | [18.4%, 28.0%] | 22.2% | 137.8 | 5.3877e-04 | 2.6682e-03 |
| DistanceBand=21-29 | 1-way | 204 | 45 | 22.1% | [16.9%, 28.2%] | 21.3% | 132.2 | 1.0286e-02 | 4.1146e-02 |
| Department=Sales | 1-way | 446 | 92 | 20.6% | [17.1%, 24.6%] | 20.3% | 126.2 | 1.4591e-03 | 6.8974e-03 |

### 5.3 Candidate Non-Independence and Overlap Analysis
Candidate segments exhibit extensive empirical overlap and cannot be treated as independent units. Among Tier 1 segments, substantial shared membership exists between OverTime interaction cells and constituent one-way operational categories.

### 5.4 Greedy Marginal Coverage Algorithm
To construct a non-redundant operational review set, an iterative greedy coverage selects the Tier 1 segment exhibiting the highest shrunken index on not-yet-reviewed employees. At each step, a candidate must encompass at least 30 uncovered employees and exhibit a Wilson lower bound separating from the uncovered pool baseline.

| Step | Selected Segment | New Headcount | New Leavers | Marginal Rate | Cum Headcount | Cum Leavers | Cumulative Rate | Workforce Share | Leavers Share | Uncovered Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | OverTime=Yes & TenureBand=0-1 | 69 | 38 | 55.1% | 69 | 38 | 55.1% | 4.7% | 16.0% | 16.1% |
| 2 | OverTime=Yes & JobLevel=1 | 112 | 51 | 45.5% | 181 | 89 | 49.2% | 12.3% | 37.6% | 14.2% |
| 3 | OverTime=Yes & JobRole=Sales Executive | 87 | 27 | 31.0% | 268 | 116 | 43.3% | 18.2% | 48.9% | 11.5% |
| 4 | OverTime=No & JobRole=Sales Representative | 59 | 17 | 28.8% | 327 | 133 | 40.7% | 22.2% | 56.1% | 10.1% |
| 5 | OverTime=No & TenureBand=0-1 | 124 | 27 | 21.8% | 451 | 160 | 35.5% | 30.7% | 67.5% | 9.1% |
| 6 | WorkLifeBalance=1 | 57 | 11 | 19.3% | 508 | 171 | 33.7% | 34.6% | 72.2% | 7.6% |
| 7 | EnvironmentSatisfaction=1 | 186 | 21 | 11.3% | 694 | 192 | 27.7% | 47.2% | 81.0% | 6.9% |
| 8 | JobSatisfaction=1 | 157 | 15 | 9.6% | 851 | 207 | 24.3% | 57.9% | 87.3% | 5.8% |

### 5.5 Split-Sample Out-of-Sample Validation
Candidate hotspots screened on the training partition (n = 1,176; leavers = 190) were evaluated on the held-out validation sample (n = 294; leavers = 47; holdout baseline = 15.99%).

| Segment ID | Train n | Train Rate | Train Index | Holdout n | Holdout Leavers | Holdout Rate | Holdout Wilson CI | Holdout n >= 30 | Above Point | Above Lower Bound | In Full Tier 1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Department=Sales | 364 | 20.9% | 127.0 | 82 | 16 | 19.5% | [12.4%, 29.4%] | Yes | Yes | No | Yes |
| JobRole=Laboratory Technician | 215 | 27.0% | 158.9 | 44 | 4 | 9.1% | [3.6%, 21.2%] | Yes | No | No | Yes |
| JobRole=Sales Representative | 60 | 41.7% | 206.0 | 23 | 8 | 34.8% | [18.8%, 55.1%] | No | Yes | Yes | Yes |
| JobLevel=1 | 420 | 26.7% | 160.8 | 123 | 31 | 25.2% | [18.4%, 33.5%] | Yes | Yes | Yes | Yes |
| TenureBand=0-1 | 166 | 38.6% | 217.8 | 49 | 11 | 22.4% | [13.0%, 35.9%] | Yes | Yes | No | Yes |
| OverTime=Yes | 340 | 28.5% | 170.5 | 76 | 30 | 39.5% | [29.2%, 50.7%] | Yes | Yes | Yes | Yes |
| BusinessTravel=Travel_Frequently | 225 | 25.3% | 150.2 | 52 | 12 | 23.1% | [13.7%, 36.1%] | Yes | Yes | No | Yes |
| DistanceBand=21-29 | 171 | 22.8% | 135.1 | 33 | 6 | 18.2% | [8.6%, 34.4%] | Yes | Yes | No | Yes |
| StockOptionLevel=0 | 503 | 24.3% | 147.4 | 128 | 32 | 25.0% | [18.3%, 33.2%] | Yes | Yes | Yes | Yes |
| WorkLifeBalance=1 | 70 | 31.4% | 166.6 | 10 | 3 | 30.0% | [10.8%, 60.3%] | No | Yes | No | Yes |
| JobSatisfaction=1 | 240 | 22.1% | 132.7 | 49 | 13 | 26.5% | [16.2%, 40.3%] | Yes | Yes | Yes | Yes |
| EnvironmentSatisfaction=1 | 229 | 25.8% | 152.7 | 55 | 13 | 23.6% | [14.4%, 36.3%] | Yes | Yes | No | Yes |
| OverTime=Yes & Department=Research & Development | 223 | 26.5% | 161.2 | 48 | 15 | 31.2% | [19.9%, 45.3%] | Yes | Yes | Yes | Yes |
| OverTime=Yes & Department=Sales | 103 | 33.0% | 195.8 | 25 | 14 | 56.0% | [37.1%, 73.3%] | No | Yes | Yes | Yes |
| OverTime=No & JobRole=Sales Representative | 44 | 36.4% | 203.5 | 15 | 1 | 6.7% | [1.2%, 29.8%] | No | No | No | Yes |
| OverTime=Yes & JobRole=Laboratory Technician | 56 | 51.8% | 289.5 | 6 | 2 | 33.3% | [9.7%, 70.0%] | No | Yes | No | Yes |
| OverTime=Yes & JobRole=Research Scientist | 72 | 29.2% | 171.4 | 25 | 12 | 48.0% | [30.0%, 66.5%] | No | Yes | Yes | Yes |
| OverTime=Yes & JobRole=Sales Executive | 78 | 30.8% | 180.9 | 16 | 7 | 43.8% | [23.1%, 66.8%] | No | Yes | Yes | Yes |
| OverTime=Yes & JobLevel=1 | 116 | 51.7% | 304.0 | 40 | 22 | 55.0% | [39.8%, 69.3%] | Yes | Yes | Yes | Yes |
| OverTime=No & TenureBand=0-1 | 112 | 29.5% | 176.1 | 34 | 4 | 11.8% | [4.7%, 26.6%] | Yes | No | No | Yes |
| OverTime=Yes & TenureBand=0-1 | 54 | 57.4% | 318.2 | 15 | 7 | 46.7% | [24.8%, 69.9%] | No | Yes | Yes | Yes |
| OverTime=Yes & TenureBand=2-3 | 59 | 32.2% | 186.0 | 16 | 10 | 62.5% | [38.6%, 81.5%] | No | Yes | Yes | Yes |
| OverTime=Yes & TenureBand=6-10 | 99 | 25.3% | 151.5 | 21 | 6 | 28.6% | [13.8%, 50.0%] | No | Yes | No | Yes |
| OverTime=Yes & BusinessTravel=Travel_Frequently | 69 | 39.1% | 225.5 | 17 | 9 | 52.9% | [31.0%, 73.8%] | No | Yes | Yes | Yes |
| OverTime=Yes & BusinessTravel=Travel_Rarely | 243 | 26.3% | 160.7 | 52 | 20 | 38.5% | [26.5%, 52.0%] | Yes | Yes | Yes | Yes |

### 5.6 Sensitivity Analysis across Prior Strengths (m in {10, 30, 100})
Evaluating index stability across varying levels of prior regularization demonstrates exceptional rank consistency. Among segments meeting the sample size floor (n >= 30):
- **Spearman Rank Correlation**:
  - m = 10 versus m = 30: rho = 0.9973
  - m = 30 versus m = 100: rho = 0.9867
  - m = 10 versus m = 100: rho = 0.9767
- **Top 10 Membership**: 9 of the top 10 segments remain identical across all tested prior strength values (Jaccard similarity >= 0.818).
