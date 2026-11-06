# Inferential Analysis Plan and Protocol (Phase 5)

This document formalizes the pre-registered inferential and statistical adjustment protocol for the Workforce Intelligence analytics program. All specifications are configured in `src/workforce_intel/config.py` prior to model evaluation.

## 1. Governance and Methodological Principles
- **Descriptive and Observational Scope**: Analyses characterize statistical associations in cross-sectional observational workforce records. No causal conclusions, directional assertions, or mechanistic assertions are made.
- **Multiplicity Control**: Benjamini-Hochberg (BH) False Discovery Rate (FDR) control is enforced across discrete pre-specified families at alpha = 0.05. Findings are reported with family identifier and family size (m).
- **Protected Attribute Governance**: Demographic attributes (`Age, Gender, MaritalStatus`) are evaluated for descriptive characterization only. They are strictly excluded from the hotspot screening universe and are incorporated into regression models solely as observational adjustment controls (Model M2).

## 2. Pre-Specified Analysis Families
1. **Categorical and Ordinal Association Family** (m = 28 dimensions):
   - Contingency Chi-square test of independence (Fisher exact on 2x2 tables).
   - Seeded permutation p-value (10,000 resamples, SEED=42) where expected cell counts are under 5.
   - Effect size: Cramer's V with 2,000-resample percentile bootstrap 95% confidence intervals.
   - Cochran-Armitage trend test for ordered dimensions with standardized integer band scores.
2. **Continuous Measure Distributional Family** (m = 9 continuous metrics):
   - Mann-Whitney U test with probability of superiority (A = U / (n1 * n2)) and 2,000-resample percentile bootstrap 95% confidence intervals.
3. **Stratified Epidemiological Family (Mantel-Haenszel)**:
   - Stratified common odds ratio pooling with Robins-Breslow-Greenland 95% confidence intervals.
   - Breslow-Day test for odds ratio homogeneity with Tarone adjustment.
   - Zero-cell handling: strata with zero exposed or zero unexposed employees are excluded from pooled calculation; +0.5 continuity correction applied solely to stratum-specific display rows with zero cells.
4. **Hotspot Index Screening Family** (m = 104 candidate segments):
   - One-way segments across 12 operational dimensions (12 dimensions, 54 segments).
   - Two-way interaction cells across OverTime intersections (5 pairs, 50 segments).
   - Two-tier gating:
     - Tier 1: n >= 30, Wilson 95% lower bound > workforce baseline, and Fisher exact BH q < 0.05.
     - Tier 2: n >= 30, Wilson 95% lower bound > workforce baseline, without FDR clearance.

## 3. Multivariable Model Formulations
- **Model M1 (Operational Predictors)**:
  - Features: `OverTime, BusinessTravel, JobLevel, StockOptionLevel, DistanceFromHome, YearsAtCompany, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance, JobInvolvement`
  - Unpenalised statsmodels Logit with continuous features standardized per 1 sample SD.
- **Model M2 (M1 + Demographic Controls)**:
  - Features: `OverTime, BusinessTravel, JobLevel, StockOptionLevel, DistanceFromHome, YearsAtCompany, EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance, JobInvolvement, Age, Gender, MaritalStatus`
  - Controls: `Age, Gender, MaritalStatus`
- **Excluded Features**: `MonthlyIncome, DailyRate, HourlyRate, MonthlyRate, TotalWorkingYears, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager`
  - Rationale: Excluded due to severe collinearity with included tenure, compensation tier (JobLevel), and baseline pay structures.

## 4. Hotspot Shrinkage & Overlap Protocol
- Empirical Bayes shrinkage: shrunken rate = (k + m * p0) / (n + m), where m is fitted via marginal maximum likelihood separately for one-way and two-way families.
- Overlap awareness: Pairwise intersection matrix and Jaccard similarity to document candidate non-independence.
- Greedy marginal coverage: Sequential optimization over not-yet-covered headcount with sample size and lower-bound separation floors.
