# Phase 4 Modelling Specification: Leakage-Safe Attrition Risk Modelling & Out-of-Sample Validation

**Author**: Workforce Intelligence Analytics Team  
**Status**: APPROVED & LOCKED (Pre-Flight Specification)  
**Target Dataset**: Phase 2 Cleaned & Feature-Engineered Dataset (`data/processed/workforce_attrition_processed.csv`, n = 1,470)  
**Random Seed**: `SEED = 42` (Fixed across all cross-validation, bootstrap resampling, and model initializations)

---

## 1. Problem Formulation and Operational Use

### 1.1 Target Definition
- **Target Variable**: `Attrition` (`0 = Retained`, `1 = Left`).
- **Target Prevalence (Workforce Baseline)**: 16.12% (237 leavers out of 1,470 employees).
- **Unit of Observation**: Individual employee at a single cross-sectional snapshot in time.

### 1.2 Decision Support & Operational Context
- **Intended Use**: Rank-order organizational cohorts and individual employee profiles to prioritize proactive retention check-ins, workload reviews, and career development interviews by HR business partners and line managers.
- **Explicit Non-Use**: Strictly decision-support only. **NO automated adverse action** (e.g., dismissal, demotion, withholding of compensation, or automated exclusion from promotion consideration).
- **Primary Metric**: **Precision-Recall Area Under Curve (PR-AUC)**. Because the target is imbalanced (16.1% positive rate), PR-AUC is far more informative than ROC-AUC, reflecting true positive identification quality across operational decision thresholds.
- **Secondary Evaluation Metrics**:
  - **Lift @ Top 10% and Top 20%**: Evaluates retention yield within the highest-risk deciles.
  - **Recall @ Top 10% and Top 20%**: Measures the proportion of total leavers captured in the prioritized triage pools.
  - **Brier Score**: Measures probability calibration and squared error (`(p - y)^2`).
  - **ROC-AUC**: Evaluates overall discrimination ability across all potential decision thresholds.

---

## 2. Input Features and Leakage Audit

### 2.1 Identifiers and Invariant Column Exclusion
The dataset contains identifiers and zero-variance constants that carry no predictive information or threaten unique employee tracking:
- `RowID`: Dropped (surrogate row identifier injected during Phase 2).
- `EmployeeCount`: Dropped in Phase 2 (constant = 1 across all rows).
- `StandardHours`: Dropped in Phase 2 (constant = 80 across all rows).
- `Over18`: Dropped in Phase 2 (constant = 'Y' across all rows).
- `EmployeeNumber`: Dropped in Phase 2 (raw employee surrogate identifier).

### 2.2 Feature Leakage Audit
Every candidate feature was audited for direct or indirect dependency on `Attrition`. No derived feature in Phase 2 utilized `Attrition` in its definition.

| Feature Name | Type | Kept / Dropped | Rationale / Derivation Audit |
| :--- | :--- | :--- | :--- |
| `RowID` | Integer | **Dropped** | Surrogate primary key; purely an index. |
| `Age` | Numeric | Kept (Set A) / Dropped (Set B) | Chronological age; protected demographic attribute. |
| `BusinessTravel` | Nominal | Kept (A & B) | Job travel requirement; pre-outcome operational attribute. |
| `DailyRate` | Numeric | Kept (A & B) | Compensation component; operational baseline. |
| `Department` | Nominal | Kept (A & B) | Organizational division; operational context. |
| `DistanceFromHome` | Numeric | Kept (A & B) | Commute burden; continuous pre-departure metric. |
| `Education` | Ordinal | Kept (A & B) | Educational attainment (1 to 5); pre-hire human capital. |
| `EducationField` | Nominal | Kept (A & B) | Academic discipline; human capital background. |
| `EnvironmentSatisfaction` | Ordinal | Kept (A & B) | Survey rating (1 to 4); pre-departure employee sentiment. |
| `Gender` | Binary | Kept (Set A) / Dropped (Set B) | Protected demographic attribute; dropped in Set B. |
| `HourlyRate` | Numeric | Kept (A & B) | Compensation component. |
| `JobInvolvement` | Ordinal | Kept (A & B) | Survey rating (1 to 4); employee sentiment. |
| `JobLevel` | Ordinal | Kept (A & B) | Organizational hierarchy tier (1 to 5). |
| `JobRole` | Nominal | Kept (A & B) | Functional job family (9 categories). |
| `JobSatisfaction` | Ordinal | Kept (A & B) | Survey rating (1 to 4); sentiment rating. |
| `MaritalStatus` | Nominal | Kept (Set A) / Dropped (Set B) | Protected demographic attribute; dropped in Set B. |
| `MonthlyIncome` | Numeric | Kept (A & B) | Continuous monthly compensation. |
| `MonthlyRate` | Numeric | Kept (A & B) | Internal billing/rate metric. |
| `NumCompaniesWorked` | Numeric | Kept (A & B) | Historical career mobility count. |
| `OverTime` | Binary | Kept (A & B) | Workload condition ('Yes' / 'No'); primary operational factor. |
| `PercentSalaryHike` | Numeric | Kept (A & B) | Most recent percentage compensation increase. |
| `PerformanceRating` | Ordinal | Kept (A & B) | Appraisal rating (3 or 4 in raw dataset). |
| `RelationshipSatisfaction` | Ordinal | Kept (A & B) | Survey rating (1 to 4); peer/supervisor sentiment. |
| `StockOptionLevel` | Ordinal | Kept (A & B) | Long-term incentive vesting tier (0 to 3). |
| `TotalWorkingYears` | Numeric | Kept (A & B) | Total cumulative career tenure across all employers. |
| `TrainingTimesLastYear` | Numeric | Kept (A & B) | Professional development frequency count. |
| `WorkLifeBalance` | Ordinal | Kept (A & B) | Survey rating (1 to 4); workload/personal balance sentiment. |
| `YearsAtCompany` | Numeric | Kept (A & B) | Organization-specific tenure; key temporal metric. |
| `YearsInCurrentRole` | Numeric | Kept (A & B) | Role stagnation / tenure metric. |
| `YearsSinceLastPromotion` | Numeric | Kept (A & B) | Promotion interval; career velocity metric. |
| `YearsWithCurrManager` | Numeric | Kept (A & B) | Relational supervisory tenure metric. |
| `AgeGroup` | Ordinal | Kept (Set A) / Dropped (Set B) | Derived from `Age` in Phase 2; dropped in Set B. |
| `TenureBand` | Ordinal | Kept (A & B) | Derived from `YearsAtCompany` in Phase 2; non-target. |
| `EarlyTenure` | Binary | Kept (A & B) | Indicator `YearsAtCompany <= 1`; non-target. |
| `CareerStage` | Ordinal | Kept (A & B) | Derived from `TotalWorkingYears`; non-target. |
| `RoleTenureBand` | Ordinal | Kept (A & B) | Derived from `YearsInCurrentRole`; non-target. |
| `ManagerTenureBand` | Ordinal | Kept (A & B) | Derived from `YearsWithCurrManager`; non-target. |
| `PromotionGapBand` | Ordinal | Kept (A & B) | Derived from `YearsSinceLastPromotion`; non-target. |
| `DistanceBand` | Ordinal | Kept (A & B) | Derived from `DistanceFromHome`; non-target. |
| `PriorExperienceYears` | Numeric | Kept (A & B) | `TotalWorkingYears - YearsAtCompany`; non-target. |
| `HasPriorExperience` | Binary | Kept (A & B) | Indicator `PriorExperienceYears > 0`; non-target. |
| `RoleTenureShare` | Numeric | Kept (A & B) | Ratio `YearsInCurrentRole / YearsAtCompany`; non-target. |
| `PromotionGapShare` | Numeric | Kept (A & B) | Ratio `YearsSinceLastPromotion / YearsAtCompany`; non-target. |
| `PromotionGapEqualsTenure`| Binary | Kept (A & B) | Indicator `YearsSinceLastPromotion == YearsAtCompany`; non-target. |
| `PromotionAnalysisCohort` | Binary | Kept (A & B) | Indicator `YearsAtCompany >= 5`; non-target. |
| `OutstandingRating` | Binary | Kept (A & B) | Indicator `PerformanceRating == 4`; non-target. |

---

## 3. Feature Sets A and B

### 3.1 Definition of Sets
- **Feature Set A (Full Candidate Pool)**:
  Includes all 45 kept features, including protected demographics: `Age`, `Gender`, `MaritalStatus`, and `AgeGroup`.
- **Feature Set B (Demographic-Free / Governance-Recommended)**:
  Excludes all protected demographic characteristics:
  - Excluded: `Age`, `Gender`, `MaritalStatus`, `AgeGroup`.
  - Retained: 41 operational, job-role, compensation, sentiment, and career tenure features.

### 3.2 Evaluation Rule and Recommendation Policy
- **Governance Principle**: Feature Set B is the **recommended operational model specification** to safeguard against disparate treatment and algorithmic bias.
- **Materiality Threshold & Timing Audit**: Feature Set A will only be considered if it provides a statistically material improvement in out-of-sample PR-AUC (point estimate gain > 0.03 and non-overlapping 95% bootstrap confidence intervals). 
  *Audit Note*: The 0.03 threshold first appeared in the repository in commit `111db00` alongside the initial modeling code and results, meaning its pre-experimental timing cannot be independently verified from git history. Feature Set A is numerically superior in 15-fold cross-validation by +0.0251 (LR) and +0.0212 (HGB). Feature Set B was locked on primary governance and ethical grounds (disparate treatment prevention). On holdout, the second criterion is confirmed: Set A PR-AUC (0.5797, 95% CI: [0.4345, 0.7074]) and Set B PR-AUC (0.5736, 95% CI: [0.4315, 0.7037]) exhibit overlapping 95% bootstrap confidence intervals, and the paired holdout difference (+0.0061, 95% CI: [-0.0481, +0.0616]) spans zero.

---

## 4. Evaluation and Validation Protocol

### 4.1 Split Strategy
- **Holdout Partition**: 80/20 stratified split (`n_train = 1,176`, `n_holdout = 294`), stratified on `Attrition` with `random_state = 42`.
- **Holdout Lock**: The holdout set is sequestered and **scored exactly once** at the conclusion of all model training, hyperparameter tuning, and probability calibration.
- **Inner Development Protocol**: Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 evaluation folds), applied strictly within the 80% training set (`random_state = 42`).

### 4.2 In-Fold Preprocessing Pipeline
To prevent data snooping across CV folds:
- All numeric features are scaled using `StandardScaler` fitted strictly on training fold data.
- Ordered categorical features are encoded using `OrdinalEncoder` with explicitly declared category ordering.
- Nominal categorical features are one-hot encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fitted strictly on training fold data.
- No transformations or imputation utilize global parameters computed over the holdout set.

### 4.3 Model Families
We deliberately evaluate a focused, interpretable set of models without combinatorial leaderboard hunting:
1. **Dummy Baseline**: Stratified / prior rate predictor (`DummyClassifier(strategy='prior')`).
2. **L2 Regularized Logistic Regression**: `LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)` tuned over $C \in \{0.01, 0.1, 1.0, 10.0\}$.
3. **HistGradientBoosting**: `HistGradientBoostingClassifier(random_state=42)` with small grid over `max_iter` $\in \{50, 100\}$, `learning_rate` $\in \{0.05, 0.1\}$, and `max_leaf_nodes` $\in \{15, 31\}$.
4. **Probability Calibration**: Calibrating the best performing model inside cross-validation using Sigmoid (Platt) scaling (`CalibratedClassifierCV(cv=5)`).

### 4.4 Uncertainty Estimation
- All holdout metrics (PR-AUC, ROC-AUC, Brier score, Recall@20%, Lift@20%) report 95% confidence intervals estimated via **2,000 bootstrap resamples** of the holdout predictions (`random_state = 42`).
