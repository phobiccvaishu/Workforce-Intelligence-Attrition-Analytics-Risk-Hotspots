# Workforce Intelligence: Data Dictionary

**Project:** Workforce Attrition Patterns and Risk Hotspot Analysis  
**Phase:** Phase 1 – Dataset Inspection & Contract Specification  
**Dataset File:** `Palo Alto Networks.csv`  

> [!IMPORTANT]
> DISCLAIMER: The dataset provenance is UNVERIFIED. This data is analyzed as a public/synthetic HR dataset. Findings must not be described as real Palo Alto Networks employment records or personnel data.  
> Ordinal label meanings and target encodings are NOT explicitly defined inside the raw CSV file. The semantic descriptions below reflect conventional interpretations from public HR benchmark datasets. Every assumed scale is tagged: **"ASSUMPTION: confirm with dataset provider"**. Currency symbols are omitted as currency denominations and pay periods are undocumented.

---

## 1. Schema & Field Definitions

| Field | Role | Observed Type | Observed Range / Levels | Assumed Meaning | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Age` | Continuous | `int64` | [18 .. 60] | Employee chronological age in years | High |
| `Attrition` | Target | `int64` | [0 .. 1] | Workforce departure status (1 = Left organization, voluntariness unknown, 0 = Retained/Stayed) [APPROVED by project owner] | High (Approved by project owner: 1=Left, voluntariness unknown) |
| `BusinessTravel` | Nominal | `str` | 3 levels: Non-Travel, Travel_Frequently, Travel_Rarely | Frequency of work-related travel (Travel_Rarely, Travel_Frequently, Non-Travel) | High |
| `DailyRate` | Continuous | `int64` | [102 .. 1,499] | Daily billing or compensation rate (undocumented currency and pay period) | Low (Questionable interpretability; undocumented currency/period) |
| `Department` | Nominal | `str` | 3 levels: Human Resources, Research & Development, Sales | Business department / division (Sales, Research & Development, Human Resources) | High |
| `DistanceFromHome` | Continuous | `int64` | [1 .. 29] | Distance from residence to workplace (undocumented unit: miles vs km) | Medium (Unit undocumented) |
| `Education` | Ordinal | `int64` | Observed [1, 2, 3, 4, 5] (Scale: 1–5) | Formal educational level attained (1 'Below College', 2 'College', 3 'Bachelor', 4 'Master', 5 'Doctor') [ASSUMPTION: confirm with dataset provider] | Medium (Ordinal scale assumed) |
| `EducationField` | Nominal | `str` | 6 levels: Human Resources, Life Sciences, Marketing, Medical, ... | Undergraduate or primary field of study | High |
| `EnvironmentSatisfaction` | Ordinal | `int64` | Observed [1, 2, 3, 4] (Scale: 1–4) | Workplace environment satisfaction (1 'Low', 2 'Medium', 3 'High', 4 'Very High') [ASSUMPTION: confirm with dataset provider] | Medium (Ordinal scale assumed) |
| `Gender` | Nominal | `str` | 2 levels: Female, Male | Reported gender (Female, Male) | High |
| `HourlyRate` | Continuous | `int64` | [30 .. 100] | Hourly billing or labor rate (undocumented currency and pay period) | Low (Questionable interpretability; undocumented currency/period) |
| `JobInvolvement` | Ordinal | `int64` | Observed [1, 2, 3, 4] (Scale: 1–4) | Degree of job engagement/involvement (1 'Low', 2 'Medium', 3 'High', 4 'Very High') [ASSUMPTION: confirm with dataset provider] | Medium (Ordinal scale assumed) |
| `JobLevel` | Ordinal | `int64` | Observed [1, 2, 3, 4, 5] (Scale: 1–5) | Hierarchy seniority level from 1 (Entry) to 5 (Executive) [ASSUMPTION: confirm with dataset provider] | High |
| `JobRole` | Nominal | `str` | 9 levels: Healthcare Representative, Human Resources, Laboratory Technician, Manager, ... | Specific organizational job title / role | High |
| `JobSatisfaction` | Ordinal | `int64` | Observed [1, 2, 3, 4] (Scale: 1–4) | Job satisfaction rating (1 'Low', 2 'Medium', 3 'High', 4 'Very High') [ASSUMPTION: confirm with dataset provider] | Medium (Ordinal scale assumed) |
| `MaritalStatus` | Nominal | `str` | 3 levels: Divorced, Married, Single | Marital status (Single, Married, Divorced) | High |
| `MonthlyIncome` | Continuous | `int64` | [1,009 .. 19,999] | Monthly compensation/salary in undocumented currency units | High (Undocumented currency) |
| `MonthlyRate` | Continuous | `int64` | [2,094 .. 26,999] | Monthly billing or overhead rate (undocumented currency and pay period) | Low (Questionable interpretability; undocumented currency/period) |
| `NumCompaniesWorked` | Discrete Count | `int64` | [0 .. 9] | Number of companies previously employed at (meaning of 0 is ambiguous: 0 prior employers or unrecorded) [ASSUMPTION: confirm with dataset provider] | Medium (Ambiguity at 0) |
| `OverTime` | Nominal | `str` | 2 levels: No, Yes | Exempt/non-exempt overtime work requirement (Yes, No) | High |
| `PercentSalaryHike` | Continuous | `int64` | [11 .. 25] | Percentage increase in salary received during last annual review | High |
| `PerformanceRating` | Ordinal | `int64` | Observed [3, 4] (Scale: 1–4) | Last annual performance appraisal rating: 'Excellent (3)' vs 'Outstanding (4)'; levels 1 and 2 unobserved [ASSUMPTION: confirm with dataset provider] | Medium (Restricted observed range [3, 4]) |
| `RelationshipSatisfaction` | Ordinal | `int64` | Observed [1, 2, 3, 4] (Scale: 1–4) | Workplace relationship satisfaction rating (1 'Low', 2 'Medium', 3 'High', 4 'Very High') [ASSUMPTION: confirm with dataset provider] | Medium (Ordinal scale assumed) |
| `StockOptionLevel` | Ordinal | `int64` | Observed [0, 1, 2, 3] (Scale: 0–3) | Company stock option incentive grant tier (0 'None', 1, 2, 3) [ASSUMPTION: confirm with dataset provider] | High |
| `TotalWorkingYears` | Discrete Count | `int64` | [0 .. 40] | Total cumulative career experience in years | High |
| `TrainingTimesLastYear` | Discrete Count | `int64` | [0 .. 6] | Number of training sessions or professional development courses attended in the prior year | High |
| `WorkLifeBalance` | Ordinal | `int64` | Observed [1, 2, 3, 4] (Scale: 1–4) | Work-life balance assessment (1 'Bad', 2 'Good', 3 'Better', 4 'Best') [ASSUMPTION: confirm with dataset provider] | Medium (Ordinal scale assumed) |
| `YearsAtCompany` | Discrete Count | `int64` | [0 .. 40] | Tenure at current enterprise in years | High |
| `YearsInCurrentRole` | Discrete Count | `int64` | [0 .. 18] | Tenure in current specific role in years | High |
| `YearsSinceLastPromotion` | Discrete Count | `int64` | [0 .. 15] | Years elapsed since most recent promotion (0 indicates promoted in current calendar year) [ASSUMPTION: confirm with dataset provider] | High |
| `YearsWithCurrManager` | Discrete Count | `int64` | [0 .. 17] | Tenure reporting to current direct manager in years | High |

---

## 2. Undocumented Field Nuances & Assumptions Requiring Confirmation

The following elements lack explicit technical metadata in the source file:
1. **Target Variable (`Attrition`):** Confirmed coding is `1 = Left / Resigned (voluntariness unknown)` and `0 = Retained / Stayed`. [APPROVED by project owner].
2. **Measurement Window:** The duration or observation window over which attrition events were recorded (e.g. past 12 months, annual fiscal cycle) is undocumented.
3. **Distance Unit (`DistanceFromHome`):** The numerical metric (1 to 29) does not specify units (miles vs kilometers).
4. **Compensation Currency & Periods (`MonthlyIncome`, `DailyRate`, `HourlyRate`, `MonthlyRate`):** Currency denomination is unstated; pay periods are unstated.
5. **Zero Interpretations:**
   - `NumCompaniesWorked = 0`: 197 individuals have `NumCompaniesWorked = 0` yet `TotalWorkingYears > YearsAtCompany`. Unclear whether 0 represents "zero prior companies" or "unreported prior experience".
   - `YearsSinceLastPromotion = 0`: 581 individuals report 0 years, interpreted as promotion within the current calendar year.

---

## 3. Fields with Questionable Interpretability

The correlation audit (C9) examined the four compensation rate metrics:
`DailyRate`, `HourlyRate`, `MonthlyRate`, `MonthlyIncome`.

- **Lack of Accounting Coherence:** In real corporate payroll systems, `DailyRate ≈ HourlyRate * 8` and `MonthlyIncome ≈ DailyRate * 21.67`. In this dataset:
  - Spearman correlation between `HourlyRate` and `DailyRate` is **-0.0028** (uncorrelated).
  - Spearman correlation between `DailyRate` and `MonthlyIncome` is **0.0076** (uncorrelated).
  - Spearman correlation between `HourlyRate` and `MonthlyIncome` is **-0.0177** (uncorrelated).
  - Spearman correlation between `MonthlyRate` and `MonthlyIncome` is **0.0347** (uncorrelated).
  - Over all 1,470 rows without trimming, the ratio `DailyRate / (HourlyRate * 8)` fluctuates between **0.15** and **6.21** (median 1.50).
- **Recommendation for Phase 2+:** Treat `DailyRate`, `HourlyRate`, and `MonthlyRate` with extreme caution. `MonthlyIncome` exhibits strong, logical hierarchy scaling with `JobLevel` (rho = 0.9204) and is the primary trustworthy compensation signal.


## 4. Phase 2 Derived Features Specification

The following 15 features (plus surrogate index `RowID`) were derived in Phase 2 in strict accordance with the Section 5.4 contract. No rows were dropped, original values were untouched, and all cut-points represent documented design proposals.

| Field | Role | Type | Categories / Bounds | Derivation Formula / Rule | Nullable | Analytical Use |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RowID` | Surrogate Identifier | `int64` | Integer Range | 0-based integer sequence (0 .. n-1) matching Phase 1 audit row numbers | `False` | `True` |
| `AgeGroup` | Categorical Band | `category` | 4 levels: 18-25, 26-35, 36-45, 46-60 | Right-inclusive cuts on Age: (17, 25], (25, 35], (35, 45], (45, 60] | `False` | `True` |
| `TenureBand` | Categorical Band | `category` | 5 levels: 0-1, 2-3, 4-5, 6-10, 11+ | Right-inclusive cuts on YearsAtCompany: [-1, 1], (1, 3], (3, 5], (5, 10], (10, 100] | `False` | `True` |
| `EarlyTenure` | Binary Flag | `int64` | {0, 1} | Binary indicator: YearsAtCompany <= 1 (fewer than 2 completed years) | `False` | `True` |
| `CareerStage` | Categorical Band | `category` | 4 levels: 0-5, 6-10, 11-20, 21+ | Right-inclusive cuts on TotalWorkingYears: [-1, 5], (5, 10], (10, 20], (20, 100] | `False` | `True` |
| `RoleTenureBand` | Categorical Band | `category` | 4 levels: 0-1, 2-3, 4-7, 8+ | Right-inclusive cuts on YearsInCurrentRole: [-1, 1], (1, 3], (3, 7], (7, 100] | `False` | `True` |
| `ManagerTenureBand` | Categorical Band | `category` | 4 levels: 0-1, 2-3, 4-7, 8+ | Right-inclusive cuts on YearsWithCurrManager: [-1, 1], (1, 3], (3, 7], (7, 100] | `False` | `True` |
| `PromotionGapBand` | Categorical Band | `category` | 4 levels: 0, 1-2, 3-5, 6+ | Right-inclusive cuts on YearsSinceLastPromotion: [-1, 0], (0, 2], (2, 5], (5, 100] | `False` | `True` |
| `DistanceBand` | Categorical Band | `category` | 4 levels: 1-5, 6-10, 11-20, 21-29 | Right-inclusive cuts on DistanceFromHome: [0, 5], (5, 10], (10, 20], (20, 29] (unit undocumented) | `False` | `True` |
| `PriorExperienceYears` | Discrete Count | `int64` | Integer Range | TotalWorkingYears - YearsAtCompany (pre-company professional experience) | `False` | `True` |
| `HasPriorExperience` | Binary Flag | `int64` | {0, 1} | Binary indicator: PriorExperienceYears > 0 | `False` | `True` |
| `RoleTenureShare` | Ratio Feature | `float64` | [0.0 .. 1.0] (44 NaNs) | YearsInCurrentRole / YearsAtCompany (NaN when YearsAtCompany == 0; zero-denominator rule) | `True` | `True` |
| `PromotionGapShare` | Ratio Feature | `float64` | [0.0 .. 1.0] (44 NaNs) | YearsSinceLastPromotion / YearsAtCompany (NaN when YearsAtCompany == 0; zero-denominator rule) | `True` | `True` |
| `PromotionGapEqualsTenure` | Binary Flag | `int64` | {0, 1} | Descriptive flag: YearsAtCompany > 0 & YearsSinceLastPromotion == YearsAtCompany | `False` | `True` |
| `PromotionAnalysisCohort` | Binary Flag | `int64` | {0, 1} | Cohort flag: YearsAtCompany >= 5 (minimum tenure for promotion analysis) | `False` | `True` |
| `OutstandingRating` | Binary Flag | `int64` | {0, 1} | Binary indicator: PerformanceRating == 4 ('Outstanding') | `False` | `True` |
