# Workforce Intelligence KPI Registry & Definitions

**Document Status**: Official Specification (Generated from `src/workforce_intel/config.py`)  
**Scope**: 10 Key Performance Indicators for Decision Support and Monitoring  
**Target Unit**: Employee (Attrition Yes = 1, voluntariness unrecorded)  
**Methodological Standard**: Descriptive association, Wilson score intervals, pre-registered static benchmarks

---

## 1. Architectural Architecture & Governance Rules

The KPI module exposes 10 standardized performance metrics divided into two distinct architectural classes:
1. **LIVE KPIs (KPI-01 through KPI-05)**: Evaluated dynamically on the processed dataset (`data/processed/workforce_attrition_processed.csv`) for any valid subpopulation filter. Protected demographic attributes (`AgeGroup`, `Gender`, `MaritalStatus`) are permitted solely as descriptive observational filters and are strictly prohibited from serving as segment definers for automated actions.
2. **STATIC Benchmarks (KPI-06 through KPI-10)**: Workforce-level benchmarks extracted from committed, immutable Phase 4 and Phase 5 statistical outputs. All static tables are verified upon loading via cryptographic SHA-256 integrity checksums. Static KPIs reject subpopulation filters with an explicit `ValueError` to prevent misleading comparisons against pre-computed full-population or split-sample models.

### Standardized Reliability Classification:
- **`reliable`**: The evaluated subpopulation meets the sample size floor (\(n \ge 30\)) and all underlying contingency cells satisfy \(a, b, c, d \ge 30\).
- **`low_n`**: Headcount \(0 < n < 30\) or one or more contingency cells fall below 30. Metric is returned for visibility but flagged as underpowered; ranking across low-n cells is prohibited.
- **`no_data`**: The filter yields zero matching employees (\(n = 0\)). Returns `value = None` without raising an error.
- **`exposure_fixed_by_filter`**: The active filter fixes a single OverTime exposure level (e.g. OverTime=Yes), leaving zero unexposed employees and returning odds_ratio = None.
- **`unreliable_zero_cell`**: Contingency table contains a zero cell. Returns odds_ratio = None in calculation paths; any continuity-corrected odds ratio is display-only and labeled 'continuity-corrected'.
- **`workforce-level, not filterable`**: Benchmark applies to the full population or sequestered split and rejects ad-hoc filtering.

---

## 2. Complete KPI Registry

| KPI ID | Name | Class | Definition | Unit | Source / Function | Reliability Rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `KPI-01` | **Attrition Rate** | `LIVE` | Proportion of workforce who departed (leavers / headcount) accompanied by Wilson 95% score interval. | % (proportion) | `src.workforce_intel.kpis.kpi_01_attrition_rate` | Flagged as 'low_n' when headcount n < 30; flagged as 'no_data' when n = 0; otherwise 'reliable'. |
| `KPI-02` | **Headcount and Leavers Breakdown** | `LIVE` | Filtered population counts (headcount, leavers) alongside share of overall workforce headcount and total leavers. | Counts & % | `src.workforce_intel.kpis.kpi_02_headcount_leavers` | Flagged as 'low_n' when headcount n < 30; flagged as 'no_data' when n = 0; otherwise 'reliable'. |
| `KPI-03` | **Relative Attrition vs Baseline** | `LIVE` | Absolute percentage-point difference and ratio relative to overall workforce baseline (16.12%), evaluated via Wilson interval separation rule. | Percentage points & Ratio | `src.workforce_intel.kpis.kpi_03_relative_attrition` | Flagged as 'low_n' when headcount n < 30; flagged as 'no_data' when n = 0; otherwise 'reliable'. |
| `KPI-04` | **OverTime Association Within Filter** | `LIVE` | Crude odds ratio for OverTime (Yes vs No) within filtered subpopulation, with Woolf 95% confidence interval and OverTime exposure share. | Odds Ratio & % | `src.workforce_intel.kpis.kpi_04_overtime_association` | Flagged as 'exposure_fixed_by_filter' returning OR=None if active filter fixes a single exposure level; flagged as 'unreliable_zero_cell' returning OR=None if any 2x2 cell equals zero; flagged as 'low_n' if any cell < 30; flagged as 'no_data' if n = 0; otherwise 'reliable'. |
| `KPI-05` | **Promotion-Gap Cohort Rate** | `LIVE` | Attrition rate and crude odds ratio for long promotion gap (6+ vs 0-5 years) restricted to PromotionAnalysisCohort (TotalWorkingYears >= 6, YearsAtCompany >= 6, non-executive). | % & Odds Ratio | `src.workforce_intel.kpis.kpi_05_promotion_gap_cohort` | Flagged as 'low_n' when cohort headcount n < 30; flagged as 'no_data' when n = 0; otherwise 'reliable'. |
| `KPI-06` | **Adjusted OverTime Association** | `STATIC` | Workforce-level OverTime association across adjustment frameworks: Mantel-Haenszel common OR stratified by JobLevel, multivariable Model M1 (operational controls), and Model M2 (with demographic controls). | Odds Ratio [95% CI] | `reports/phase5/tables/overtime_comparison.csv` | Workforce-level static benchmark; verified via SHA-256 integrity check. Rejects ad-hoc filters. |
| `KPI-07` | **Hotspot Universe Summary** | `STATIC` | Structured summary of pre-registered 104 candidate segments: segments meeting sample size floor (n >= 30), Tier 1 priority hotspots, Tier 2 secondary hotspots, and fitted empirical Bayes prior strengths. | Segment counts & fitted m | `reports/phase5/tables/hotspot_universe.csv` | Workforce-level static benchmark; verified via SHA-256 integrity check. Rejects ad-hoc filters. |
| `KPI-08` | **Hotspot Holdout Validation** | `STATIC` | Out-of-sample replication summary for candidate hotspots screened on training partition (n = 1,176) and evaluated on sequestered holdout partition (n = 294). | Replicated segment counts | `reports/phase5/tables/hotspot_split_validation.csv` | Workforce-level static benchmark; verified via SHA-256 integrity check. Rejects ad-hoc filters. |
| `KPI-09` | **Hotspot Greedy Coverage** | `STATIC` | Cumulative marginal workforce and leaver coverage progression across greedy non-redundant segment selection steps. | Cumulative % | `reports/phase5/tables/greedy_coverage.csv` | Workforce-level static benchmark; verified via SHA-256 integrity check. Rejects ad-hoc filters. |
| `KPI-10` | **Model Triage Yield** | `STATIC` | Out-of-sample decision-support yield for calibrated regularised logistic regression at top-10% and top-20% risk triage cutoffs on sequestered holdout partition. | Precision, Recall, Lift | `reports/phase4/tables/holdout_metrics.csv` | Workforce-level static benchmark; verified via SHA-256 integrity check. Rejects ad-hoc filters. Labeled advisory decision support; no automated adverse actions. |

---

## 3. Allowed Filter Dimensions (Conjunctive AND Logic)

Filter keys must belong to the pre-approved domain schema configured in `src/workforce_intel/config.py`. Unknown dimensions or unrecognized category levels raise an immediate `ValueError`:

- `AgeGroup`
- `BusinessTravel`
- `CareerStage`
- `Department`
- `DistanceBand`
- `EarlyTenure`
- `Education`
- `EducationField`
- `EnvironmentSatisfaction`
- `Gender`
- `HasPriorExperience`
- `JobInvolvement`
- `JobLevel`
- `JobRole`
- `JobSatisfaction`
- `ManagerTenureBand`
- `MaritalStatus`
- `NumCompaniesWorked`
- `OverTime`
- `PerformanceRating`
- `PromotionGapBand`
- `PromotionGapEqualsTenure`
- `RelationshipSatisfaction`
- `RoleTenureBand`
- `StockOptionLevel`
- `TenureBand`
- `TrainingTimesLastYear`
- `WorkLifeBalance`

---

## 4. Decision Support Governance Principle

All KPI outputs are generated exclusively for **retention review, workforce planning, and organizational decision support**. Under no circumstances should individual-level or segment-level risk metrics be utilized for automated employment actions, disciplinary decisions, or adverse termination workflows.
