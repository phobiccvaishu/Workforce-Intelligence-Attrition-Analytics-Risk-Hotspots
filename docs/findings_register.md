# Findings Register: Exploratory Data Analysis

> [!IMPORTANT]
> **Mechanical Screening Protocol Notice:** The findings documented below result from a mechanical,
> pre-specified confidence interval screening rule applied across 28 analysis dimensions.
> **Flags are not statistical significance tests.** In a scan of multiple subgroup intervals,
> approximately 5.8 flags would be expected by random variation alone (0.05 x 115 segments evaluated).
> All findings are purely descriptive; formal multivariable hypothesis testing and multiplicity adjustments are reserved for Phase 4.

## 1. Synthesis & Scan Summary

> [!NOTE]
> **Multi-Dimension Correlation Notice:** Several evaluated dimensions exhibit substantial structural correlation
> (for example, multiple tenure measures, job level and stock option tier, job role and department).
> Consequently, flagged subgroups are not mutually independent findings.

- **Dimensions Evaluated:** 28
- **Subgroups Evaluated (Headcount n >= 30):** 115
- **Workforce Baseline Attrition Rate:** 16.1224%
- **PromotionAnalysisCohort Baseline Rate:** 10.7865% (96 leavers / 890 employees)
- **Subgroups Flagged Above Baseline:** 26
- **Subgroups Flagged Below Baseline:** 26
- **Dimensions with Zero Flagged Segments:** 4 (Gender, Education, HasPriorExperience, PerformanceRating)
- **Rough Count Expected by Chance Alone:** 5.8 (stated as: flags are not statistical tests)
- **Chance Comparison Statement:** 52 segments flagged against roughly 5.8 expected by chance alone.

---

## 2. Pre-Specified Risk Hotspot Candidates & Union Analysis

Across 58 scanned two-way interaction cells, exactly 16 subgroups met the pre-specified qualification criteria (headcount n >= 30 and Wilson 95% lower bound exceeding the workforce baseline 16.1224%).

> [!WARNING]
> **Non-Independence Notice:** The 16 candidate subgroups overlap considerably (some are nested subsets) and do not represent 16 independent factors.

- **Distinct Employees in Candidate Union:** 599 (40.7483% of workforce)
- **Distinct Leavers in Candidate Union:** 171 (72.1519% of total departures)
- **Attrition Rate within Union:** 28.5476%
- **Workforce Outside Candidate Union:** 871 employees, 66 leavers, attrition rate 7.5775%

---

## 3. Register of Subgroup Findings

| Finding ID | Dimension | Segment | Headcount (n) | Leavers | Rate (%) | 95% Wilson CI (%) | Baseline (%) | Direction | Robustness | Figure | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| `FIND-01` | Department | Sales | 446 | 92 | 20.6% | [17.1%, 24.6%] | 16.1% | `above baseline` | not assessed | `FIG-01` | None |
| `FIND-02` | JobRole | Healthcare Representative | 131 | 9 | 6.9% | [3.7%, 12.5%] | 16.1% | `below baseline` | not assessed | `FIG-02` | None |
| `FIND-03` | JobRole | Laboratory Technician | 259 | 62 | 23.9% | [19.1%, 29.5%] | 16.1% | `above baseline` | not assessed | `FIG-02` | None |
| `FIND-04` | JobRole | Manager | 102 | 5 | 4.9% | [2.1%, 11.0%] | 16.1% | `below baseline` | not assessed | `FIG-02` | None |
| `FIND-05` | JobRole | Manufacturing Director | 145 | 10 | 6.9% | [3.8%, 12.2%] | 16.1% | `below baseline` | not assessed | `FIG-02` | None |
| `FIND-06` | JobRole | Research Director | 80 | 2 | 2.5% | [0.7%, 8.7%] | 16.1% | `below baseline` | not assessed | `FIG-02` | None |
| `FIND-07` | JobRole | Sales Representative | 83 | 33 | 39.8% | [29.9%, 50.5%] | 16.1% | `above baseline` | not assessed | `FIG-02` | None |
| `FIND-08` | JobLevel | 1 | 543 | 143 | 26.3% | [22.8%, 30.2%] | 16.1% | `above baseline` | not assessed | `FIG-10` | None |
| `FIND-09` | JobLevel | 2 | 534 | 52 | 9.7% | [7.5%, 12.5%] | 16.1% | `below baseline` | not assessed | `FIG-10` | None |
| `FIND-10` | JobLevel | 4 | 106 | 5 | 4.7% | [2.0%, 10.6%] | 16.1% | `below baseline` | not assessed | `FIG-10` | None |
| `FIND-11` | JobLevel | 5 | 69 | 5 | 7.2% | [3.1%, 15.9%] | 16.1% | `below baseline` | not assessed | `FIG-10` | None |
| `FIND-12` | AgeGroup | 18-25 | 123 | 44 | 35.8% | [27.9%, 44.6%] | 16.1% | `above baseline` | same across cut points | `FIG-05` | Protected demographic attribute; descriptive only, not for individual scoring. Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-13` | AgeGroup | 26-35 | 606 | 116 | 19.1% | [16.2%, 22.5%] | 16.1% | `above baseline` | not assessed | `FIG-05` | Protected demographic attribute; descriptive only, not for individual scoring. Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-14` | AgeGroup | 36-45 | 468 | 43 | 9.2% | [6.9%, 12.1%] | 16.1% | `below baseline` | not assessed | `FIG-05` | Protected demographic attribute; descriptive only, not for individual scoring. Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-15` | Gender | None (all segments within baseline interval) | 1470 | 237 | - | - | 16.1% | `no separation` | not assessed | `FIG-14` | Protected demographic attribute; descriptive only, not for individual scoring. |
| `FIND-16` | MaritalStatus | Divorced | 327 | 33 | 10.1% | [7.3%, 13.8%] | 16.1% | `below baseline` | not assessed | `FIG-14` | Protected demographic attribute; descriptive only, not for individual scoring. |
| `FIND-17` | MaritalStatus | Married | 673 | 84 | 12.5% | [10.2%, 15.2%] | 16.1% | `below baseline` | not assessed | `FIG-14` | Protected demographic attribute; descriptive only, not for individual scoring. |
| `FIND-18` | MaritalStatus | Single | 470 | 120 | 25.5% | [21.8%, 29.7%] | 16.1% | `above baseline` | not assessed | `FIG-14` | Protected demographic attribute; descriptive only, not for individual scoring. |
| `FIND-19` | Education | None (all segments within baseline interval) | 1470 | 237 | - | - | 16.1% | `no separation` | not assessed | `FIG-14` | None |
| `FIND-20` | EducationField | Marketing | 159 | 35 | 22.0% | [16.3%, 29.1%] | 16.1% | `above baseline` | not assessed | `FIG-14` | None |
| `FIND-21` | EducationField | Technical Degree | 132 | 32 | 24.2% | [17.7%, 32.2%] | 16.1% | `above baseline` | not assessed | `FIG-14` | None |
| `FIND-22` | TenureBand | 0-1 | 215 | 75 | 34.9% | [28.8%, 41.5%] | 16.1% | `above baseline` | same across cut points | `FIG-06` | Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-23` | TenureBand | 6-10 | 448 | 55 | 12.3% | [9.6%, 15.6%] | 16.1% | `below baseline` | not assessed | `FIG-06` | Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-24` | TenureBand | 11+ | 246 | 20 | 8.1% | [5.3%, 12.2%] | 16.1% | `below baseline` | not assessed | `FIG-06` | Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-25` | EarlyTenure | 0 | 1255 | 162 | 12.9% | [11.2%, 14.9%] | 16.1% | `below baseline` | not assessed | `FIG-06` | Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-26` | EarlyTenure | 1 | 215 | 75 | 34.9% | [28.8%, 41.5%] | 16.1% | `above baseline` | same across cut points | `FIG-06` | Tenure and age bands share substantial overlap; early tenure and younger age cohorts both exhibit elevated attrition. |
| `FIND-27` | CareerStage | 0-5 | 316 | 91 | 28.8% | [24.1%, 34.0%] | 16.1% | `above baseline` | not assessed | `FIG-06` | None |
| `FIND-28` | CareerStage | 11-20 | 340 | 39 | 11.5% | [8.5%, 15.3%] | 16.1% | `below baseline` | not assessed | `FIG-06` | None |
| `FIND-29` | CareerStage | 21+ | 207 | 16 | 7.7% | [4.8%, 12.2%] | 16.1% | `below baseline` | not assessed | `FIG-06` | None |
| `FIND-30` | RoleTenureBand | 0-1 | 301 | 84 | 27.9% | [23.1%, 33.2%] | 16.1% | `above baseline` | not assessed | `FIG-06` | None |
| `FIND-31` | RoleTenureBand | 4-7 | 399 | 49 | 12.3% | [9.4%, 15.9%] | 16.1% | `below baseline` | not assessed | `FIG-06` | None |
| `FIND-32` | RoleTenureBand | 8+ | 263 | 20 | 7.6% | [5.0%, 11.5%] | 16.1% | `below baseline` | not assessed | `FIG-06` | None |
| `FIND-33` | ManagerTenureBand | 0-1 | 339 | 96 | 28.3% | [23.8%, 33.3%] | 16.1% | `above baseline` | not assessed | `FIG-06` | None |
| `FIND-34` | ManagerTenureBand | 8+ | 271 | 22 | 8.1% | [5.4%, 12.0%] | 16.1% | `below baseline` | not assessed | `FIG-06` | None |
| `FIND-35` | PromotionGapBand | 3-5 | 158 | 16 | 10.1% | [6.3%, 15.8%] | 16.1% | `below baseline` | not assessed | `FIG-07` | Promotion gap progression is confounded by tenure in pooled workforce; isolated within PromotionAnalysisCohort (YearsAtCompany >= 5). |
| `FIND-36` | PromotionGapEqualsTenure | 1 | 135 | 33 | 24.4% | [18.0%, 32.3%] | 16.1% | `above baseline` | not assessed | `FIG-07` | None |
| `FIND-37` | HasPriorExperience | None (all segments within baseline interval) | 1470 | 237 | - | - | 16.1% | `no separation` | not assessed | `FIG-14` | None |
| `FIND-38` | NumCompaniesWorked | 3 | 159 | 16 | 10.1% | [6.3%, 15.7%] | 16.1% | `below baseline` | not assessed | `FIG-06` | None |
| `FIND-39` | NumCompaniesWorked | 5 | 63 | 16 | 25.4% | [16.3%, 37.3%] | 16.1% | `above baseline` | not assessed | `FIG-06` | None |
| `FIND-40` | OverTime | No | 1054 | 110 | 10.4% | [8.7%, 12.4%] | 16.1% | `below baseline` | not assessed | `FIG-08` | OverTime leaver rates remain elevated across job levels, with highest rates in early career (Job Level 1). |
| `FIND-41` | OverTime | Yes | 416 | 127 | 30.5% | [26.3%, 35.1%] | 16.1% | `above baseline` | not assessed | `FIG-08` | OverTime leaver rates remain elevated across job levels, with highest rates in early career (Job Level 1). |
| `FIND-42` | BusinessTravel | Non-Travel | 150 | 12 | 8.0% | [4.6%, 13.5%] | 16.1% | `below baseline` | not assessed | `FIG-08` | None |
| `FIND-43` | BusinessTravel | Travel_Frequently | 277 | 69 | 24.9% | [20.2%, 30.3%] | 16.1% | `above baseline` | not assessed | `FIG-08` | None |
| `FIND-44` | DistanceBand | 21-29 | 204 | 45 | 22.1% | [16.9%, 28.2%] | 16.1% | `above baseline` | same across cut points | `FIG-08` | None |
| `FIND-45` | WorkLifeBalance | 1 | 80 | 25 | 31.2% | [22.2%, 42.1%] | 16.1% | `above baseline` | not assessed | `FIG-09` | None |
| `FIND-46` | JobSatisfaction | 1 | 289 | 66 | 22.8% | [18.4%, 28.0%] | 16.1% | `above baseline` | not assessed | `FIG-09` | None |
| `FIND-47` | JobSatisfaction | 4 | 459 | 52 | 11.3% | [8.7%, 14.6%] | 16.1% | `below baseline` | not assessed | `FIG-09` | None |
| `FIND-48` | EnvironmentSatisfaction | 1 | 284 | 72 | 25.4% | [20.6%, 30.7%] | 16.1% | `above baseline` | not assessed | `FIG-09` | None |
| `FIND-49` | RelationshipSatisfaction | 1 | 276 | 57 | 20.7% | [16.3%, 25.8%] | 16.1% | `above baseline` | not assessed | `FIG-09` | None |
| `FIND-50` | JobInvolvement | 1 | 83 | 28 | 33.7% | [24.5%, 44.4%] | 16.1% | `above baseline` | not assessed | `FIG-09` | None |
| `FIND-51` | JobInvolvement | 4 | 144 | 13 | 9.0% | [5.4%, 14.8%] | 16.1% | `below baseline` | not assessed | `FIG-09` | None |
| `FIND-52` | StockOptionLevel | 0 | 631 | 154 | 24.4% | [21.2%, 27.9%] | 16.1% | `above baseline` | not assessed | `FIG-08` | None |
| `FIND-53` | StockOptionLevel | 1 | 596 | 56 | 9.4% | [7.3%, 12.0%] | 16.1% | `below baseline` | not assessed | `FIG-08` | None |
| `FIND-54` | StockOptionLevel | 2 | 158 | 12 | 7.6% | [4.4%, 12.8%] | 16.1% | `below baseline` | not assessed | `FIG-08` | None |
| `FIND-55` | TrainingTimesLastYear | 0 | 54 | 15 | 27.8% | [17.6%, 40.9%] | 16.1% | `above baseline` | not assessed | `FIG-08` | None |
| `FIND-56` | PerformanceRating | None (all segments within baseline interval) | 1470 | 237 | - | - | 16.1% | `no separation` | not assessed | `FIG-14` | None |
| `FIND-57` | PromotionGapBand_Cohort | 6+ | 215 | 35 | 16.3% | [11.9%, 21.8%] | 10.8% | `above baseline` | same across cut points | `FIG-07` | Evaluated strictly within PromotionAnalysisCohort (YearsAtCompany >= 5) against cohort baseline to control for tenure progression. |

---

## 4. What This Analysis Cannot Show

This exploratory assessment is subject to substantial analytical and observational limitations:

- **No Causal Inference:** Observed elevated rates reflect statistical associations in observational records. They do not demonstrate that any attribute produced or generated departures.
- **Cross-Sectional Single Snapshot:** The data represents a single cross-sectional view without temporal departure timestamps, survival histories, or sequence intervals.
- **Unobserved Departure Reasons:** Voluntariness of departures is unrecorded; voluntary resignations cannot be separated from involuntary terminations, retirements, or organizational reductions.
- **Undocumented Measurement Units:** Several numeric features (such as commute distance) lack documented operational units (miles versus kilometers).
- **Assumed Survey Scales:** Employee sentiment and satisfaction ratings represent assumed equidistant Likert scales without external calibration or longitudinal validation.
- **Unverified Dataset Provenance:** The dataset originates from public synthetic HR records and must not be interpreted as authentic operational records of Palo Alto Networks.
