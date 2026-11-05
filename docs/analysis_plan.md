# Workforce Intelligence: Exploratory Data Analysis & Hotspot Protocol Plan

This document formalizes the pre-specified analytical protocol for Phase 3A (Descriptive EDA Engine). All analyses, interaction cross-tabulations, and sensitivity checks are enumerated in advance to prevent selective reporting, p-hacking, and post-hoc data dredging.

---

## 1. Governance, Ethical Guardrails & Language Policy

1. **Strict Descriptive Focus:** Phase 3A reports observed employee attrition rates, proportions, counts, and statistical intervals. In accordance with project policy, no causal statements, attributions, or directional assertions are made. Observed subgroup differences represent historical patterns and do not indicate individual probability or organizational fault.
2. **Protected Attributes Guardrail:** Demographic attributes (such as Age, Gender, and Marital Status) are evaluated strictly for aggregate organizational benchmarking and representation parity. They are never designated for individual scoring, decision-making, or predictive modeling.
3. **Multiplicity Control & Hotspot Screening:** Interaction screening evaluates multiple cross-classifications simultaneously. Cells with elevated rates are designated exclusively as *hotspot candidates for formal statistical evaluation in Phase 4*, not established findings.
4. **Sample Size Floor (n < 30):** Any subgroup with total sample size n < 30 is explicitly tagged with a low-sample warning flag. Confidence intervals reflect small-sample uncertainty, and findings in small cohorts are interpreted with caution.

---

## 2. Pre-Specified Analysis Registry

The following analyses are systematically executed against `data/processed/workforce_attrition_processed.csv`:

| Stage | Analysis Name | Primary Dimension | Analytical Objective | Protected Attribute |
| :--- | :--- | :--- | :--- | :--- |
| `Stage B` | **Overall Baseline** | `overall` | Baseline workforce size, total leavers, overall attrition rate, and 95% Wilson confidence interval. | No |
| `Stage C` | **Department Breakdown** | `Department` | Attrition rate and leaver distribution by business unit. | No |
| `Stage C` | **JobRole Breakdown** | `JobRole` | Attrition rate and leaver distribution across all 9 job roles. | No |
| `Stage C` | **Department x JobRole Hierarchy** | `Department` | Rate breakdown across populated organizational hierarchy cells. | No |
| `Stage C` | **JobLevel Breakdown** | `JobLevel` | Attrition rate by hierarchical job seniority level (1 to 5). | No |
| `Stage C` | **Leaver Concentration (Roles & Departments)** | `concentration` | Cumulative leaver volume rankings illustrating organizational concentration. | No |
| `Stage C` | **Funnel Control Limits (JobRole & Department)** | `funnel` | Exact binomial control limits (95% and 99.8%) around overall baseline rate. | No |
| `Stage D` | **AgeGroup Breakdown** | `AgeGroup` | Attrition rate across age cohorts (descriptive evaluation). | Yes (Descriptive Only) |
| `Stage D` | **Gender Breakdown** | `Gender` | Attrition rate by gender (descriptive evaluation). | Yes (Descriptive Only) |
| `Stage D` | **MaritalStatus Breakdown** | `MaritalStatus` | Attrition rate by marital status (descriptive evaluation). | Yes (Descriptive Only) |
| `Stage D` | **Education Breakdown** | `Education` | Attrition rate across educational attainment levels (1 to 5). | No |
| `Stage D` | **EducationField Breakdown** | `EducationField` | Attrition rate by field of study, highlighting small-sample categories. | No |
| `Stage E` | **TenureBand Breakdown** | `TenureBand` | Attrition rate across company tenure brackets. | No |
| `Stage E` | **EarlyTenure Breakdown** | `EarlyTenure` | Attrition rate comparison for new joiners (<= 1 year tenure) versus tenured staff. | No |
| `Stage E` | **CareerStage Breakdown** | `CareerStage` | Attrition rate across total career experience brackets. | No |
| `Stage E` | **RoleTenureBand Breakdown** | `RoleTenureBand` | Attrition rate across years in current role brackets. | No |
| `Stage E` | **ManagerTenureBand Breakdown** | `ManagerTenureBand` | Attrition rate across years with current manager brackets. | No |
| `Stage E` | **PromotionGapBand Pooled Breakdown** | `PromotionGapBand` | Attrition rate across promotion interval bands across all employees. | No |
| `Stage E` | **PromotionGapBand Cohort Breakdown** | `PromotionGapBand` | Attrition rate across promotion interval bands restricted to eligible cohort (tenure >= 5). | No |
| `Stage E` | **PromotionGap Pooled vs Cohort Comparison** | `side_by_side` | Side-by-side comparison of pooled versus cohort-restricted promotion gap rates. | No |
| `Stage E` | **TenureBand x PromotionGapBand Matrix** | `TenureBand` | Cross-classification illustrating the intersection of organizational tenure and promotion delay. | No |
| `Stage E` | **PromotionGapEqualsTenure Breakdown** | `PromotionGapEqualsTenure` | Attrition rate for employees whose entire tenure matches their promotion interval. | No |
| `Stage E` | **HasPriorExperience Breakdown** | `HasPriorExperience` | Attrition rate comparing experienced joiners versus company-only careers. | No |
| `Stage E` | **NumCompaniesWorked Breakdown** | `NumCompaniesWorked` | Attrition rate by reported number of previous employers, noting zero-count ambiguity. | No |
| `Stage F` | **OverTime Breakdown** | `OverTime` | Attrition rate comparison between overtime and non-overtime cohorts. | No |
| `Stage F` | **BusinessTravel Breakdown** | `BusinessTravel` | Attrition rate across business travel frequencies. | No |
| `Stage F` | **DistanceBand Breakdown** | `DistanceBand` | Attrition rate across commute distance brackets. | No |
| `Stage F` | **WorkLifeBalance Breakdown** | `WorkLifeBalance` | Attrition rate across survey work-life balance scores (1 to 4). | No |
| `Stage F` | **JobSatisfaction Breakdown** | `JobSatisfaction` | Attrition rate across job satisfaction scores (1 to 4). | No |
| `Stage F` | **EnvironmentSatisfaction Breakdown** | `EnvironmentSatisfaction` | Attrition rate across work environment satisfaction scores (1 to 4). | No |
| `Stage F` | **RelationshipSatisfaction Breakdown** | `RelationshipSatisfaction` | Attrition rate across peer relationship satisfaction scores (1 to 4). | No |
| `Stage F` | **JobInvolvement Breakdown** | `JobInvolvement` | Attrition rate across job involvement scores (1 to 4). | No |
| `Stage G` | **StockOptionLevel Breakdown** | `StockOptionLevel` | Attrition rate across equity grant tiers (0 to 3). | No |
| `Stage G` | **TrainingTimesLastYear Breakdown** | `TrainingTimesLastYear` | Attrition rate across annual training session frequencies (0 to 6). | No |
| `Stage G` | **PerformanceRating Breakdown** | `PerformanceRating` | Attrition rate between observed performance tiers 3 and 4. | No |
| `Stage G` | **MonthlyIncome Summary by JobLevel & Outcome** | `stratified_numeric` | Compensation medians and quartiles stratified by job level and departure outcome. | No |
| `Stage G` | **Numeric Summaries by Outcome** | `numeric_summary` | Distributional summary statistics (mean, std, median, quartiles) for continuous metrics by outcome. | No |
| `Stage H` | **Hotspot Candidates Scan** | `hotspot_scan` | Screening of all two-way interaction cells with n >= 30 where Wilson lower bound exceeds overall baseline. | No |
| `Stage I` | **Sensitivity Analysis** | `sensitivity` | Evaluation of alternative bin boundary definitions and threshold assumptions. | No |

---

## 3. Pre-Specified Two-Way Interaction Specifications

The following two-way interactions are exhaustively computed to examine cross-dimensional attrition patterns:

| Interaction Pair | Evaluation Type | Destination Table |
| :--- | :--- | :--- |
| `OverTime x Department` | Full Cartesian cross-tabulation | `reports/phase3/tables/two_way_overtime_x_department.csv` |
| `OverTime x JobRole` | Full Cartesian cross-tabulation | `reports/phase3/tables/two_way_overtime_x_jobrole.csv` |
| `OverTime x JobLevel` | Full Cartesian cross-tabulation | `reports/phase3/tables/two_way_overtime_x_joblevel.csv` |
| `OverTime x AgeGroup` | Full Cartesian cross-tabulation | `reports/phase3/tables/two_way_overtime_x_agegroup.csv` |
| `OverTime x TenureBand` | Full Cartesian cross-tabulation | `reports/phase3/tables/two_way_overtime_x_tenureband.csv` |
| `OverTime x BusinessTravel` | Full Cartesian cross-tabulation | `reports/phase3/tables/two_way_overtime_x_businesstravel.csv` |

---

## 4. Sensitivity Analysis Specifications

Alternative binning definitions and threshold parameters are systematically evaluated to confirm that findings do not depend on arbitrary categorization boundaries:

| Dimension Family | Specification Name | Parameter Type | Target Metric |
| :--- | :--- | :--- | :--- |
| `EarlyTenure` | **YearsAtCompany <= 1 (Default)** | `threshold` | `YearsAtCompany` |
| `EarlyTenure` | **YearsAtCompany <= 2** | `threshold` | `YearsAtCompany` |
| `EarlyTenure` | **YearsAtCompany <= 3** | `threshold` | `YearsAtCompany` |
| `AgeGroup` | **4 Bands (Default: 18-25, 26-35, 36-45, 46-60)** | `bins` | `Age` |
| `AgeGroup` | **5 Bands (18-24, 25-34, 35-44, 45-54, 55-60)** | `bins` | `Age` |
| `AgeGroup` | **3 Bands (18-30, 31-45, 46-60)** | `bins` | `Age` |
| `TenureBand` | **5 Bands (Default: 0-1, 2-3, 4-5, 6-10, 11+)** | `bins` | `YearsAtCompany` |
| `TenureBand` | **4 Bands (0-2, 3-5, 6-10, 11+)** | `bins` | `YearsAtCompany` |
| `DistanceBand` | **Fixed Bins (Default: 1-5, 6-10, 11-20, 21-29)** | `bins` | `DistanceFromHome` |
| `DistanceBand` | **Empirical Quartiles** | `quantiles` | `DistanceFromHome` |
| `PromotionGapBand_Cohort` | **Default Bins (0, 1-2, 3-5, 6+)** | `bins` | `YearsSinceLastPromotion` |
| `PromotionGapBand_Cohort` | **Alternative Bins (0-1, 2-3, 4-6, 7+)** | `bins` | `YearsSinceLastPromotion` |

---

## 5. Statistical Estimation Standards

- **Proportion Intervals:** All subgroup attrition rates are presented with 95% Wilson score confidence intervals to handle small sample sizes and boundary proportions reliably.
- **Funnel Control Limits:** Organizational units (Job Roles and Departments) are plotted against exact binomial control limits at 95% (warning tier) and 99.8% (action tier) centered on the overall workforce baseline rate (16.12%).
- **Hotspot Screening Criteria:** Interaction cells are flagged as hotspot candidates only when they satisfy:
  1. Cell headcount `n >= 30`
  2. Wilson 95% lower confidence bound strictly exceeds the overall organizational baseline rate (16.12%).
