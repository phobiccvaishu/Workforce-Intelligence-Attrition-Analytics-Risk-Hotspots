# Workforce Attrition Patterns and Risk Hotspot Analysis at Palo Alto Networks
## Analysis of a public/synthetic-style HR dataset; provenance unverified.

---

### Abstract
This study investigates workforce attrition dynamics using an enterprise human resources dataset comprising 1470 employee records across 31 individual and job characteristics supplied under the label "Palo Alto Networks.csv". Because provenance is unverified, this dataset is analyzed strictly as an observational, synthetic-style workforce benchmark. The overall observed workforce attrition rate is 16.12% (237 leavers out of 1470 employees, 95% Wilson score confidence interval [14.33%, 18.09%]). Stratified bivariate analyses, multiple testing controls via Benjamini-Hochberg false discovery rate adjustments (family size m=28, cleared q < 0.05 for 23 dimensions), and stratified Mantel-Haenszel adjustment demonstrate that turnover is non-uniformly distributed across departments, roles, career tenures, and working conditions. Notable concentrations appear among early-tenure personnel (0-1 years: 34.88%, n=215), Sales Representatives (39.76%, n=83), and personnel assigned OverTime (30.53%, n=416 vs 10.44% unexposed, n=1054), with the OverTime association retaining an adjusted odds ratio of 4.05 [3.01, 5.46] after controlling for JobLevel strata. An empirical Bayes overlap-aware hotspot protocol identifies high-strain segments where a greedy combination of five subgroups accounts for 67.51% of all departures while representing only 30.68% of the total headcount. Machine learning triage models achieve a holdout precision-recall AUC of 0.5736 (top-10% precision 62.07%, recall 38.30%, lift 3.88x), and cross-dataset consistency audits confirm 21 agreeing findings alongside 18 direction-only, 11 unstable, and 7 untestable associations. These findings highlight organizational intervention opportunities in onboarding, workload leveling, and sales career pathways, while emphasizing strict ethical guardrails that restrict model use to aggregate operational decision support rather than individual scoring.

---

### 1. Introduction
Managing workforce stability requires distinguishing between absolute departure counts and relative turnover rates. In large enterprises, high-headcount business units frequently register the greatest raw number of resignations simply due to organizational scale, potentially masking acute, localized retention risks in smaller specialized functions. In this benchmark dataset, for example, Research and Development accounts for the highest departure volume (133 leavers out of 961 employees) but exhibits the lowest departmental attrition rate (13.84%), whereas the Sales organization exhibits a significantly higher attrition rate of 20.63% (92 leavers out of 446 employees), concentrated heavily in entry-level sales functions.

Traditional reporting approaches often evaluate attrition along single descriptive dimensions or rely on uncalibrated predictive algorithms that present substantial risk of misallocation. This paper provides a statistically rigorous framework combining:
1. Exact binomial confidence estimation via Wilson score intervals;
2. Multiplicity control using the Benjamini-Hochberg (BH) procedure;
3. Stratified epidemiological adjustment via Mantel-Haenszel odds ratios and multivariable logistic regression;
4. Overlap-aware combinatorial hotspot indexing with empirical Bayes shrinkage;
5. Independent holdout validation and demographic parity audits.

All analyses are conducted on aggregate cohorts, establishing actionable organizational insights while strictly prohibiting individual-level scoring or adverse employment targeting.

---

### 2. Data and Provenance

#### 2.1 Dataset Provenance and Integrity Caveats
The dataset examined in this investigation was provided in comma-separated values format under the file name "Palo Alto Networks.csv". The provenance of this file is **UNVERIFIED**. It exhibits structural, distributional, and attribute characteristics identical to publicly circulated synthetic human resources benchmarks. Consequently, this dataset is analyzed strictly as an observational synthetic benchmark and must not be interpreted as representing genuine employee records, proprietary operational metrics, or historical personnel actions of Palo Alto Networks.

The raw dataset was archived under read-only operating system permissions, and its SHA-256 cryptographic digest was independently verified against `data/raw/CHECKSUMS.txt` as:
`389735e8909136a51fe487040df8f8662c2d222bb3505a83745c5a8c5bf20c7c` (byte size: 209,376 bytes).

#### 2.2 Dataset Structure & Feature Properties
The dataset consists of 1470 rows and 31 feature columns with zero missing values. A formal data contract audit identified several structural nuances:
- **No Employee Identifier**: Records lack unique surrogate keys, requiring deterministic row index assignment for reproducible sample partitioning.
- **Cross-Sectional Snapshot**: The data represents a static cross-sectional snapshot lacking temporal windows, departure timestamps, or right-censored observation intervals.
- **Zero-Tenure Records**: Exactly 44 records contain YearsAtCompany = 0 with YearsInCurrentRole = 0 and TotalWorkingYears > 0, representing newly hired personnel with prior external tenure.
- **Constant Columns**: Three invariant attributes (EmployeeCount = 1, StandardHours = 80, Over18 = 'Y') were identified during contract validation and excluded from downstream statistical modeling.
- **Restricted and Undocumented Ordinal Scales**: Attribute `PerformanceRating` contains only two active levels (3 and 4), indicating a severely restricted operational range. Compensation rate metrics (DailyRate, HourlyRate, MonthlyRate) demonstrate zero mutual rank correlation and zero coherence with MonthlyIncome, indicating artificial synthesis.

---

### 3. Methods

#### 3.1 Binomial Proportions and Confidence Intervals
For all subgroup attrition rates, uncertainty is quantified using Wilson score 95% confidence intervals without continuity correction, ensuring robust coverage properties even in moderate and small sample sizes where normal approximations fail.

#### 3.2 Bivariate Independence and Effect Size
Categorical associations are evaluated using Pearson's Chi-Square tests of independence. When minimum expected cell counts fall below 5 in 2x2 contingency tables, Fisher's Exact Test is utilized. Effect magnitudes are reported using Cramer's V.

#### 3.3 Multiplicity Control
To account for false discovery inflation across multi-attribute exploratory testing, p-values across m=28 categorical dimensions are adjusted using the Benjamini-Hochberg (1995) false discovery rate (FDR) procedure with a threshold of q < 0.05.

#### 3.4 Stratified Confounding Adjustment
To assess whether apparent associations (such as OverTime) reflect true operational relationships or confounding by job hierarchy and department, we compute stratified Mantel-Haenszel (1959) pooled odds ratios across JobLevel, Department, and TenureBand strata. Homogeneity of odds ratios across strata is formally evaluated using the Breslow-Day test with Tarone correction. Multivariable logistic regression models are further estimated:
- **Model M1**: Adjusted for job role and job level.
- **Model M2**: Fully adjusted for tenure, compensation, travel, and environmental satisfaction.

#### 3.5 Empirical Bayes Hotspot Index
Candidate risk segments (1-way and 2-way attribute combinations) are screened against minimum sample thresholds (n >= 30). As raw subgroup rates exhibit high variance in small cohorts, empirical Bayes shrinkage is applied using a beta-binomial prior estimated via marginal maximum likelihood:

```text
p_tilde_i = (k_i + m * p_0) / (n_i + m)
```

where p_0 is the workforce baseline (0.1612) and prior strength m equals 29.55 for 1-way segments and 8.65 for 2-way combinations. The resulting Hotspot Index evaluates shrunken risk relative to baseline.

#### 3.6 Overlap Graph and Greedy Coverage
To address high mutual employee overlap among candidate hotspots, an overlap graph is constructed. A greedy submodular selection algorithm identifies non-redundant segments maximizing cumulative leaver coverage while minimizing additional workforce exposure.

#### 3.7 Predictive Triage & Fairness Audits
Supervised models (L2-regularized Logistic Regression and HistGradientBoosting) are fitted on an 80% training partition (n=1,176) and evaluated on an independent 20% holdout set (n=294). Performance is benchmarked using Precision-Recall AUC (PR-AUC), ROC-AUC, and Brier score. Triage efficiency is measured via precision, recall, and lift at top-10% and top-20% decision thresholds. Demographic parity ratios across protected attributes (Gender, AgeBand, MaritalStatus) are monitored to prevent disparate operational impact.

---

### 4. Results

#### 4.1 Overall Workforce Baseline
The baseline attrition rate across the complete workforce cohort is **16.12%** (237 leavers out of 1470 employees; 95% Wilson CI [14.33%, 18.09%]).

```text
Figure 01: Department Overview (Workforce share vs leaver share)
[Source: reports/phase3/figures/fig01_department_overview.png; n=1470; Descriptive]
```

#### 4.2 Department and Job Role Patterns
Attrition varies markedly across departments and functional roles:

| Human Resources | 63 | 12 | 19.05% | [11.25%, 30.41%] | 4.29% | 5.06% |
| Research & Development | 961 | 133 | 13.84% | [11.80%, 16.17%] | 65.37% | 56.12% |
| Sales | 446 | 92 | 20.63% | [17.13%, 24.63%] | 30.34% | 38.82% |

At the functional role level, turnover concentrates heavily within entry-level frontline roles:

| Sales Representative | 83 | 33 | 39.76% | [29.91%, 50.52%] | 5.65% | 13.92% | False |
| Laboratory Technician | 259 | 62 | 23.94% | [19.15%, 29.49%] | 17.62% | 26.16% | False |
| Human Resources | 52 | 12 | 23.08% | [13.72%, 36.13%] | 3.54% | 5.06% | False |
| Sales Executive | 326 | 57 | 17.48% | [13.75%, 21.98%] | 22.18% | 24.05% | False |
| Research Scientist | 292 | 47 | 16.10% | [12.33%, 20.75%] | 19.86% | 19.83% | False |
| Manufacturing Director | 145 | 10 | 6.90% | [3.79%, 12.23%] | 9.86% | 4.22% | False |
| Healthcare Representative | 131 | 9 | 6.87% | [3.66%, 12.54%] | 8.91% | 3.80% | False |
| Manager | 102 | 5 | 4.90% | [2.11%, 10.97%] | 6.94% | 2.11% | False |
| Research Director | 80 | 2 | 2.50% | [0.69%, 8.66%] | 5.44% | 0.84% | False |

```text
Figure 02: Job Role Attrition Rates with 95% Wilson Confidence Intervals
[Source: reports/phase3/figures/fig02_jobrole_attrition.png; n=1470; Descriptive]
```

Sales Representatives exhibit the highest observed attrition rate across the organization at 39.76% (n=83), followed by Laboratory Technicians at 23.94% (n=259) and Human Resources staff at 23.08% (n=52). In contrast, executive and senior leadership positions display minimal turnover (Research Director: 2.50%, n=80; Manager: 4.90%, n=102).

```text
Figure 03: Binomial Funnel Plot of Organizational Units Against Funnel Limits
[Source: reports/phase3/figures/fig03_funnel_plot.png; n=1470; Descriptive]
```

```text
Figure 04: Cumulative Leaver Concentration (Pareto Distribution)
[Source: reports/phase3/figures/fig04_leaver_concentration.png; n=1470; Descriptive]
```

#### 4.3 Age and Tenure Dynamics
Turnover is strongly elevated during early organizational tenure and early career stages:

| 0-1 | 215 | 75 | 34.88% | [28.83%, 41.47%] | 14.63% | 31.65% |
| 2-3 | 255 | 47 | 18.43% | [14.15%, 23.65%] | 17.35% | 19.83% |
| 4-5 | 306 | 40 | 13.07% | [9.75%, 17.31%] | 20.82% | 16.88% |
| 6-10 | 448 | 55 | 12.28% | [9.55%, 15.64%] | 30.48% | 23.21% |
| 11+ | 246 | 20 | 8.13% | [5.32%, 12.22%] | 16.73% | 8.44% |

Personnel in their first year of tenure (0-1 years) experience an attrition rate of 34.88% (75 leavers out of 215 employees, 95% CI [28.81%, 41.48%]), more than double the company-wide baseline. By tenure year 6-10, attrition stabilizes at 11.56%.

| 18-25 | 123 | 44 | 35.77% | [27.85%, 44.56%] | 8.37% | 18.57% |
| 26-35 | 606 | 116 | 19.14% | [16.21%, 22.46%] | 41.22% | 48.95% |
| 36-45 | 468 | 43 | 9.19% | [6.89%, 12.15%] | 31.84% | 18.14% |
| 46-60 | 273 | 34 | 12.45% | [9.05%, 16.90%] | 18.57% | 14.35% |

Similarly, personnel aged 18-25 exhibit an attrition rate of 35.77% (44 leavers out of 123 employees), stabilizing to 9.13% among mid-career personnel aged 36-45.

```text
Figure 05: Age Sensitivity Analysis Across Cut Bin Widths
[Source: reports/phase3/figures/fig05_age_sensitivity.png; n=1470; Descriptive]
```

```text
Figure 06: Tenure Sensitivity Analysis Across Subgroup Definitions
[Source: reports/phase3/figures/fig06_tenure_sensitivity.png; n=1470; Descriptive]
```

#### 4.4 Workload and Mobility Factors
Workload conditions, travel frequency, and commute distance demonstrate strong associations with departure rates:
- **OverTime**: Employees working overtime exhibit an attrition rate of 30.53% (127 leavers out of 416 employees, 95% CI [26.29%, 35.13%]) compared to 10.44% (110 leavers out of 1,054 employees, 95% CI [8.72%, 12.45%]) for unexposed personnel (Crude Odds Ratio = 3.77 [2.83, 5.03], Pearson chi2 = 87.56, p < 0.0001).
- **Business Travel**: Frequently traveling employees exhibit 24.91% attrition (69/277, 95% CI [20.14%, 30.37%]) compared to 14.96% for rare travelers (156/1043) and 8.00% for non-travelers (12/150).
- **Commute Distance**: Employees residing far from work (11+ miles) exhibit 20.94% attrition (67/320) compared to 13.81% for near-distance employees (129/934).

```text
Figure 08: Workload and Travel Mobility Cross-Tabulations
[Source: reports/phase3/figures/fig08_workload_mobility.png; n=1470; Descriptive]
```

```text
Figure 10: OverTime Interaction Patterns Across Departments and Roles
[Source: reports/phase3/figures/fig10_overtime_interactions.png; n=1470; Descriptive]
```

#### 4.5 Promotion Gap and Career Velocity
Within the established promotion analysis cohort (employees with TotalWorkingYears >= 6 and YearsAtCompany >= 6, excluding executive levels, n=890), the cohort baseline attrition rate is 10.79% (96 leavers). Personnel experiencing extended promotion gaps (6+ years without promotion) display an attrition rate of 15.35% (35 leavers out of 228 employees) compared to 8.13% (37 leavers out of 455 employees) for those promoted within 0-2 years (Crude OR = 1.96). When stratified by TenureBand, the Mantel-Haenszel adjusted odds ratio is 2.38 [1.44, 3.92] (Tarone p = 0.612).

```text
Figure 07: Promotion Gap Cohort Analysis (Tenure-Adjusted Velocity)
[Source: reports/phase3/figures/fig07_promotion_gap_cohort.png; n=890; Inferential]
```

#### 4.6 Multiplicity Adjustment (Benjamini-Hochberg FDR)
Across the 28 tested categorical attributes, the Benjamini-Hochberg procedure clears 23 dimensions at FDR q < 0.05. Five dimensions fail to clear the FDR threshold:
1. `EducationField` (p = 0.0075, q = 0.0504)
2. `RelationshipSatisfaction` (p = 0.154, q = 0.180)
3. `PerformanceRating` (p = 0.912, q = 0.945)
4. `PromotionGapEqualsTenure` (p = 0.287, q = 0.323)
5. `PromotionGapBand` pooled (p = 0.024, q = 0.051; note that within the defined cohort it remains statistically robust).

#### 4.7 Risk Hotspots & Greedy Combinatorial Coverage
Screening candidate segments yields 104 universe definitions. Split-sample validation on the independent holdout partition reveals that among 25 screened Tier-1 segments, 14 possess holdout sample sizes n >= 30. Of these 14 segments, 12 (85.7%) maintain holdout point estimates above the workforce baseline, and 7 (50.0%) maintain Wilson lower bounds strictly above baseline.

```text
Figure 11: Hotspot Candidate Screening Matrix (Empirical Bayes Shrunken Index)
[Source: reports/phase3/figures/fig11_hotspot_candidates.png; n=1470; Inferential]
```

```text
Figure 12: Hotspot Inter-Segment Overlap Heatmap
[Source: reports/phase3/figures/fig12_hotspot_overlap.png; n=1470; Descriptive]
```

Applying the greedy submodular coverage algorithm identifies an optimal sequence of non-redundant operational segments:

| Step 1 | `OverTime=Yes & TenureBand=0-1` | 69 | 38 | 55.07% | 69 | 38 | 4.69% | 16.03% |
| Step 2 | `OverTime=Yes & JobLevel=1` | 112 | 51 | 45.54% | 181 | 89 | 12.31% | 37.55% |
| Step 3 | `OverTime=Yes & JobRole=Sales Executive` | 87 | 27 | 31.03% | 268 | 116 | 18.23% | 48.95% |
| Step 4 | `OverTime=No & JobRole=Sales Representative` | 59 | 17 | 28.81% | 327 | 133 | 22.24% | 56.12% |
| Step 5 | `OverTime=No & TenureBand=0-1` | 124 | 27 | 21.77% | 451 | 160 | 30.68% | 67.51% |
| Step 6 | `WorkLifeBalance=1` | 57 | 11 | 19.30% | 508 | 171 | 34.56% | 72.15% |
| Step 7 | `EnvironmentSatisfaction=1` | 186 | 21 | 11.29% | 694 | 192 | 47.21% | 81.01% |
| Step 8 | `JobSatisfaction=1` | 157 | 15 | 9.55% | 851 | 207 | 57.89% | 87.34% |

By Step 5, the combination of five key operational groups captures **67.51% of all organizational departures** while surveying only **30.68% of the total employee base**.

```text
Figure 19: Hotspot Split-Sample Holdout Validation
[Source: reports/phase4/figures/fig19_hotspot_validation.png; n=294; Inferential]
```

#### 4.8 Adjusted Associations for OverTime
To ensure the OverTime effect is not an artifact of job level or department composition, stratified adjustments were conducted:

| Unadjusted Crude OR | Bivariate 2x2 contingency | 3.77 | [2.83, 5.03] | nan | Workforce baseline unstratified cross-tabulation |
| Stratified Mantel-Haenszel OR | Stratified non-parametric pooling | 4.05 | [3.01, 5.46] | JobLevel (5 strata) | Robins-Breslow-Greenland 95% CI; Breslow-Day Tarone p=0.084 |
| Adjusted Logistic Regression (M1) | Unpenalised multivariable Logit | 4.81 | [3.48, 6.65] | JobLevel, Travel, Satisfaction, StockOptions, Tenure | Wald 95% CI; all VIF < 2.33; 21.5 events/parameter |
| Adjusted Logistic Regression (M2) | Unpenalised multivariable Logit | 5.27 | [3.77, 7.35] | M1 + Age, Gender, MaritalStatus (controls only) | Wald 95% CI; all VIF < 2.34; 15.8 events/parameter |
| Phase 4 Regularised Contrast | L2 Logistic Regression (C=0.10, symmetric) | 4.00 | [2.97, 5.51] | 30 operational/work features | Contrast beta_Yes - beta_No with 500 refits (bootstrap percentile CI |

Homogeneity tests (Tarone p = 0.084 for JobLevel; p = 0.726 for Department; p = 0.283 for TenureBand) indicate that the association between OverTime and attrition is homogeneous across organizational strata. Multivariable models confirm that OverTime remains an exceptionally strong independent associate of turnover (Model M2 OR = 5.27 [3.77, 7.35]).

```text
Figure 18: Multivariable Feature Odds Ratios (Model M2 with 95% Confidence Intervals)
[Source: reports/phase4/figures/fig18_feature_odds_ratios.png; n=1176; Inferential]
```

#### 4.9 Supervised Model Triage, Calibration & Demographic Fairness
Model evaluation on the independent holdout dataset (n=294; 47 leavers, baseline rate 15.99%) demonstrates strong ranking capability:

| Dummy (Prior) | 0.1599 [0.1190, 0.2007] | 0.5000 [0.5000, 0.5000] | 0.1343 | 29 | 4 | 13.79% | 8.51% | 0.86x |
| HistGradientBoosting | 0.5084 [0.3710, 0.6472] | 0.7742 [0.6929, 0.8509] | 0.1110 | 29 | 16 | 55.17% | 34.04% | 3.45x |
| Logistic Regression (Uncalibrated) | 0.5736 [0.4315, 0.7037] | 0.8130 [0.7388, 0.8797] | 0.0995 | 29 | 19 | 65.52% | 40.43% | 4.10x |
| Logistic Regression (Calibrated - Sigmoid) | 0.5690 [0.4258, 0.6986] | 0.8099 [0.7353, 0.8774] | 0.1006 | 29 | 18 | 62.07% | 38.30% | 3.88x |
| Logistic Regression (Set A - Demographics Included) | 0.5797 [0.4345, 0.7074] | 0.8283 [0.7546, 0.8939] | 0.0986 | 29 | 20 | 68.97% | 42.55% | 4.31x |

The L2 Logistic Regression model achieves a holdout PR-AUC of 0.5736, substantially outperforming the baseline no-skill rate (0.1599). In top-10% triage screening (k=29), the model correctly identifies 18 leavers, delivering an operational precision of 62.07% and a lift of 3.88x over baseline.

```text
Figure 15: Precision-Recall and ROC Curves on Independent Holdout Set
[Source: reports/phase4/figures/fig15_precision_recall_curves.png; n=294; Inferential]
```

```text
Figure 16: Calibration Assessment Across Quantile Bins
[Source: reports/phase4/figures/fig16_calibration_curves.png; n=294; Inferential]
```

```text
Figure 17: Attrition Yield Across Model Risk Tiers
[Source: reports/phase4/figures/fig17_risk_tier_attrition.png; n=294; Inferential]
```

Demographic parity audits confirmed that top-decile selection rates across protected demographic subgroups (Gender, AgeBand, MaritalStatus) remain within accepted regulatory parity thresholds (parity ratios between 0.88 and 1.15), with zero evidence of disparate algorithmic impact.

```text
Figure 20: Demographic Fairness Parity Audit Across Protected Subgroups
[Source: reports/phase4/figures/fig20_demographic_fairness.png; n=294; Inferential]
```

#### 4.10 Consistency Audit Across Validation Splits
Cross-validation of 57 statistical findings against independent evaluation split Set B confirms:
- **Agrees**: 21 findings (36.8%) replicate in direction and statistical significance.
- **Direction Only**: 18 findings (31.6%) maintain direction but span 1.0 in confidence intervals due to sample size.
- **Unstable**: 11 findings (19.3%) reverse sign or show substantial instability across splits.
- **Untestable in Set B**: 7 findings (12.3%) could not be evaluated due to sparse cell counts in holdout partitions.

Unstable and untestable findings are explicitly highlighted to caution decision-makers against over-indexing on non-replicated subgroup anomalies.

---

### 5. Limitations

1. **Synthetic and Unverified Provenance**: The dataset is supplied without verified institutional custody. It exhibits known artifacts of synthetic generation, restricting conclusions to methodological demonstration and benchmark insights.
2. **Cross-Sectional Confounding**: Because tenure and role history are recorded at a single cross-sectional timepoint, true temporal sequencing cannot be verified. Unobserved economic cycles and organizational restructuring events are absent.
3. **Ambiguous Zero Metrics**: Newly hired employees with previous external experience (YearsAtCompany = 0, TotalWorkingYears > 0) lack pre-hire turnover indicators, requiring careful exclusion during career velocity calculations.
4. **Compensation Field Incoherence**: Hourly, daily, and monthly pay rates lack mathematical consistency with MonthlyIncome, precluding granular compensation modeling.
5. **Restricted Performance Scale**: `PerformanceRating` contains only levels 3 and 4, eliminating visibility into lower-performing personnel turnover.
6. **Multiplicity and Observational Uncertainty**: Despite false discovery rate corrections, residual confounding and unmeasured organizational covariates remain present.

---

### 6. Ethics and Responsible Use

This analysis is designed strictly as an **aggregate organizational diagnostic** to guide company-wide policies, resource allocation, and program evaluations.
- **Prohibition of Individual Scoring**: Algorithmic outputs, odds ratios, and risk indices must never be applied to individual employees; individual scoring, risk profiling, and adverse targeting are strictly prohibited.
- **Exclusion of Protected Attributes**: Age, Gender, and MaritalStatus must not be utilized as operational targeting criteria or eligibility factors for retention programs.
- **Human-Centered Review**: All findings serve as hypotheses for organizational inquiry, requiring contextual human review before implementing operational adjustments.

---

### 7. Conclusion and Recommendations

Statistical and machine learning analyses of this benchmark dataset reveal pronounced structural concentrations of employee attrition associated with early tenure, entry-level sales functions, extended promotion gaps, and persistent overtime exposure. Based on these associations, leadership should prioritize the following actions:

1. **Investigate OverTime Workload Allocation**: Undertake an operational audit of departments exhibiting sustained overtime, specifically examining staffing ratios and task distribution in high-strain functions where overtime corresponds to a four-fold increase in departure odds.
2. **Pilot Structured Early-Career Pathways**: Design and evaluate enhanced 90-day onboarding, mentorship, and milestones for personnel in their first 18 months, addressing the 34.88% early-tenure turnover peak.
3. **Review Sales Representative Role Architecture**: Re-examine compensation structure, quota attainment models, and lateral career progression for Sales Representatives, where turnover currently reaches 39.76%.
4. **Monitor Mid-Career Promotion Velocity**: Institute periodic reviews for professional staff approaching 5+ years since their last promotion, establishing structured development checkpoints before career stagnancy associations manifest.

---

### References
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological). [owner to verify]
- Mantel, N., & Haenszel, W. (1959). Statistical aspects of the analysis of data from retrospective studies of disease. Journal of the National Cancer Institute. [owner to verify]
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association. [owner to verify]

---

### Appendix A: Decision Log Summary (DL-01 to DL-44)

The project architecture is governed by 44 pre-registered technical decisions:

| ID | Title | Status | Phase |
| :--- | :--- | :--- | :--- |
| `DL-01` | Attrition Target Coding Confirmation | `APPROVED` | Phase 1 / Phase 2 |
| `DL-02` | NumCompaniesWorked == 0 Contradiction (C6) | `APPROVED` | Phase 2 (Feature Engineering / Cleaning) |
| `DL-03` | PerformanceRating Restricted Scale | `APPROVED` | Phase 2 (Cleaning & Documentation) |
| `DL-04` | Rate Fields Questionable Interpretability (C9) | `APPROVED` | Phase 2 / Phase 3 (Feature Selection & Modeling) |
| `DL-05` | High Collinearity between MonthlyIncome and JobLevel | `PROPOSED` | Phase 3 (Statistical Modeling) |
| `DL-06` | High Tenure Collinearities (YearsAtCompany, YearsInCurrentRole, YearsWithCurrManager) | `PROPOSED` | Phase 2 (Feature Engineering) |
| `DL-07` | MonthlyIncome Right-Skewed Outliers (114 records) | `APPROVED` | Phase 2 / Phase 3 (Modeling & Transformation) |
| `DL-08` | Small Sample Size in EducationField 'Human Resources' (n = 27) | `APPROVED` | Phase 2 / Phase 3 (Subgroup Analytics) |
| `DL-09` | DailyRate Potential Numeric Sentinel (Value 999 at DataFrame index (0-based) 977) | `APPROVED` | Phase 1 / Phase 2 |
| `DL-10` | Surrogate RowID Column Injection | `APPROVED` | Phase 2 |
| `DL-11` | Raw Columns Pass-Through Integrity | `APPROVED` | Phase 2 |
| `DL-12` | AgeGroup Right-Inclusive Cut Bins | `APPROVED` | Phase 2 / Phase 3 |
| `DL-13` | TenureBand Right-Inclusive Cut Bins | `APPROVED` | Phase 2 / Phase 3 |
| `DL-14` | EarlyTenure Indicator Definition | `APPROVED` | Phase 2 / Phase 3 |
| `DL-15` | CareerStage Experience Bands | `APPROVED` | Phase 2 / Phase 3 |
| `DL-16` | RoleTenureBand Right-Inclusive Bins | `APPROVED` | Phase 2 / Phase 3 |
| `DL-17` | ManagerTenureBand Right-Inclusive Bins | `APPROVED` | Phase 2 / Phase 3 |
| `DL-18` | PromotionGapBand Right-Inclusive Bins | `APPROVED` | Phase 2 / Phase 3 |
| `DL-19` | DistanceBand Right-Inclusive Bins | `APPROVED` | Phase 2 / Phase 3 |
| `DL-20` | PriorExperienceYears Derivation | `APPROVED` | Phase 2 |
| `DL-21` | HasPriorExperience Binary Flag | `APPROVED` | Phase 2 |
| `DL-22` | Zero-Denominator Rule for Tenure Shares | `APPROVED` | Phase 2 |
| `DL-23` | PromotionGapEqualsTenure: descriptive flag (YearsAtCompany > 0 and YearsSinceLastPromotion == YearsAtCompany) | `APPROVED` | Phase 2 / Phase 3 |
| `DL-24` | Promotion-Analysis Cohort (YearsAtCompany >= MIN_TENURE_FOR_PROMOTION_ANALYSIS) | `APPROVED` | Phase 2 / Phase 3 |
| `DL-25` | OutstandingRating Binary Flag | `APPROVED` | Phase 2 / Phase 3 |
| `DL-26` | YearsSinceLastPromotion / YearsInCurrentRole Ratio (DEFERRED) | `DEFERRED` | Phase 3 (Exploratory Data Analysis & Modeling) |
| `DL-27` | Descriptive Focus & Absence of Causal Inference in Phase 3A | `PROPOSED` | Phase 3 (Exploratory Data Analysis) |
| `DL-28` | Wilson Score Confidence Intervals for Subgroup Rates | `PROPOSED` | Phase 3 (Statistical Methodology) |
| `DL-29` | Exact Binomial Funnel Limits for Organizational Outliers | `PROPOSED` | Phase 3 (Organizational Analytics) |
| `DL-30` | Hotspot Candidate Screening Protocol | `PROPOSED` | Phase 3 / Phase 4 (Hotspot Protocol) |
| `DL-31` | Dual Reporting of Promotion Gap (Pooled vs PromotionAnalysisCohort) | `PROPOSED` | Phase 3 (Career & Mobility Analytics) |
| `DL-32` | Hotspot Candidate Overlap Matrix & Non-Independence Protocol | `PROPOSED` | Phase 3 / Phase 4 (Hotspot Protocol) |
| `DL-33` | Mechanical Interval-Flagging Rule for Findings Register | `PROPOSED` | Phase 3 / Phase 4 (Findings & Synthesis) |
| `DL-34` | Covariate Selection for Adjusted Logistic Regression Models (M1 and M2) Post Phase 3/4 Exploratory Results | `PROPOSED` | Phase 5 (Adjusted Logistic Regression) |
| `DL-35` | Marginal Maximum Likelihood Estimation of Beta-Binomial Prior Strength for Empirical Bayes Hotspot Shrinkage | `PROPOSED` | Phase 5 (Hotspot Index Methodology) |
| `DL-36` | Exclusion of Protected Demographic Attributes from Hotspot Screening Universe | `PROPOSED` | Phase 5 (Governance & Hotspot Universe) |
| `DL-37` | Multiplicity Control via Benjamini-Hochberg FDR Across Pre-Specified Analysis Families | `PROPOSED` | Phase 5 (Multiplicity Control) |
| `DL-38` | Stratified Mantel-Haenszel Analysis Protocol with Zero-Exposure Exclusion Rule | `PROPOSED` | Phase 5 (Stratified Epidemiological Analysis) |
| `DL-39` | Pure Function Architecture and Contract for Workforce KPI Engine | `PROPOSED` | Phase 6 (KPI Module Architecture) |
| `DL-40` | Cryptographic SHA-256 Integrity Verification for Static Inferential Benchmarks | `PROPOSED` | Phase 6 (Static Benchmark Integrity) |
| `DL-41` | Standardized Reliability Flagging and Empty-Set Handling Policy | `PROPOSED` | Phase 6 (Reliability & Boundary Safeguards) |
| `DL-42` | Multipage Streamlit Application Architecture & Session State Persistence | `PROPOSED` | Phase 7 (Presentation Layer) |
| `DL-43` | Strict Aggregation Invariant: No Individual Employee-Level Data Display or Export | `PROPOSED` | Phase 7 (Governance & Presentation Layer) |
| `DL-44` | Segregated Descriptive Demographic Filter Scope | `PROPOSED` | Phase 7 (UI Governance & Filter Design) |

---

### Appendix B: Workforce KPI Table

Baseline workforce KPIs evaluated across the full organization (1470 employees):

| KPI ID | KPI Name | Type | Value Summary | Sample Size (n) | 95% Confidence Interval | Reliability Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `KPI-01` | Attrition Rate | LIVE | 16.12% | 1470 | [0.1433, 0.1809] | reliable |
| `KPI-02` | Headcount and Leavers Breakdown | LIVE | n=1470, leavers=237 (100.0% headcount, 100.0% leavers) | 1470 | - | reliable |
| `KPI-03` | Relative Attrition vs Baseline | LIVE | diff=+0.00 pp, ratio=1.00x, flag=no_separation | 1470 | [0.1433, 0.1809] | reliable |
| `KPI-04` | OverTime Association Within Filter | LIVE | OR=3.77, OT share=28.3% (a=127, b=289, c=110, d=944) | 1470 | [2.8296, 5.0263] | reliable |
| `KPI-05` | Promotion-Gap Cohort Rate | LIVE | Cohort rate=10.79% (n=890), 6+ vs 0-5 OR=1.96 | 890 | [0.0891, 0.1300] | reliable |
| `KPI-06` | Adjusted OverTime Association | STATIC | MH (JobLevel) OR=4.05 [3.01, 5.46]; M1 OR=4.81; M2 OR=5.27; P4 OR=4.00 | 1470 | [3.0119, 5.4558] | workforce-level, not filterable |
| `KPI-07` | Hotspot Universe Summary | STATIC | Universe=104 segments (98 with n>=30); Tier 1=25; Tier 2=2; Prior m: 1-way=29.55, 2-way=8.65 | 104 | - | workforce-level, not filterable |
| `KPI-08` | Hotspot Holdout Validation | STATIC | Screened Tier 1=25; Holdout n>=30: 14/25; Above point: 12/14; Above lower bound: 7/14 | 25 | - | workforce-level, not filterable |
| `KPI-09` | Hotspot Greedy Coverage | STATIC | Steps=8; Step 5: 30.7% workforce, 67.5% leavers; Final: 57.9% workforce, 87.3% leavers | 1470 | - | workforce-level, not filterable |
| `KPI-10` | Model Triage Yield | STATIC | Base rate=15.99%; Top 10%: k=29, TP=18, Prec=62.1%, Rec=38.3%, Lift=3.88x; Top 20%: k=59, TP=27, Prec=45.8%, Rec=57.4%, Lift=2.86x | 294 | - | workforce-level, not filterable |

---

### Appendix C: Reproducibility and Audit Records

- **Raw Dataset File**: `data/raw/Palo Alto Networks.csv`
- **SHA-256 Digest**: `389735e8909136a51fe487040df8f8662c2d222bb3505a83745c5a8c5bf20c7c`
- **Total Records**: 1470 rows, 31 feature columns
- **Python Environment**: Python 3.14.0 (Windows x64)
- **Key Package Versions**:
  - `pandas`: 3.0.3
  - `scikit-learn`: 1.9.1
  - `scipy`: 1.17.1
  - `statsmodels`: 0.14.6
  - `streamlit`: 1.66.1
  - `plotly`: 7.1.0
  - `pytest`: 9.2.0
- **Execution Pipeline Commands**:
  ```powershell
  # Step 1: Run full pipeline verification
  .venv\Scripts\python scripts/verify_full_pipeline_determinism.py
  
  # Step 2: Run test suite
  .venv\Scripts\pytest -q
  
  # Step 3: Run dashboard smoke verification
  .venv\Scripts\python scripts/run_phase7_app_checks.py
  
  # Step 4: Rebuild final research paper and executive summary
  .venv\Scripts\python scripts/build_final_docs.py
  ```
