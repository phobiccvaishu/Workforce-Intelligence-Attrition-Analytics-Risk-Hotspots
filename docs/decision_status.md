# Workforce Intelligence: Decision Status Registry

**Source**: `docs/decision_log.md` (Dynamically parsed by Phase 6 pipeline)  
**Scope**: Complete register of architectural decisions (41 entries)  
**Ingestion Target**: Streamlit governance and methodology dashboard

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
