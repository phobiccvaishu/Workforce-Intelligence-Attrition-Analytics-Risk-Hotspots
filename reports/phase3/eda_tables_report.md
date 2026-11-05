# Phase 3A Exploratory Data Analysis Report
## Descriptive Attrition Rate Tables, Sensitivity Checks & Risk Hotspot Screening

- **Execution Timestamp (UTC):** 2026-11-05T19:37:31.471774+00:00
- **Dataset File:** `workforce_attrition_processed.csv`
- **Processed Dataset SHA-256:** `48553275587384b98f091dacfd4466cadb9bddda131fe32bbd586e0b15bacb2d`
- **Workforce Dimensions:** 1,470 employees across 47 columns
- **Runtime Environment:** Python 3.14.0, pandas 3.0.3, numpy 2.4.4

> [!IMPORTANT]
> **Governance & Descriptive Standard:** This report presents strictly descriptive calculations (counts, observed rates, and statistical confidence intervals). In compliance with analytical guidelines, this document makes no causal claims and asserts no directional attributions. Protected demographic attributes are reported solely for aggregate equity monitoring and are never to be utilized for individual employee assessment.

---

## 1. Overall Workforce Attrition Baseline

Across the entire monitored workforce of 1,470 employees:
- **Total Leavers:** 237 employees departed the organization.
- **Total Retained:** 1,233 employees remained active.
- **Baseline Attrition Rate:** **16.1%** (95% Wilson Score CI: **[14.3%, 18.1%]**).
- **Workforce Imbalance Ratio:** 5.20 retained employees per departing employee.

---

## 2. Organizational Hierarchy Breakdowns

### 2.1 Department Breakdown
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Human Resources | 63 | 12 | 51 | 19.0% | [11.2%, 30.4%] | 4.3% | 5.1% | OK |
| Research & Development | 961 | 133 | 828 | 13.8% | [11.8%, 16.2%] | 65.4% | 56.1% | OK |
| Sales | 446 | 92 | 354 | 20.6% | [17.1%, 24.6%] | 30.3% | 38.8% | OK |

### 2.2 Job Role Breakdown
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Healthcare Representative | 131 | 9 | 122 | 6.9% | [3.7%, 12.5%] | 8.9% | 3.8% | OK |
| Human Resources | 52 | 12 | 40 | 23.1% | [13.7%, 36.1%] | 3.5% | 5.1% | OK |
| Laboratory Technician | 259 | 62 | 197 | 23.9% | [19.1%, 29.5%] | 17.6% | 26.2% | OK |
| Manager | 102 | 5 | 97 | 4.9% | [2.1%, 11.0%] | 6.9% | 2.1% | OK |
| Manufacturing Director | 145 | 10 | 135 | 6.9% | [3.8%, 12.2%] | 9.9% | 4.2% | OK |
| Research Director | 80 | 2 | 78 | 2.5% | [0.7%, 8.7%] | 5.4% | 0.8% | OK |
| Research Scientist | 292 | 47 | 245 | 16.1% | [12.3%, 20.7%] | 19.9% | 19.8% | OK |
| Sales Executive | 326 | 57 | 269 | 17.5% | [13.7%, 22.0%] | 22.2% | 24.1% | OK |
| Sales Representative | 83 | 33 | 50 | 39.8% | [29.9%, 50.5%] | 5.6% | 13.9% | OK |

### 2.3 Job Level Breakdown
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 543 | 143 | 400 | 26.3% | [22.8%, 30.2%] | 36.9% | 60.3% | OK |
| 2 | 534 | 52 | 482 | 9.7% | [7.5%, 12.5%] | 36.3% | 21.9% | OK |
| 3 | 218 | 32 | 186 | 14.7% | [10.6%, 20.0%] | 14.8% | 13.5% | OK |
| 4 | 106 | 5 | 101 | 4.7% | [2.0%, 10.6%] | 7.2% | 2.1% | OK |
| 5 | 69 | 5 | 64 | 7.2% | [3.1%, 15.9%] | 4.7% | 2.1% | OK |

---

## 3. Binomial Funnel Limits & Organizational Disproportion

Exact binomial control limits evaluate observed organizational subgroup rates against expected random variation around the workforce baseline (16.1%):

| Dimension | Segment | Headcount (n) | Leavers | Attrition Rate (%) | 95% Limits (%) | 95% Status | 99.8% Limits (%) | 99.8% Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| JobRole | Healthcare Representative | 131 | 9 | 6.9% | [9.9%, 22.9%] | `Below` | [6.9%, 26.7%] | `Below` |
| JobRole | Human Resources | 52 | 12 | 23.1% | [7.7%, 26.9%] | `Inside` | [1.9%, 32.7%] | `Inside` |
| JobRole | Laboratory Technician | 259 | 62 | 23.9% | [12.0%, 20.8%] | `Above` | [9.7%, 23.6%] | `Above` |
| JobRole | Manager | 102 | 5 | 4.9% | [9.8%, 23.5%] | `Below` | [5.9%, 28.4%] | `Below` |
| JobRole | Manufacturing Director | 145 | 10 | 6.9% | [10.3%, 22.1%] | `Below` | [7.6%, 26.2%] | `Below` |
| JobRole | Research Director | 80 | 2 | 2.5% | [8.8%, 25.0%] | `Below` | [5.0%, 30.0%] | `Below` |
| JobRole | Research Scientist | 292 | 47 | 16.1% | [12.0%, 20.5%] | `Inside` | [9.9%, 22.9%] | `Inside` |
| JobRole | Sales Executive | 326 | 57 | 17.5% | [12.3%, 20.2%] | `Inside` | [10.1%, 22.7%] | `Inside` |
| JobRole | Sales Representative | 83 | 33 | 39.8% | [8.4%, 24.1%] | `Above` | [4.8%, 30.1%] | `Above` |
| Department | Human Resources | 63 | 12 | 19.0% | [7.9%, 25.4%] | `Inside` | [3.2%, 31.7%] | `Inside` |
| Department | Research & Development | 961 | 133 | 13.8% | [13.8%, 18.5%] | `Inside` | [12.6%, 19.9%] | `Inside` |
| Department | Sales | 446 | 92 | 20.6% | [12.8%, 19.5%] | `Above` | [11.0%, 21.7%] | `Inside` |

---

## 4. Workload, Mobility & Tenure Patterns

### 4.1 Overtime Status
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No | 1,054 | 110 | 944 | 10.4% | [8.7%, 12.4%] | 71.7% | 46.4% | OK |
| Yes | 416 | 127 | 289 | 30.5% | [26.3%, 35.1%] | 28.3% | 53.6% | OK |

### 4.2 Business Travel Frequency
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Non-Travel | 150 | 12 | 138 | 8.0% | [4.6%, 13.5%] | 10.2% | 5.1% | OK |
| Travel_Frequently | 277 | 69 | 208 | 24.9% | [20.2%, 30.3%] | 18.8% | 29.1% | OK |
| Travel_Rarely | 1,043 | 156 | 887 | 15.0% | [12.9%, 17.2%] | 71.0% | 65.8% | OK |

### 4.3 Distance Band (Commute)
*Note: Commute distance units are undocumented in the source data.*
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1-5 | 632 | 87 | 545 | 13.8% | [11.3%, 16.7%] | 43.0% | 36.7% | OK |
| 6-10 | 394 | 57 | 337 | 14.5% | [11.3%, 18.3%] | 26.8% | 24.1% | OK |
| 11-20 | 240 | 48 | 192 | 20.0% | [15.4%, 25.5%] | 16.3% | 20.3% | OK |
| 21-29 | 204 | 45 | 159 | 22.1% | [16.9%, 28.2%] | 13.9% | 19.0% | OK |

### 4.4 Company Tenure Band
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0-1 | 215 | 75 | 140 | 34.9% | [28.8%, 41.5%] | 14.6% | 31.6% | OK |
| 2-3 | 255 | 47 | 208 | 18.4% | [14.2%, 23.6%] | 17.3% | 19.8% | OK |
| 4-5 | 306 | 40 | 266 | 13.1% | [9.7%, 17.3%] | 20.8% | 16.9% | OK |
| 6-10 | 448 | 55 | 393 | 12.3% | [9.6%, 15.6%] | 30.5% | 23.2% | OK |
| 11+ | 246 | 20 | 226 | 8.1% | [5.3%, 12.2%] | 16.7% | 8.4% | OK |

### 4.5 Early Tenure Cohort (<= 1 Year vs > 1 Year)
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 1,255 | 162 | 1,093 | 12.9% | [11.2%, 14.9%] | 85.4% | 68.4% | OK |
| 1 | 215 | 75 | 140 | 34.9% | [28.8%, 41.5%] | 14.6% | 31.6% | OK |

### 4.6 Promotion Gap Band (Within Promotion Analysis Cohort, n = 890)
*Evaluated exclusively for employees with company tenure >= 5 years to separate career stagnation from initial onboarding tenure.*
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 266 | 23 | 243 | 8.6% | [5.8%, 12.6%] | 29.9% | 24.0% | OK |
| 1-2 | 257 | 23 | 234 | 8.9% | [6.0%, 13.1%] | 28.9% | 24.0% | OK |
| 3-5 | 152 | 15 | 137 | 9.9% | [6.1%, 15.6%] | 17.1% | 15.6% | OK |
| 6+ | 215 | 35 | 180 | 16.3% | [11.9%, 21.8%] | 24.2% | 36.5% | OK |

---

## 5. Demographic Cohorts (Descriptive Monitoring Only)

> [!NOTE]
> Demographic characteristics are analyzed solely for aggregate organizational parity benchmarking.

### 5.1 Age Group Distribution
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 18-25 | 123 | 44 | 79 | 35.8% | [27.9%, 44.6%] | 8.4% | 18.6% | OK |
| 26-35 | 606 | 116 | 490 | 19.1% | [16.2%, 22.5%] | 41.2% | 48.9% | OK |
| 36-45 | 468 | 43 | 425 | 9.2% | [6.9%, 12.1%] | 31.8% | 18.1% | OK |
| 46-60 | 273 | 34 | 239 | 12.5% | [9.1%, 16.9%] | 18.6% | 14.3% | OK |

### 5.2 Gender Distribution
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Female | 588 | 87 | 501 | 14.8% | [12.2%, 17.9%] | 40.0% | 36.7% | OK |
| Male | 882 | 150 | 732 | 17.0% | [14.7%, 19.6%] | 60.0% | 63.3% | OK |

### 5.3 Marital Status Distribution
| segment | Headcount (n) | Leavers | Retained | Attrition Rate (%) | 95% Wilson CI (%) | Workforce Share (%) | Leaver Share (%) | Flag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Divorced | 327 | 33 | 294 | 10.1% | [7.3%, 13.8%] | 22.2% | 13.9% | OK |
| Married | 673 | 84 | 589 | 12.5% | [10.2%, 15.2%] | 45.8% | 35.4% | OK |
| Single | 470 | 120 | 350 | 25.5% | [21.8%, 29.7%] | 32.0% | 50.6% | OK |

---

## 6. Pre-Specified Risk Hotspot Candidates Scan

Across 58 scanned two-way interaction cells, exactly 15 subgroups met the dual pre-specified qualification criteria:
1. Headcount `n >= 30`
2. Wilson 95% lower confidence bound strictly exceeding the organizational baseline (16.1%).

> [!WARNING]
> **Preliminary Screening Notice:** These interaction cells represent *candidates for formal multivariable statistical testing in Phase 4*, not validated independent factors.

| Candidate Subgroup | Interaction | Headcount (n) | Leavers | Attrition Rate (%) | 95% Wilson CI (%) | Diff vs Overall (pp) | Rate Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `OverTime=Yes & AgeGroup=18-25` | OverTime x AgeGroup | 39 | 25 | 64.1% | [48.4%, 77.3%] | +48.0 pp | 3.98x |
| `OverTime=Yes & TenureBand=0-1` | OverTime x TenureBand | 69 | 38 | 55.1% | [43.4%, 66.2%] | +39.0 pp | 3.42x |
| `OverTime=Yes & JobLevel=1` | OverTime x JobLevel | 156 | 82 | 52.6% | [44.8%, 60.2%] | +36.4 pp | 3.26x |
| `OverTime=Yes & JobRole=Laboratory Technician` | OverTime x JobRole | 62 | 31 | 50.0% | [37.9%, 62.1%] | +33.9 pp | 3.10x |
| `OverTime=Yes & BusinessTravel=Travel_Frequently` | OverTime x BusinessTravel | 86 | 36 | 41.9% | [32.0%, 52.4%] | +25.7 pp | 2.60x |
| `OverTime=Yes & AgeGroup=26-35` | OverTime x AgeGroup | 159 | 62 | 39.0% | [31.8%, 46.7%] | +22.9 pp | 2.42x |
| `OverTime=Yes & TenureBand=2-3` | OverTime x TenureBand | 75 | 29 | 38.7% | [28.5%, 50.0%] | +22.5 pp | 2.40x |
| `OverTime=Yes & Department=Sales` | OverTime x Department | 128 | 48 | 37.5% | [29.6%, 46.1%] | +21.4 pp | 2.33x |
| `OverTime=Yes & JobRole=Research Scientist` | OverTime x JobRole | 97 | 33 | 34.0% | [25.4%, 43.9%] | +17.9 pp | 2.11x |
| `OverTime=Yes & JobRole=Sales Executive` | OverTime x JobRole | 94 | 31 | 33.0% | [24.3%, 43.0%] | +16.9 pp | 2.05x |
| `OverTime=No & JobRole=Sales Representative` | OverTime x JobRole | 59 | 17 | 28.8% | [18.8%, 41.4%] | +12.7 pp | 1.79x |
| `OverTime=Yes & BusinessTravel=Travel_Rarely` | OverTime x BusinessTravel | 295 | 84 | 28.5% | [23.6%, 33.9%] | +12.4 pp | 1.77x |
| `OverTime=Yes & Department=Research & Development` | OverTime x Department | 271 | 74 | 27.3% | [22.3%, 32.9%] | +11.2 pp | 1.69x |
| `OverTime=Yes & TenureBand=6-10` | OverTime x TenureBand | 120 | 31 | 25.8% | [18.8%, 34.3%] | +9.7 pp | 1.60x |
| `OverTime=No & TenureBand=0-1` | OverTime x TenureBand | 146 | 37 | 25.3% | [19.0%, 33.0%] | +9.2 pp | 1.57x |

---

## 7. Artifact Manifest & Verification Tables

- **Summary Metadata:** `reports/phase3/eda_summary.json`
- **Formal Analysis Plan:** `docs/analysis_plan.md`
- **Output CSV Tables:** Exactly 45 CSV tables stored in `reports/phase3/tables/` with 4-decimal precision (`ROUND_DIGITS = 4`) and LF line endings.
