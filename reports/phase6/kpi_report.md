# Phase 6 Report: Workforce KPI Module & Contract Validation

**Author**: Workforce Intelligence Engineering & Governance Team  
**Evaluation Target**: Pure-Function KPI Engine (10 Metrics)  
**Execution Environment**: Isolated Virtual Environment (.venv)  
**Baseline Population**: n = 1,470 employees, 237 leavers (16.12% baseline attrition)

---

## 1. Executive Summary

Phase 6 implements and validates the standalone, deterministic Key Performance Indicator (KPI) engine that will power the interactive Phase 7 analytics interface. The engine provides 10 pre-registered metrics adhering strictly to a standardized data contract:

\[
\\{ \\text{kpi\\_id}, \\text{name}, \\text{value}, n, \\text{ci\\_low}, \\text{ci\\_high}, \\text{reliability}, \\text{source}, \\text{as\\_of\\_filter} \\}
\]

Key verification outcomes:
1. **Contract Uniformity**: Every KPI returns the complete required schema. Rates are coupled with denominators (\(n\)) and 95% intervals where defined.
2. **Cryptographic Static Integrity**: Static benchmarks (KPIs 6-10) verify source table SHA-256 hashes upon initialization, rejecting ad-hoc filters with an explicit error.
3. **Robust Boundary Handling**: Empty filter results (\(n = 0\)) return `reliability = "no_data"` and `value = None` without runtime exceptions; small samples (\(n < 30\)) return flagged point estimates without hiding data.

---

## 2. Workforce Baseline KPI Summary (Unfiltered Population, n = 1,470)

The table below reports all 10 performance indicators computed across the unfiltered workforce baseline:

| KPI ID | Metric Name | Class | Baseline Value Summary | Base n | 95% Confidence Interval | Reliability Status | Source Path / Function |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `KPI-01` | **Attrition Rate** | `LIVE` | 16.12% | 1470 | [0.1433, 0.1809] | `reliable` | `src.workforce_intel.kpis.kpi_01_attrition_rate` |
| `KPI-02` | **Headcount and Leavers Breakdown** | `LIVE` | n=1470, leavers=237 (100.0% headcount, 100.0% leavers) | 1470 | - | `reliable` | `src.workforce_intel.kpis.kpi_02_headcount_leavers` |
| `KPI-03` | **Relative Attrition vs Baseline** | `LIVE` | diff=+0.00 pp, ratio=1.00x, flag=no_separation | 1470 | [0.1433, 0.1809] | `reliable` | `src.workforce_intel.kpis.kpi_03_relative_attrition` |
| `KPI-04` | **OverTime Association Within Filter** | `LIVE` | OR=3.77, OT share=28.3% (a=127, b=289, c=110, d=944) | 1470 | [2.8296, 5.0263] | `reliable` | `src.workforce_intel.kpis.kpi_04_overtime_association` |
| `KPI-05` | **Promotion-Gap Cohort Rate** | `LIVE` | Cohort rate=10.79% (n=890), 6+ vs 0-5 OR=1.96 | 890 | [0.0891, 0.1300] | `reliable` | `src.workforce_intel.kpis.kpi_05_promotion_gap_cohort` |
| `KPI-06` | **Adjusted OverTime Association** | `STATIC` | MH (JobLevel) OR=4.05 [3.01, 5.46]; M1 OR=4.81; M2 OR=5.27; P4 OR=4.00 | 1470 | [3.0119, 5.4558] | `workforce-level, not filterable` | `reports/phase5/tables/overtime_comparison.csv` |
| `KPI-07` | **Hotspot Universe Summary** | `STATIC` | Universe=104 segments (98 with n>=30); Tier 1=25; Tier 2=2; Prior m: 1-way=29.55, 2-way=8.65 | 104 | - | `workforce-level, not filterable` | `reports/phase5/tables/hotspot_universe.csv` |
| `KPI-08` | **Hotspot Holdout Validation** | `STATIC` | Screened Tier 1=25; Holdout n>=30: 14/25; Above point: 12/14; Above lower bound: 7/14 | 25 | - | `workforce-level, not filterable` | `reports/phase5/tables/hotspot_split_validation.csv` |
| `KPI-09` | **Hotspot Greedy Coverage** | `STATIC` | Steps=8; Step 5: 30.7% workforce, 67.5% leavers; Final: 57.9% workforce, 87.3% leavers | 1470 | - | `workforce-level, not filterable` | `reports/phase5/tables/greedy_coverage.csv` |
| `KPI-10` | **Model Triage Yield** | `STATIC` | Base rate=15.99%; Top 10%: k=29, TP=18, Prec=62.1%, Rec=38.3%, Lift=3.88x; Top 20%: k=59, TP=27, Prec=45.8%, Rec=57.4%, Lift=2.86x | 294 | - | `workforce-level, not filterable` | `reports/phase4/tables/holdout_metrics.csv` |

---

## 3. Representative Filter Case Evaluations (Live KPIs 1 through 5)

The live KPI functions were evaluated across 12 representative organizational, operational, multi-attribute, and boundary scenarios:

| Case ID | Scenario Description | Filter JSON | KPI ID | Metric Name | Output Value Summary | n | 95% Confidence Interval | Reliability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CASE-01` | Unfiltered Workforce (Baseline) | `{}` | `KPI-01` | **Attrition Rate** | 16.12% | 1470 | [0.1433, 0.1809] | `reliable` |
| `CASE-01` | Unfiltered Workforce (Baseline) | `{}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=1470, leavers=237 (100.0% headcount, 100.0% leavers) | 1470 | - | `reliable` |
| `CASE-01` | Unfiltered Workforce (Baseline) | `{}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+0.00 pp, ratio=1.00x, flag=no_separation | 1470 | [0.1433, 0.1809] | `reliable` |
| `CASE-01` | Unfiltered Workforce (Baseline) | `{}` | `KPI-04` | **OverTime Association Within Filter** | OR=3.77, OT share=28.3% (a=127, b=289, c=110, d=944) | 1470 | [2.8296, 5.0263] | `reliable` |
| `CASE-01` | Unfiltered Workforce (Baseline) | `{}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=10.79% (n=890), 6+ vs 0-5 OR=1.96 | 890 | [0.0891, 0.1300] | `reliable` |
| `CASE-02` | Department: Sales | `{"Department": "Sales"}` | `KPI-01` | **Attrition Rate** | 20.63% | 446 | [0.1713, 0.2463] | `reliable` |
| `CASE-02` | Department: Sales | `{"Department": "Sales"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=446, leavers=92 (30.3% headcount, 38.8% leavers) | 446 | - | `reliable` |
| `CASE-02` | Department: Sales | `{"Department": "Sales"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+4.51 pp, ratio=1.28x, flag=above_baseline | 446 | [0.1713, 0.2463] | `reliable` |
| `CASE-02` | Department: Sales | `{"Department": "Sales"}` | `KPI-04` | **OverTime Association Within Filter** | OR=3.74, OT share=28.7% (a=48, b=80, c=44, d=274) | 446 | [2.3145, 6.0318] | `reliable` |
| `CASE-02` | Department: Sales | `{"Department": "Sales"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=13.88% (n=281), 6+ vs 0-5 OR=2.40 | 281 | [0.1032, 0.1841] | `low_n` |
| `CASE-03` | Department: Research & Development | `{"Department": "Research & Development"}` | `KPI-01` | **Attrition Rate** | 13.84% | 961 | [0.1180, 0.1617] | `reliable` |
| `CASE-03` | Department: Research & Development | `{"Department": "Research & Development"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=961, leavers=133 (65.4% headcount, 56.1% leavers) | 961 | - | `reliable` |
| `CASE-03` | Department: Research & Development | `{"Department": "Research & Development"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=-2.28 pp, ratio=0.86x, flag=no_separation | 961 | [0.1180, 0.1617] | `reliable` |
| `CASE-03` | Department: Research & Development | `{"Department": "Research & Development"}` | `KPI-04` | **OverTime Association Within Filter** | OR=4.02, OT share=28.2% (a=74, b=197, c=59, d=631) | 961 | [2.7538, 5.8607] | `reliable` |
| `CASE-03` | Department: Research & Development | `{"Department": "Research & Development"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=9.46% (n=571), 6+ vs 0-5 OR=1.73 | 571 | [0.0732, 0.1214] | `low_n` |
| `CASE-04` | Department: Human Resources | `{"Department": "Human Resources"}` | `KPI-01` | **Attrition Rate** | 19.05% | 63 | [0.1125, 0.3041] | `reliable` |
| `CASE-04` | Department: Human Resources | `{"Department": "Human Resources"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=63, leavers=12 (4.3% headcount, 5.1% leavers) | 63 | - | `reliable` |
| `CASE-04` | Department: Human Resources | `{"Department": "Human Resources"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+2.93 pp, ratio=1.18x, flag=no_separation | 63 | [0.1125, 0.3041] | `reliable` |
| `CASE-04` | Department: Human Resources | `{"Department": "Human Resources"}` | `KPI-04` | **OverTime Association Within Filter** | OR=2.32, OT share=27.0% (a=5, b=12, c=7, d=39) | 63 | [0.6217, 8.6680] | `low_n` |
| `CASE-04` | Department: Human Resources | `{"Department": "Human Resources"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=7.89% (n=38), 6+ vs 0-5 OR=None | 38 | [0.0272, 0.2080] | `unreliable_zero_cell` |
| `CASE-05` | JobLevel: Entry-Level 1 | `{"JobLevel": 1}` | `KPI-01` | **Attrition Rate** | 26.34% | 543 | [0.2281, 0.3020] | `reliable` |
| `CASE-05` | JobLevel: Entry-Level 1 | `{"JobLevel": 1}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=543, leavers=143 (36.9% headcount, 60.3% leavers) | 543 | - | `reliable` |
| `CASE-05` | JobLevel: Entry-Level 1 | `{"JobLevel": 1}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+10.21 pp, ratio=1.63x, flag=above_baseline | 543 | [0.2281, 0.3020] | `reliable` |
| `CASE-05` | JobLevel: Entry-Level 1 | `{"JobLevel": 1}` | `KPI-04` | **OverTime Association Within Filter** | OR=5.92, OT share=28.7% (a=82, b=74, c=61, d=326) | 543 | [3.9045, 8.9821] | `reliable` |
| `CASE-05` | JobLevel: Entry-Level 1 | `{"JobLevel": 1}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=17.96% (n=206), 6+ vs 0-5 OR=3.11 | 206 | [0.1332, 0.2377] | `low_n` |
| `CASE-06` | OverTime: Yes (Exposed) | `{"OverTime": "Yes"}` | `KPI-01` | **Attrition Rate** | 30.53% | 416 | [0.2630, 0.3512] | `reliable` |
| `CASE-06` | OverTime: Yes (Exposed) | `{"OverTime": "Yes"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=416, leavers=127 (28.3% headcount, 53.6% leavers) | 416 | - | `reliable` |
| `CASE-06` | OverTime: Yes (Exposed) | `{"OverTime": "Yes"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+14.41 pp, ratio=1.89x, flag=above_baseline | 416 | [0.2630, 0.3512] | `reliable` |
| `CASE-06` | OverTime: Yes (Exposed) | `{"OverTime": "Yes"}` | `KPI-04` | **OverTime Association Within Filter** | OR=None (exposure fixed by filter), OT share=100.0% | 416 | - | `exposure_fixed_by_filter` |
| `CASE-06` | OverTime: Yes (Exposed) | `{"OverTime": "Yes"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=21.49% (n=242), 6+ vs 0-5 OR=2.09 | 242 | [0.1678, 0.2709] | `low_n` |
| `CASE-07` | OverTime: No (Unexposed) | `{"OverTime": "No"}` | `KPI-01` | **Attrition Rate** | 10.44% | 1054 | [0.0873, 0.1243] | `reliable` |
| `CASE-07` | OverTime: No (Unexposed) | `{"OverTime": "No"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=1054, leavers=110 (71.7% headcount, 46.4% leavers) | 1054 | - | `reliable` |
| `CASE-07` | OverTime: No (Unexposed) | `{"OverTime": "No"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=-5.69 pp, ratio=0.65x, flag=below_baseline | 1054 | [0.0873, 0.1243] | `reliable` |
| `CASE-07` | OverTime: No (Unexposed) | `{"OverTime": "No"}` | `KPI-04` | **OverTime Association Within Filter** | OR=None (exposure fixed by filter), OT share=0.0% | 1054 | - | `exposure_fixed_by_filter` |
| `CASE-07` | OverTime: No (Unexposed) | `{"OverTime": "No"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=6.79% (n=648), 6+ vs 0-5 OR=1.91 | 648 | [0.0510, 0.0899] | `low_n` |
| `CASE-08` | TenureBand: 0-1 Years | `{"TenureBand": "0-1"}` | `KPI-01` | **Attrition Rate** | 34.88% | 215 | [0.2883, 0.4147] | `reliable` |
| `CASE-08` | TenureBand: 0-1 Years | `{"TenureBand": "0-1"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=215, leavers=75 (14.6% headcount, 31.6% leavers) | 215 | - | `reliable` |
| `CASE-08` | TenureBand: 0-1 Years | `{"TenureBand": "0-1"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+18.76 pp, ratio=2.16x, flag=above_baseline | 215 | [0.2883, 0.4147] | `reliable` |
| `CASE-08` | TenureBand: 0-1 Years | `{"TenureBand": "0-1"}` | `KPI-04` | **OverTime Association Within Filter** | OR=3.61, OT share=32.1% (a=38, b=31, c=37, d=109) | 215 | [1.9751, 6.6023] | `reliable` |
| `CASE-08` | TenureBand: 0-1 Years | `{"TenureBand": "0-1"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | No Data (n=0) | 0 | - | `no_data` |
| `CASE-09` | Multi-Attribute: Sales AND OverTime=Yes | `{"Department": "Sales", "OverTime": "Yes"}` | `KPI-01` | **Attrition Rate** | 37.50% | 128 | [0.2959, 0.4614] | `reliable` |
| `CASE-09` | Multi-Attribute: Sales AND OverTime=Yes | `{"Department": "Sales", "OverTime": "Yes"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=128, leavers=48 (8.7% headcount, 20.3% leavers) | 128 | - | `reliable` |
| `CASE-09` | Multi-Attribute: Sales AND OverTime=Yes | `{"Department": "Sales", "OverTime": "Yes"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+21.38 pp, ratio=2.33x, flag=above_baseline | 128 | [0.2959, 0.4614] | `reliable` |
| `CASE-09` | Multi-Attribute: Sales AND OverTime=Yes | `{"Department": "Sales", "OverTime": "Yes"}` | `KPI-04` | **OverTime Association Within Filter** | OR=None (exposure fixed by filter), OT share=100.0% | 128 | - | `exposure_fixed_by_filter` |
| `CASE-09` | Multi-Attribute: Sales AND OverTime=Yes | `{"Department": "Sales", "OverTime": "Yes"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=26.25% (n=80), 6+ vs 0-5 OR=2.18 | 80 | [0.1786, 0.3682] | `low_n` |
| `CASE-10` | Low-n Boundary: Manager AND OverTime=Yes (n=27 < 30) | `{"JobRole": "Manager", "OverTime": "Yes"}` | `KPI-01` | **Attrition Rate** | 14.81% | 27 | [0.0592, 0.3248] | `low_n` |
| `CASE-10` | Low-n Boundary: Manager AND OverTime=Yes (n=27 < 30) | `{"JobRole": "Manager", "OverTime": "Yes"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=27, leavers=4 (1.8% headcount, 1.7% leavers) | 27 | - | `low_n` |
| `CASE-10` | Low-n Boundary: Manager AND OverTime=Yes (n=27 < 30) | `{"JobRole": "Manager", "OverTime": "Yes"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=-1.31 pp, ratio=0.92x, flag=no_separation | 27 | [0.0592, 0.3248] | `low_n` |
| `CASE-10` | Low-n Boundary: Manager AND OverTime=Yes (n=27 < 30) | `{"JobRole": "Manager", "OverTime": "Yes"}` | `KPI-04` | **OverTime Association Within Filter** | OR=None (exposure fixed by filter), OT share=100.0% | 27 | - | `exposure_fixed_by_filter` |
| `CASE-10` | Low-n Boundary: Manager AND OverTime=Yes (n=27 < 30) | `{"JobRole": "Manager", "OverTime": "Yes"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=17.39% (n=23), 6+ vs 0-5 OR=2.17 | 23 | [0.0698, 0.3714] | `low_n` |
| `CASE-11` | Descriptive Protected Filter: Female AND Single | `{"Gender": "Female", "MaritalStatus": "Single"}` | `KPI-01` | **Attrition Rate** | 23.62% | 199 | [0.1825, 0.2998] | `reliable` |
| `CASE-11` | Descriptive Protected Filter: Female AND Single | `{"Gender": "Female", "MaritalStatus": "Single"}` | `KPI-02` | **Headcount and Leavers Breakdown** | n=199, leavers=47 (13.5% headcount, 19.8% leavers) | 199 | - | `reliable` |
| `CASE-11` | Descriptive Protected Filter: Female AND Single | `{"Gender": "Female", "MaritalStatus": "Single"}` | `KPI-03` | **Relative Attrition vs Baseline** | diff=+7.50 pp, ratio=1.46x, flag=above_baseline | 199 | [0.1825, 0.2998] | `reliable` |
| `CASE-11` | Descriptive Protected Filter: Female AND Single | `{"Gender": "Female", "MaritalStatus": "Single"}` | `KPI-04` | **OverTime Association Within Filter** | OR=3.85, OT share=31.7% (a=26, b=37, c=21, d=115) | 199 | [1.9417, 7.6264] | `low_n` |
| `CASE-11` | Descriptive Protected Filter: Female AND Single | `{"Gender": "Female", "MaritalStatus": "Single"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | Cohort rate=15.52% (n=116), 6+ vs 0-5 OR=1.95 | 116 | [0.1005, 0.2320] | `low_n` |
| `CASE-12` | Empty Set: Sales AND Research Director (n=0) | `{"Department": "Sales", "JobRole": "Research Director"}` | `KPI-01` | **Attrition Rate** | No Data (n=0) | 0 | - | `no_data` |
| `CASE-12` | Empty Set: Sales AND Research Director (n=0) | `{"Department": "Sales", "JobRole": "Research Director"}` | `KPI-02` | **Headcount and Leavers Breakdown** | No Data (n=0) | 0 | - | `no_data` |
| `CASE-12` | Empty Set: Sales AND Research Director (n=0) | `{"Department": "Sales", "JobRole": "Research Director"}` | `KPI-03` | **Relative Attrition vs Baseline** | No Data (n=0) | 0 | - | `no_data` |
| `CASE-12` | Empty Set: Sales AND Research Director (n=0) | `{"Department": "Sales", "JobRole": "Research Director"}` | `KPI-04` | **OverTime Association Within Filter** | No Data (n=0) | 0 | - | `no_data` |
| `CASE-12` | Empty Set: Sales AND Research Director (n=0) | `{"Department": "Sales", "JobRole": "Research Director"}` | `KPI-05` | **Promotion-Gap Cohort Rate** | No Data (n=0) | 0 | - | `no_data` |

---

## 4. Multiplicity and Governance Assurances

1. **Observational Scope**: All live indicators quantify descriptive associations in cross-sectional records. No causal inferences or directional mechanisms are asserted.
2. **Demographic Filter Safeguards**: Protected attributes are permitted solely for descriptive audits and equity characterization. They cannot define operational risk hotspots or alter baseline regularised models.
3. **Decision Support Labeling**: Model triage yields (KPI-10) are explicitly designated for advisory decision support, prohibiting automated adverse personnel actions.
