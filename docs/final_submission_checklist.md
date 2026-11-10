# Final Submission Checklist and Deliverables Manifest

This checklist documents the final deliverables, verification results, and operational limitations for the Workforce Attrition Patterns and Risk Hotspots project.

---

## 1. Deliverables Inventory

| Deliverable | Location | Formats | Status |
| :--- | :--- | :--- | :--- |
| **Research Paper** | `reports/final/paper.*` | Markdown, DOCX, PDF | Completed (410 lines, 20 figures referenced) |
| **Executive Summary** | `reports/final/executive_summary.*` | Markdown, DOCX, PDF | Completed (61 lines, plain business language) |
| **Canonical Figures** | `reports/final/figures/` | PNG (20 figures) | Copied from Phase 3 & 4 canonical outputs |
| **Interactive Dashboard** | `app/` | Streamlit multipage application | Completed (5 pages, hash-verified table access) |
| **KPI Baseline Table** | `reports/phase6/tables/kpi_baseline.csv` | CSV | Completed (8 core KPIs, exact Wilson CIs) |
| **Audited Technical Decisions** | `docs/decision_status.md` | Markdown | Completed (44 decisions: DL-01 through DL-44) |
| **Document Build Runner** | `scripts/build_final_docs.py` | Python CLI runner | Completed (deterministic, zero timestamps) |
| **Document Generation Engine** | `src/workforce_intel/final_docs.py` | Python module | Completed (loads tables, enforces language) |
| **Test Suite** | `tests/` | Pytest suite | Completed (all existing + new tests passing) |

---

## 2. Core Constraints & Governance Verification

- [x] **Zero Hand-Typed Statistics**: All counts, percentages, odds ratios, and test statistics in `paper.md` and `executive_summary.md` are dynamically read by code from committed pipeline tables (`reports/phase1..6`, `kpi_baseline.csv`, `consistency_audit.csv`).
- [x] **Zero Timestamps in Document Bodies**: Deliverable markdown documents contain no build timestamps, ensuring byte determinism across runs.
- [x] **Zero Dollar Signs**: Strict check confirms zero literal `$` characters in deliverable document bodies (currency is unknown/undocumented).
- [x] **Language Hygiene**: Zero causal claims (`causes`, `caused by`, `leads to`, `drives`, `proves`, `guarantees`, `flight risk`, `real employees`, `because of`). All relationships framed as observational associations.
- [x] **Provenance Transparency**: Provenance is explicitly documented as unverified, synthetic-style HR benchmark data. No company claims, employee identity assertions, or corporate logos.
- [x] **Responsible Use & Anti-Targeting**: Explicit prohibition against individual employee scoring, flight risk ranking, or operational targeting using protected attributes (Age, Gender, MaritalStatus).
- [x] **Approved References Only**: Strictly limited to Wilson (1927, JASA), Benjamini and Hochberg (1995, JRSS B), and Mantel and Haenszel (1959, JNCI), each tagged `[owner to verify]`. Zero fabricated citations, DOIs, or page numbers.
- [x] **Raw Dataset Integrity**: SHA-256 digest `389735e8909136a51fe487040df8f8662c2d222bb3505a83745c5a8c5bf20c7c` (209,376 bytes) preserved byte-exact in working tree and git blob.

---

## 3. Headline Metrics Alignment

| Metric | Source Table (`kpi_baseline.csv`) | Research Paper (`paper.md`) | Executive Summary (`executive_summary.md`) | Alignment |
| :--- | :--- | :--- | :--- | :--- |
| **Total Headcount** | 1,470 | 1,470 | 1,470 | Exact match |
| **Total Leavers** | 237 | 237 | 237 | Exact match |
| **Baseline Attrition Rate** | 16.12% | 16.12% | 16.12% | Exact match |
| **95% Wilson CI** | [14.33%, 18.09%] | [14.33%, 18.09%] | [14.33%, 18.09%] | Exact match |
| **OverTime Attrition Rate** | 30.53% (n=416) | 30.53% (n=416) | 30.53% (n=416) | Exact match |
| **Non-OverTime Rate** | 10.44% (n=1,054) | 10.44% (n=1,054) | 10.44% (n=1,054) | Exact match |
| **OverTime Crude OR** | 3.77 [2.83, 5.03] | 3.77 [2.83, 5.03] | 3.77 [2.83, 5.03] | Exact match |
| **OverTime MH JobLevel OR** | 4.05 [3.01, 5.46] | 4.05 [3.01, 5.46] | 4.05 [3.01, 5.46] | Exact match |
| **Promotion Gap MH OR** | 2.38 [1.44, 3.92] | 2.38 [1.44, 3.92] | 2.38 [1.44, 3.92] | Exact match |
| **Holdout PR-AUC** | 0.5736 | 0.5736 | 0.5736 | Exact match |

---

## 4. Known Analytical & Practical Limitations

1. **Unverified Provenance**: The source dataset was provided as an unverified flat file with synthetic characteristics; findings serve as methodological benchmarks and must not be treated as empirical records of Palo Alto Networks.
2. **Cross-Sectional Observational Nature**: Lack of longitudinal departure timestamps or right-censoring prevents survival modeling or hazard rate estimation.
3. **Sparse Subgroup Cell Sizes**: Low sample sizes (n < 30) in specific combinations require empirical Bayes shrinkage and cannot support definitive ranking claims.
4. **Compensation Incoherence**: Zero rank correlation between compensation rates (Daily, Hourly, Monthly) and MonthlyIncome precludes meaningful compensation analysis.
5. **Restricted Performance Scale**: `PerformanceRating` contains only levels 3 and 4, preventing assessment of low-rated employee turnover.
6. **DOCX and PDF Conversion Route**: Automated DOCX and PDF compilation utilizes Microsoft Word COM automation on Windows; on environments without Word installed, Markdown files remain canonical.
