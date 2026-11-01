# Phase 2 Data Cleaning & Preprocessing Report
## Workforce Attrition Analytics System

- **Execution Timestamp (UTC):** 2026-11-01T19:00:19.854262+00:00
- **Raw Dataset SHA-256:** `389735e8909136a51fe487040df8f8662c2d222bb3505a83745c5a8c5bf20c7c`
- **Processed Dataset SHA-256:** `e8ecca3cbdfc45877851d4c6618734cf127dc8efb594580a01bbed544c4236ed`
- **Environment:** pandas 3.0.3, numpy 2.4.4

> [!IMPORTANT]
> **PROVENANCE & METHODOLOGY NOTICE:**  
> The raw dataset is immutable and preserved at `data/raw/`. Exactly 1,470 rows were loaded and 1,470 rows were output. Exactly 0 rows were removed, winsorized, imputed, or modified. All 31 original columns pass through identical to the source. Exactly 15 documented features were derived in accordance with Phase 2 specifications. Currency symbols are omitted.

---

## 1. Row-Level & Schema Reconciliation

| Metric | Computed Value | Expected Value | Status |
| :--- | :--- | :--- | :--- |
| **Raw Dataset Rows** | `1470` | `1470` | **PASS** |
| **Processed Dataset Rows** | `1470` | `1470` | **PASS** |
| **Rows Dropped** | `0` | `0` | **PASS** |
| **Original 31 Columns Exact Match** | `100% Identical` | `100% Identical` | **PASS** |
| **RowID Sequence [0 .. n-1]** | `Valid & Continuous` | `Valid & Continuous` | **PASS** |
| **Derived Columns Count** | `15` | `15` | **PASS** |
| **Total Processed Columns** | `47` | `47` | **PASS** |

- **Row Removal Register:** Exactly 0 rows dropped. Dropped records audit table is empty.
- **Surrogate Identifier:** `RowID` injected as column 0 (`0 .. 1469`). Maps directly to DataFrame index and Phase 1 audit row numbers (CSV line number = RowID + 2).

---

## 2. Derived Feature Specifications & Audits

Exactly 15 features were derived using pure functions. All cut-points are right-inclusive and documented in `config.py`.

| Derived Column | Data Type | Non-Null Count | Null Count | Domain Summary |
| :--- | :--- | :--- | :--- | :--- |
| `AgeGroup` | `category` | 1,470 | 0 | 4 categories: ['18-25', '26-35', '36-45', '46-60'] |
| `TenureBand` | `category` | 1,470 | 0 | 5 categories: ['0-1', '2-3', '4-5', '6-10', '11+'] |
| `EarlyTenure` | `int64` | 1,470 | 0 | Range [0 .. 1] |
| `CareerStage` | `category` | 1,470 | 0 | 4 categories: ['0-5', '6-10', '11-20', '21+'] |
| `RoleTenureBand` | `category` | 1,470 | 0 | 4 categories: ['0-1', '2-3', '4-7', '8+'] |
| `ManagerTenureBand` | `category` | 1,470 | 0 | 4 categories: ['0-1', '2-3', '4-7', '8+'] |
| `PromotionGapBand` | `category` | 1,470 | 0 | 4 categories: ['0', '1-2', '3-5', '6+'] |
| `DistanceBand` | `category` | 1,470 | 0 | 4 categories: ['1-5', '6-10', '11-20', '21-29'] |
| `PriorExperienceYears` | `int64` | 1,470 | 0 | Range [0 .. 33] |
| `HasPriorExperience` | `int64` | 1,470 | 0 | Range [0 .. 1] |
| `RoleTenureShare` | `float64` | 1,426 | 44 | Range [0.0 .. 1.0] |
| `PromotionGapShare` | `float64` | 1,426 | 44 | Range [0.0 .. 1.0] |
| `PromotionGapEqualsTenure` | `int64` | 1,470 | 0 | Range [0 .. 1] |
| `StagnationCohort` | `int64` | 1,470 | 0 | Range [0 .. 1] |
| `OutstandingRating` | `int64` | 1,470 | 0 | Range [0 .. 1] |

---

## 3. Zero-Denominator Rule & Missingness Audit

In accordance with Section 5.4, ratio features `RoleTenureShare` and `PromotionGapShare` evaluate strictly to `NaN` when `YearsAtCompany == 0` (44 rows). No zero imputation was performed, and no divide-by-zero warnings were emitted.

| Column | NaN Count | Share | Expected Count | Verification |
| :--- | :--- | :--- | :--- | :--- |
| `RoleTenureShare` | 44 | 2.99% | 44 | **PASS** |
| `PromotionGapShare` | 44 | 2.99% | 44 | **PASS** |

- **Global Missingness Check:** Exactly 0 NaN cells exist across all other 45 columns.

---

## 4. Subgroup Size Audits (n < 30 Rule)

In accordance with Section 5.7 and Rule 0.4, small sample sizes are explicitly flagged for downstream analysis and never silently merged.

- **Small Subgroups (n < 30) Flagged:** None detected across any derived category.

---

## 5. Artifact Manifest

- **Processed Dataset:** `data/processed/workforce_attrition_processed.csv` (UTF-8, LF line endings, 1,470 rows x 47 columns)
- **Machine-Readable Schema:** `data/processed/schema.json`
- **Reconciliation Tables:** Located in `reports/phase2/tables/`
  - `reconciliation.csv`
  - `bucket_sizes.csv`
  - `derived_feature_audit.csv`
  - `nan_audit.csv`
- **Machine-Readable Summary:** `reports/phase2/preprocessing_summary.json`
