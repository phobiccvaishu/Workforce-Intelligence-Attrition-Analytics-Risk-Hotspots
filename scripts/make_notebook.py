"""Generate the thin inspection notebook 01_data_inspection.ipynb."""

from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell(
        "# Phase 1: Data Inspection & Validation Notebook\n"
        "## Project: Workforce Attrition Patterns and Risk Hotspot Analysis\n"
        "**Scope:** Phase 1 – Dataset Inspection, Structural Integrity & Validation Audits\n\n"
        "> **DISCLAIMER:** The dataset provenance is UNVERIFIED. This data is analyzed as a "
        "public/synthetic HR dataset. Findings must not be described as real Palo Alto "
        "Networks employment records or personnel data. No data transformations or cleaning "
        "are performed in this inspection phase."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n\n"
        "# Ensure project src is in python path\n"
        "project_root = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
        "sys.path.insert(0, str(project_root / 'src'))\n\n"
        "import pandas as pd\n"
        "from IPython.display import display, Image\n\n"
        "from workforce_intel.config import (\n"
        "    RAW_DATA_FILE, EXPECTED_COLUMNS, NOMINAL_COLUMNS,\n"
        "    ORDINAL_COLUMNS_DOCUMENTED_RANGES, PHASE1_FIGURES_DIR\n"
        ")\n"
        "from workforce_intel.io import load_raw\n"
        "from workforce_intel.validation import (\n"
        "    structure_report, missing_report, duplicate_report,\n"
        "    categorical_audit, range_audit, target_audit,\n"
        "    consistency_checks, zero_audit, outlier_report,\n"
        "    collinearity_report, group_size_report, constant_column_check\n"
        ")\n\n"
        "print('Libraries and validation modules imported successfully.')"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Safe Dataset Loading & Schema Structure\n"
        "Load the raw dataset using `load_raw()` and verify structural conformity against `EXPECTED_COLUMNS`."
    ),
    nbf.v4.new_code_cell(
        "df, delimiter = load_raw(RAW_DATA_FILE)\n"
        "print(f'Detected Delimiter: {delimiter}')\n"
        "struct = structure_report(df)\n"
        "print(f\"Dimensions: {struct['n_rows']} rows x {struct['n_cols']} columns\")\n"
        "print(f\"Memory: {struct['memory_mb']} MB\")\n"
        "print(f\"Contract Violated: {struct['contract_violated']}\")\n"
        "print(f\"Column order matches expected: {struct['order_matches_expected']}\")\n"
        "display(struct['details_table'].head(10))"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Missing Values, Sentinels & Duplicates\n"
        "Audit missing cells, empty/whitespace strings, and sentinel values (`?`, `NA`, `-1`, `999`)."
    ),
    nbf.v4.new_code_cell(
        "missing_df = missing_report(df)\n"
        "print('Total NaN cells across dataset:', missing_df['nan_count'].sum())\n"
        "print('Total empty string cells:', missing_df['empty_string_count'].sum())\n"
        "print('Total sentinel tokens:', missing_df['string_sentinel_count'].sum() + missing_df['numeric_sentinel_count'].sum())\n\n"
        "dup_report = duplicate_report(df)\n"
        "print('Full-row duplicates:', dup_report['full_row_duplicates'])\n"
        "print('Demographic key subset duplicates:', dup_report['key_subset_duplicates'])"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Target Variable Audit (`Attrition`)\n"
        "Verify binary compliance {0, 1}, class balance, and the Wilson 95% Confidence Interval."
    ),
    nbf.v4.new_code_cell(
        "target = target_audit(df)\n"
        "print('Target column:', target['target_column'])\n"
        "print('Distinct values:', target['distinct_values'])\n"
        "print(f\"Class 0 (Stayed): {target['count_attrition_0']} ({target['rate_attrition_0_pct']}%)\")\n"
        "print(f\"Class 1 (Left):   {target['count_attrition_1']} ({target['rate_attrition_1_pct']}%)\")\n"
        "print(f\"Imbalance Ratio: {target['imbalance_ratio_stayed_to_left']}:1\")\n"
        "print(f\"Wilson 95% Score CI for Attrition=1: [{target['wilson_95_ci_lower_pct']}%, {target['wilson_95_ci_upper_pct']}%]\")\n"
        "display(Image(filename=str(PHASE1_FIGURES_DIR / 'target_distribution.png')))"
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Domain & Temporal Consistency Checks (C1 – C9)\n"
        "Evaluate tenure progression rules, hierarchy structures, and compensation rates coherence."
    ),
    nbf.v4.new_code_cell(
        "consistency = consistency_checks(df)\n"
        "print('C1 (YearsAtCompany <= TotalWorkingYears) violations:', consistency['C1']['violation_count'])\n"
        "print('C2 (YearsInCurrentRole <= YearsAtCompany) violations:', consistency['C2']['violation_count'])\n"
        "print('C3 (YearsSinceLastPromotion <= YearsAtCompany) violations:', consistency['C3']['violation_count'])\n"
        "print('C4 (YearsWithCurrManager <= YearsAtCompany) violations:', consistency['C4']['violation_count'])\n"
        "print('C5 Hard (TotalWorkingYears <= Age - 14) violations:', consistency['C5_hard']['violation_count'])\n"
        "print('C5 Soft (TotalWorkingYears <= Age - 18) violations:', consistency['C5_soft']['violation_count'])\n"
        "print('C6 (NumCompaniesWorked == 0 & TotalWorkingYears > YearsAtCompany) ambiguities:', consistency['C6']['violation_count'])\n"
        "print(f\"C7 Department x JobRole populated cells: {consistency['C7']['populated_cells']} of {consistency['C7']['total_cells']}\")\n"
        "display(consistency['C7']['crosstab_table'])"
    ),
    nbf.v4.new_markdown_cell(
        "## 5. Zero-Value Audit & Outliers\n"
        "Inspect zero counts and detect 1.5x / 3.0x IQR outliers."
    ),
    nbf.v4.new_code_cell(
        "zero_df = zero_audit(df)\n"
        "display(zero_df[zero_df['zero_count'] > 0].sort_values(by='zero_count', ascending=False))\n\n"
        "outliers_df = outlier_report(df)\n"
        "display(outliers_df[outliers_df['outliers_count_1_5x'] > 0][['column', 'q1', 'median', 'q3', 'iqr', 'outliers_count_1_5x', 'outliers_pct_1_5x', 'skewness']])\n"
        "display(Image(filename=str(PHASE1_FIGURES_DIR / 'boxplots_outliers.png')))"
    ),
    nbf.v4.new_markdown_cell(
        "## 6. Ordinal Ranges, Collinearity & Figures\n"
        "Audit scale adherence, high correlation pairs (|rho| >= 0.7), and visualize Spearman matrix."
    ),
    nbf.v4.new_code_cell(
        "range_df = range_audit(df)\n"
        "print('Restricted range columns:', range_df[range_df['is_restricted_range']]['column'].tolist())\n\n"
        "collin = collinearity_report(df)\n"
        "print('High collinearity pairs (|rho| >= 0.7):')\n"
        "display(collin['high_correlation_pairs'])\n"
        "display(Image(filename=str(PHASE1_FIGURES_DIR / 'spearman_heatmap.png')))\n"
        "display(Image(filename=str(PHASE1_FIGURES_DIR / 'numeric_distributions_grid.png')))"
    ),
]

nb.cells = cells
out_path = Path("notebooks/01_data_inspection.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook generated successfully at:", out_path)
