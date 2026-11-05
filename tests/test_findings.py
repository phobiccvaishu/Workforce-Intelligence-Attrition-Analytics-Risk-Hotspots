"""Test suite for Phase 3B findings register engine (src/workforce_intel/findings.py)."""

import json
from pathlib import Path
import re
import pytest
import pandas as pd

from workforce_intel.config import (
    ANALYSIS_DIMENSIONS,
    FINDINGS_REGISTER_MD,
    FINDINGS_RULE,
    FINDINGS_SUMMARY_JSON,
    PHASE3_TABLES_DIR,
)
from workforce_intel.findings import (
    build_findings,
    compile_findings_summary,
    render_register_markdown,
)
from tests.test_language_hygiene import FORBIDDEN_WORDS


@pytest.fixture(scope="module")
def loaded_tables():
    """Load all Phase 3A tables."""
    tables = {}
    for f in PHASE3_TABLES_DIR.glob("*.csv"):
        tables[f.name] = pd.read_csv(f)
        tables[f.stem] = tables[f.name]
    return tables


def test_findings_register_all_dimensions_present(loaded_tables):
    """Verify all 28 dimensions appear in findings and the cohort baseline rule is applied."""
    findings_df = build_findings(loaded_tables, FINDINGS_RULE)

    # 28 standard dimensions + 1 cohort dimension
    dimensions_in_findings = findings_df["dimension"].unique().tolist()
    expected_dims = [d["dimension"] for d in ANALYSIS_DIMENSIONS]
    for d in expected_dims:
        assert d in dimensions_in_findings, f"Dimension {d} missing from findings register."

    assert "PromotionGapBand_Cohort" in dimensions_in_findings, "Cohort promotion gap dimension missing."

    # Verify cohort baseline rule: PromotionGapBand_Cohort baseline must be cohort rate (~10.8%)
    cohort_findings = findings_df[findings_df["dimension"] == "PromotionGapBand_Cohort"]
    for _, row in cohort_findings.iterrows():
        assert row["baseline_type"] == "Cohort Baseline"
        assert round(row["baseline_rate_pct"], 1) == 10.8

    # Null result dimensions
    no_sep_dims = findings_df[findings_df["direction"] == "no separation"]["dimension"].tolist()
    assert sorted(no_sep_dims) == ["Education", "Gender", "HasPriorExperience", "PerformanceRating"]


def test_findings_register_numeric_tokens_traceable(loaded_tables):
    """Verify every numeric token in docs/findings_register.md traces to a table cell or summary JSON."""
    assert FINDINGS_REGISTER_MD.exists(), "docs/findings_register.md must exist."
    content = FINDINGS_REGISTER_MD.read_text(encoding="utf-8")

    assert FINDINGS_SUMMARY_JSON.exists(), "findings_summary.json must exist."
    with open(FINDINGS_SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    summary_tokens = set()

    def extract_from_obj(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                extract_from_obj(v)
        elif isinstance(obj, list):
            for v in obj:
                extract_from_obj(v)
        elif isinstance(obj, (int, float)):
            summary_tokens.add(str(obj))
            summary_tokens.add(f"{obj:.1f}")
            summary_tokens.add(f"{obj:.2f}")
            summary_tokens.add(f"{obj:.4f}")

    extract_from_obj(summary_data)

    tables_tokens = set()
    for csv_path in PHASE3_TABLES_DIR.glob("*.csv"):
        df = pd.read_csv(csv_path)
        for col in df.columns:
            for val in df[col].dropna():
                if isinstance(val, (int, float)):
                    tables_tokens.add(str(val))
                    tables_tokens.add(f"{val:.1f}")
                    tables_tokens.add(f"{val:.2f}")
                    tables_tokens.add(f"{val:.4f}")
                else:
                    for tok in re.findall(r"\b\d+(?:\.\d+)?\b", str(val)):
                        tables_tokens.add(tok)

    known_tokens = summary_tokens | tables_tokens

    # Standard configuration, scale levels, and finding numbering
    known_tokens.update([
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "28", "30", "95",
    ])
    for i in range(1, 100):
        known_tokens.add(f"{i:02d}")
        known_tokens.add(str(i))

    doc_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", content)
    untraced = [t for t in doc_tokens if t not in known_tokens]

    assert not untraced, f"Untraced numeric tokens found in findings_register.md: {set(untraced)}"


def test_language_hygiene_phase3b_documents():
    """Verify zero forbidden causal terms across all Phase 3B generated documents and scripts."""
    files_to_check = [
        Path("docs/findings_register.md"),
        Path("src/workforce_intel/findings.py"),
        Path("src/workforce_intel/viz.py"),
        Path("scripts/run_phase3b_figures.py"),
    ]

    violations = []
    for file_path in files_to_check:
        if not file_path.exists():
            continue
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_num, line in enumerate(lines, start=1):
            if "FORBIDDEN_WORDS" in line or "test_language_hygiene" in line:
                continue
            for pat in FORBIDDEN_WORDS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    violations.append(
                        f"{file_path}:{line_num} contains forbidden pattern '{match.group(0)}': {line.strip()}"
                    )

    assert not violations, "Forbidden terms found in Phase 3B documents:\n" + "\n".join(violations)


def test_findings_provenance_synthetic_data():
    """Verify build_findings and render_register_markdown contain zero hardcoded real numbers.
    
    Generates a genuinely distinct synthetic dataset (~300 rows, different segments/roles/departments,
    different attrition rate), runs through Phase 3A table engine, build_findings, and render_register_markdown.
    (a) Asserts expected synthetic values independently computed with pandas appear without 'or'.
    (b) Asserts zero real-data numeric tokens and segment names appear in synthetic render.
    """
    from workforce_intel.eda import overall_summary, rate_table
    import numpy as np

    np.random.seed(42)
    n_rows = 300

    syn_data = {
        "Department": ["Hardware"] * 150 + ["Operations"] * 150,
        "JobRole": (["Architect"] * 50 + ["Technician"] * 50 + ["Operator"] * 50) * 2,
        "JobLevel": ([1] * 50 + [2] * 50 + [3] * 50) * 2,
        "AgeGroup": (["Youth"] * 50 + ["Prime"] * 50 + ["Senior"] * 50) * 2,
        "Gender": ["NonBinary"] * 150 + ["Agender"] * 150,
        "MaritalStatus": ["DomesticPartner"] * 150 + ["Widowed"] * 150,
        "Education": ([1] * 75 + [2] * 75) * 2,
        "EducationField": (["Physics"] * 75 + ["Chemistry"] * 75) * 2,
        "TenureBand": ["BandA"] * 150 + ["BandB"] * 150,
        "EarlyTenure": ([1] * 75 + [0] * 75) * 2,
        "CareerStage": ["EarlyCareer"] * 150 + ["LateCareer"] * 150,
        "RoleTenureBand": ["RoleEarly"] * 150 + ["RoleSenior"] * 150,
        "ManagerTenureBand": ["MgrEarly"] * 150 + ["MgrSenior"] * 150,
        "PromotionGapBand": ["GapLow"] * 150 + ["GapHigh"] * 150,
        "PromotionGapEqualsTenure": ([1] * 75 + [0] * 75) * 2,
        "HasPriorExperience": ([1] * 75 + [0] * 75) * 2,
        "NumCompaniesWorked": ([1] * 75 + [2] * 75) * 2,
        "OverTime": ["Always"] * 150 + ["Never"] * 150,
        "BusinessTravel": ["FrequentFlyer"] * 150 + ["Stationary"] * 150,
        "DistanceBand": ["NearOffice"] * 150 + ["FarOffice"] * 150,
        "WorkLifeBalance": ([1] * 75 + [2] * 75) * 2,
        "JobSatisfaction": ([1] * 75 + [2] * 75) * 2,
        "EnvironmentSatisfaction": ([1] * 75 + [2] * 75) * 2,
        "RelationshipSatisfaction": ([1] * 75 + [2] * 75) * 2,
        "JobInvolvement": ([1] * 75 + [2] * 75) * 2,
        "StockOptionLevel": ([1] * 75 + [2] * 75) * 2,
        "TrainingTimesLastYear": ([1] * 75 + [2] * 75) * 2,
        "PerformanceRating": ([1] * 75 + [2] * 75) * 2,
        "PromotionAnalysisCohort": [1] * 200 + [0] * 100,
    }

    # 60 leavers in first 150, 15 leavers in last 150 => 75 total leavers (25.0%)
    syn_data["Attrition"] = [1] * 60 + [0] * 90 + [1] * 15 + [0] * 135
    df_syn = pd.DataFrame(syn_data)

    # Run through Phase 3A table engine
    syn_tables = {}
    syn_tables["overall_baseline"] = overall_summary(df_syn, target_col="Attrition")
    syn_tables["overall_baseline.csv"] = syn_tables["overall_baseline"]

    for dim_info in ANALYSIS_DIMENSIONS:
        dim = dim_info["dimension"]
        tbl = dim_info["table"]
        rt = rate_table(df_syn, dim, target_col="Attrition")
        syn_tables[tbl] = rt
        syn_tables[tbl.replace(".csv", "")] = rt

    cohort_syn = df_syn[df_syn["PromotionAnalysisCohort"] == 1].copy()
    cohort_rt = rate_table(cohort_syn, "PromotionGapBand", target_col="Attrition")
    syn_tables["promotiongap_cohort_attrition.csv"] = cohort_rt
    syn_tables["promotiongap_cohort_attrition"] = cohort_rt

    # Build findings & markdown
    syn_findings = build_findings(syn_tables)
    syn_summary = compile_findings_summary(syn_tables, syn_findings)
    syn_md = render_register_markdown(syn_findings, syn_summary)

    # (a) Assert synthetic values independently computed appear without "or"
    exp_rate_pct = f"{(df_syn['Attrition'].sum() / len(df_syn)) * 100.0:.1f}%"
    assert exp_rate_pct in syn_md
    assert str(len(df_syn)) in syn_md
    assert str(int(df_syn["Attrition"].sum())) in syn_md
    cohort_rate_pct = f"{(cohort_syn['Attrition'].sum() / len(cohort_syn)) * 100.0:.1f}%"
    assert cohort_rate_pct in syn_md
    assert str(len(cohort_syn)) in syn_md
    assert str(int(cohort_syn["Attrition"].sum())) in syn_md
    assert "Hardware" in syn_md
    assert "Operations" in syn_md

    # (b) Token cross-check against REAL findings_register.md and findings_summary.json
    real_md = FINDINGS_REGISTER_MD.read_text(encoding="utf-8")
    with open(FINDINGS_SUMMARY_JSON, "r", encoding="utf-8") as f:
        real_summary = json.load(f)

    syn_table_tokens = set()
    for df in syn_tables.values():
        for col in df.columns:
            for val in df[col].dropna():
                if isinstance(val, (int, float)):
                    syn_table_tokens.add(str(val))
                    syn_table_tokens.add(f"{val:.1f}")
                    syn_table_tokens.add(f"{val:.2f}")
                    syn_table_tokens.add(f"{val:.4f}")
                else:
                    for tok in re.findall(r"\b\d+(?:\.\d+)?\b", str(val)):
                        syn_table_tokens.add(tok)

    allowed_constants = {"30", "0.05", "95", "1.5"}
    structural_tokens = {"1", "2", "3", "4", "28"}
    for i in range(1, 100):
        structural_tokens.add(f"{i:02d}")
        structural_tokens.add(str(i))

    real_tokens = set(re.findall(r"\b\d+(?:\.\d+)?\b", real_md))
    forbidden_real_tokens = real_tokens - syn_table_tokens - allowed_constants - structural_tokens

    syn_md_tokens = set(re.findall(r"\b\d+(?:\.\d+)?\b", syn_md))
    leaked_tokens = syn_md_tokens & forbidden_real_tokens
    assert not leaked_tokens, f"Real data numeric tokens leaked into synthetic render: {leaked_tokens}"

    real_segment_names = [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Sales Representative",
        "Research & Development", "Human Resources", "Travel_Frequently",
        "Travel_Rarely", "Non-Travel", "Life Sciences", "Medical",
        "Technical Degree", "Marketing",
    ]
    for seg in real_segment_names:
        assert seg not in syn_md, f"Real segment name '{seg}' leaked into synthetic markdown render."

    # (c) Sentence-level non-numeric finding phrase cross-check
    # Real finding-specific phrases describing real segments or rate comparisons must not appear in synthetic render
    real_finding_phrases = [
        "OverTime rate exceeds non-overtime rate at job levels 1, 2, 3, 4",
        "Non-overtime rate is lower than overtime rate at job levels 1, 2, 3, 4",
        "| MaritalStatus | Single |",
        "| EducationField | Technical Degree |",
        "| JobRole | Sales Representative |",
        "| JobRole | Laboratory Technician |",
    ]
    for phrase in real_finding_phrases:
        assert phrase not in syn_md, f"Real finding phrase '{phrase}' leaked into synthetic markdown render."
