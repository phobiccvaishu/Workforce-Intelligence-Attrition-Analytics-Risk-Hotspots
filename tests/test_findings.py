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


def test_findings_provenance_synthetic_data(loaded_tables):
    """Verify build_findings and render_register_markdown contain zero hardcoded real numbers."""
    synthetic_tables = {}
    for name, df in loaded_tables.items():
        syn_df = df.copy()
        if "attrition_rate_pct" in syn_df.columns:
            syn_df["attrition_rate_pct"] = syn_df["attrition_rate_pct"] + 11.1111
        if "ci95_low_pct" in syn_df.columns:
            syn_df["ci95_low_pct"] = syn_df["ci95_low_pct"] + 11.1111
        if "ci95_high_pct" in syn_df.columns:
            syn_df["ci95_high_pct"] = syn_df["ci95_high_pct"] + 11.1111
        synthetic_tables[name] = syn_df

    syn_findings = build_findings(synthetic_tables)
    syn_summary = compile_findings_summary(synthetic_tables, syn_findings)
    syn_md = render_register_markdown(syn_findings, syn_summary)

    # Check that synthetic baseline rate (16.1224 + 11.1111 = 27.2335) appears in markdown
    assert "27.2335" in syn_md or "27.2%" in syn_md, "Synthetic baseline rate must propagate dynamically."
    # Real data baseline 16.1224 must NOT appear in synthetic markdown
    assert "16.1224" not in syn_md, "Real baseline rate 16.1224 was unexpectedly found in synthetic render."
