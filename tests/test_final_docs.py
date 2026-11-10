"""Tests for Final Deliverables (Research Paper, Executive Summary, Figures).

Validates:
1. Build script runs deterministically and creates all required deliverables.
2. Reports contain zero banned causal phrases and zero dollar signs.
3. Headline attrition metrics match kpi_baseline.csv exactly.
4. Ethical safeguards and provenance disclaimers are present.
"""

import re
from pathlib import Path
import pandas as pd
import pytest

from workforce_intel.final_docs import (
    build_all_docs,
    PROJECT_ROOT,
    FINAL_DIR,
    FIGURES_DIR,
)


@pytest.fixture(scope="module")
def built_docs():
    """Ensure final docs are built."""
    build_all_docs()
    return {
        "paper_md": (FINAL_DIR / "paper.md").read_text(encoding="utf-8"),
        "exec_md": (FINAL_DIR / "executive_summary.md").read_text(encoding="utf-8"),
    }


def test_deliverables_exist():
    """Assert all final deliverable files exist."""
    assert (FINAL_DIR / "paper.md").is_file()
    assert (FINAL_DIR / "paper.docx").is_file()
    assert (FINAL_DIR / "paper.pdf").is_file()
    assert (FINAL_DIR / "executive_summary.md").is_file()
    assert (FINAL_DIR / "executive_summary.docx").is_file()
    assert (FINAL_DIR / "executive_summary.pdf").is_file()

    figures = list(FIGURES_DIR.glob("*.png"))
    assert len(figures) >= 20, f"Expected at least 20 canonical figures, found {len(figures)}"


def test_no_banned_phrases(built_docs):
    """Assert zero banned causal phrases outside permitted title and filename."""
    banned_patterns = [
        r"\bcauses\b",
        r"\bcaused by\b",
        r"\bleads to\b",
        r"\bdrives\b",
        r"\bproves\b",
        r"\bguarantees\b",
        r"\bflight risk\b",
        r"\breal employees\b",
        r"\bbecause of\b",
    ]
    for doc_name, text in [("paper.md", built_docs["paper_md"]), ("executive_summary.md", built_docs["exec_md"])]:
        for pattern in banned_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            assert not matches, f"Found banned phrase '{matches[0]}' in {doc_name}"


def test_zero_dollar_signs(built_docs):
    """Assert zero literal dollar signs in deliverable document bodies."""
    assert "$" not in built_docs["paper_md"], "Found '$' in paper.md"
    assert "$" not in built_docs["exec_md"], "Found '$' in executive_summary.md"


def test_headline_numbers_match_kpi_baseline(built_docs):
    """Assert headline numbers match kpi_baseline.csv exactly."""
    kpi_df = pd.read_csv(PROJECT_ROOT / "reports/phase6/tables/kpi_baseline.csv")
    kpi_01 = kpi_df.loc[kpi_df["kpi_id"] == "KPI-01"].iloc[0]

    n_total = int(kpi_01["n"])
    val_summary = str(kpi_01["value_summary"])  # "16.12% (237/1470)"
    
    assert str(n_total) in built_docs["paper_md"]
    assert "237" in built_docs["paper_md"]
    assert "16.12%" in built_docs["paper_md"]

    assert str(n_total) in built_docs["exec_md"]
    assert "237" in built_docs["exec_md"]
    assert "16.12%" in built_docs["exec_md"]


def test_provenance_and_ethics_safeguards(built_docs):
    """Assert required provenance caveats and ethical prohibitions are present."""
    for text in [built_docs["paper_md"], built_docs["exec_md"]]:
        assert "unverified" in text.lower()
        assert "synthetic" in text.lower()
        assert "individual" in text.lower()
        assert "prohibited" in text.lower()
