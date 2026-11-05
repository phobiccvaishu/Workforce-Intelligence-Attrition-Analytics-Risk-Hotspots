"""Language hygiene test enforcing zero forbidden causal terms across all project documents and reports."""

from pathlib import Path
import re
import pytest

FORBIDDEN_WORDS = [
    r"\bcause\b",
    r"\bcauses\b",
    r"\bcaused\b",
    r"\bdrives\b",
    r"\bdriver\b",
    r"\bdrivers\b",
    r"\bleads to\b",
    r"\bbecause\b",
    r"\beffect of\b",
    r"\bimpact of\b",
    r"\brisk factor\b",
    r"\brisk factors\b",
    r"\bpredicts\b",
]


def test_language_hygiene_phase3_reports():
    """Reports, plans, and EDA source code must contain zero forbidden causal terms."""
    files_to_check = [
        Path("docs/analysis_plan.md"),
        Path("reports/phase3/eda_tables_report.md"),
        Path("src/workforce_intel/eda.py"),
        Path("src/workforce_intel/reporting.py"),
        Path("scripts/run_phase3a_eda.py"),
    ]

    violations = []

    for file_path in files_to_check:
        if not file_path.exists():
            continue
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_num, line in enumerate(lines, start=1):
            # Exclude code lines that define FORBIDDEN_WORDS or tests asserting them
            if "FORBIDDEN_WORDS" in line or "test_language_hygiene" in line:
                continue
            for pat in FORBIDDEN_WORDS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    violations.append(
                        f"{file_path}:{line_num} contains forbidden pattern '{match.group(0)}': {line.strip()}"
                    )

    assert not violations, (
        "Forbidden causal terms found in Phase 3 documents:\n" + "\n".join(violations)
    )
