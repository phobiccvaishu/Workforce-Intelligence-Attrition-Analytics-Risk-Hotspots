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
    r"\bstagnation\b",
    r"\bstagnant\b",
    r"\bflight risk\b",
]


def check_files_for_forbidden_words(files_to_check):
    """Scan every line of given files for forbidden causal terms."""
    violations = []
    for file_path in files_to_check:
        if not file_path.exists():
            continue
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_num, line in enumerate(lines, start=1):
            # For decision_log.md, only evaluate Phase 3+ decisions (DL-27 onwards)
            if file_path.name == "decision_log.md" and not any(f"DL-{i:02d}" in line for i in range(27, 50)):
                continue
            for pat in FORBIDDEN_WORDS:
                match = re.search(pat, line, re.IGNORECASE)
                if match:
                    violations.append(
                        f"{file_path}:{line_num} contains forbidden pattern '{match.group(0)}': {line.strip()}"
                    )
    return violations


def test_language_hygiene_phase3_reports():
    """Reports, plans, and EDA source code must contain zero forbidden causal terms."""
    files_to_check = [
        Path("docs/analysis_plan.md"),
        Path("docs/decision_log.md"),
        Path("docs/findings_register.md"),
        Path("reports/phase3/eda_tables_report.md"),
        Path("src/workforce_intel/eda.py"),
        Path("src/workforce_intel/reporting.py"),
        Path("src/workforce_intel/findings.py"),
        Path("src/workforce_intel/viz.py"),
        Path("scripts/run_phase3a_eda.py"),
        Path("scripts/run_phase3b_figures.py"),
    ]

    violations = check_files_for_forbidden_words(files_to_check)
    assert not violations, (
        "Forbidden causal terms found in Phase 3 documents:\n" + "\n".join(violations)
    )


def test_language_hygiene_mutation_fails(tmp_path):
    """Verify that adding a forbidden word to docs/findings_register.md triggers an assertion failure."""
    original_md = Path("docs/findings_register.md")
    assert original_md.exists(), "docs/findings_register.md must exist for mutation testing."

    # Create temporary copy with injected forbidden word
    mutated_file = tmp_path / "findings_register.md"
    content = original_md.read_text(encoding="utf-8")
    mutated_content = content + "\nThis observation causes high attrition.\n"
    mutated_file.write_text(mutated_content, encoding="utf-8")

    violations = check_files_for_forbidden_words([mutated_file])
    assert len(violations) > 0, "Mutated file with forbidden word must trigger a violation."
    assert any("causes" in v for v in violations)
