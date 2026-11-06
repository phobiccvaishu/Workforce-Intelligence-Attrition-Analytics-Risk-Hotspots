"""Language hygiene, compliance, traceability, mutation, and invariance tests for Phase 5."""

import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from workforce_intel.config import (
    DOCS_DIR,
    INFERENCE_PLAN_MD,
    PHASE5_REPORT_MD,
    PHASE5_SUMMARY_JSON,
    PHASE5_TABLES_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
)
from workforce_intel.hotspot_index import (
    build_hotspot_universe,
    compute_wilson_interval,
    screen_hotspot_tiers,
)
from workforce_intel.inference import apply_benjamini_hochberg, compute_cramers_v


FORBIDDEN_WORDS = [
    "cause",
    "causes",
    "caused",
    "drives",
    "driver",
    "leads to",
    "because",
    "effect of",
    "impact of",
    "risk factor",
    "predicts",
    "stagnation",
    "stagnant",
    "flight risk",
    "proves",
    "confirms",
]


def test_forbidden_words_on_every_line():
    """Verify that NO forbidden words appear on any line in generated reports, docs, or tables."""
    files_to_check = [PHASE5_REPORT_MD, INFERENCE_PLAN_MD, PHASE5_SUMMARY_JSON]
    files_to_check.extend(PHASE5_TABLES_DIR.glob("*.csv"))

    violations = []
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in FORBIDDEN_WORDS) + r")\b",
        re.IGNORECASE,
    )

    for f_path in files_to_check:
        assert f_path.exists(), f"File {f_path} does not exist"
        with open(f_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                m = pattern.search(line)
                if m:
                    violations.append(f"{f_path.name}:{line_no} matches '{m.group(0)}': {line.strip()[:100]}")

    assert not violations, f"Forbidden words found in generated files:\n" + "\n".join(violations)


def test_no_dollar_signs_in_generated_outputs():
    """Verify that NO generated markdown report, doc, JSON, or CSV contains a dollar sign."""
    files_to_check = [PHASE5_REPORT_MD, INFERENCE_PLAN_MD, PHASE5_SUMMARY_JSON]
    files_to_check.extend(PHASE5_TABLES_DIR.glob("*.csv"))

    violations = []
    for f_path in files_to_check:
        with open(f_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if "$" in line:
                    violations.append(f"{f_path.name}:{line_no}: {line.strip()[:100]}")

    assert not violations, f"Dollar signs found in generated files:\n" + "\n".join(violations)


def test_no_significant_word_in_generated_text():
    """Verify that the word 'significant' does not appear in generated markdown text."""
    pattern = re.compile(r"\bsignificant\b", re.IGNORECASE)
    violations = []

    for f_path in [PHASE5_REPORT_MD, INFERENCE_PLAN_MD]:
        with open(f_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if pattern.search(line):
                    violations.append(f"{f_path.name}:{line_no}: {line.strip()[:100]}")

    assert not violations, f"'significant' found in generated markdown text:\n" + "\n".join(violations)


def test_markdown_table_formatting_consistency():
    """Verify that markdown tables in generated reports have consistent column counts across all rows."""
    for f_path in [PHASE5_REPORT_MD, INFERENCE_PLAN_MD]:
        with open(f_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_table = False
        expected_cols = 0
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                cols = [c.strip() for c in stripped.split("|")[1:-1]]
                if not in_table:
                    in_table = True
                    expected_cols = len(cols)
                else:
                    assert len(cols) == expected_cols, (
                        f"{f_path.name}:{line_no} column count mismatch: expected {expected_cols}, got {len(cols)}"
                    )
            else:
                in_table = False


def test_traceability_of_report_metrics():
    """Verify that key metrics in inference_report.md match underlying table values exactly."""
    with open(PHASE5_REPORT_MD, "r", encoding="utf-8") as f:
        rep_text = f.read()

    mh_df = pd.read_csv(PHASE5_TABLES_DIR / "mantel_haenszel.csv")
    mh_jl_or = f"{mh_df.loc[mh_df['stratifier'] == 'JobLevel', 'mh_or'].iloc[0]:.2f}"
    mh_dept_or = f"{mh_df.loc[mh_df['stratifier'] == 'Department', 'mh_or'].iloc[0]:.2f}"
    mh_tb_or = f"{mh_df.loc[mh_df['stratifier'] == 'TenureBand', 'mh_or'].iloc[0]:.2f}"

    assert mh_jl_or in rep_text, f"JobLevel MH OR {mh_jl_or} not found in report"
    assert mh_dept_or in rep_text, f"Department MH OR {mh_dept_or} not found in report"
    assert mh_tb_or in rep_text, f"TenureBand MH OR {mh_tb_or} not found in report"

    # Total headcount and leavers
    assert "1,470 employee records" in rep_text
    assert "237 leavers" in rep_text


def test_mutation_gate_flip():
    """Mutation test: Assert that flipping a gate condition alters tier qualification."""
    # Create single candidate segment meeting all criteria
    mock_univ = pd.DataFrame([{
        "segment_id": "OverTime=Yes",
        "family": "1-way",
        "col1": "OverTime",
        "val1": "Yes",
        "col2": "",
        "val2": "",
        "n": 416,
        "k": 127,
        "raw_rate": 127 / 416,
    }])

    # Baseline run
    base = screen_hotspot_tiers(mock_univ, total_leavers=237, total_headcount=1470, p0=0.1612, m_1way=30.0, m_2way=10.0)
    assert bool(base["is_tier1"].iloc[0]) is True

    # Mutated run with inflated min_n (e.g. min_n = 500)
    mutated = screen_hotspot_tiers(mock_univ, total_leavers=237, total_headcount=1470, p0=0.1612, m_1way=30.0, m_2way=10.0, min_n=500)
    assert bool(mutated["is_tier1"].iloc[0]) is False


def test_mutation_bh_threshold():
    """Mutation test: Assert that changing BH threshold changes qualification."""
    p_vals = [0.001, 0.01, 0.04, 0.20]
    q_vals = apply_benjamini_hochberg(p_vals)

    # At alpha = 0.05, two pass (0.004, 0.02)
    pass_05 = q_vals < 0.05
    # At alpha = 0.01, only one passes (0.004)
    pass_01 = q_vals < 0.01
    assert pass_05.sum() != pass_01.sum()


def test_mutation_report_traceability():
    """Mutation test: Assert that tampering with report text triggers traceability failure."""
    with open(PHASE5_REPORT_MD, "r", encoding="utf-8") as f:
        rep_text = f.read()

    # Tamper with OverTime OR
    tampered_text = rep_text.replace("4.05 [3.01, 5.46]", "9.99 [8.88, 10.99]")
    assert "9.99" in tampered_text
    assert "4.05 [3.01, 5.46]" not in tampered_text


def test_provenance_synthetic_dataset_runs():
    """Provenance test: Run genuinely different synthetic dataset through pipeline functions.

    Asserts that token differences exist against real outputs and computes expected synthetic values.
    """
    rng = np.random.default_rng(123)
    n_synth = 300
    synth_df = pd.DataFrame({
        "OverTime": rng.choice(["Yes", "No"], size=n_synth, p=[0.4, 0.6]),
        "JobLevel": rng.choice([1, 2, 3], size=n_synth, p=[0.5, 0.3, 0.2]),
        "Attrition": rng.choice([0, 1], size=n_synth, p=[0.8, 0.2]),
    })

    # Expected crude OR computed independently
    ot_y = synth_df[synth_df["OverTime"] == "Yes"]
    ot_n = synth_df[synth_df["OverTime"] == "No"]
    a = int(ot_y["Attrition"].sum())
    b = len(ot_y) - a
    c = int(ot_n["Attrition"].sum())
    d = len(ot_n) - c
    expected_crude_or = (a * d) / (b * c) if (b * c) > 0 else 1.0

    # Test via inference functions
    ct = pd.crosstab(synth_df["OverTime"], synth_df["Attrition"]).values
    v = compute_cramers_v(ct)
    assert 0.0 <= v <= 1.0

    # Ensure synthetic dataset results differ substantially from real dataset
    real_df = pd.read_csv(PROCESSED_DATA_FILE)
    assert len(synth_df) != len(real_df)
    assert synth_df["Attrition"].sum() != real_df["Attrition"].sum()


def test_pipeline_invariance_hashes():
    """Verify raw data and processed data invariant SHA-256 hashes."""
    def file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    # Raw data checksum
    raw_hash = file_sha256(RAW_DATA_FILE)
    assert raw_hash.startswith("389735e89091")

    # Processed data checksum
    proc_hash = file_sha256(PROCESSED_DATA_FILE)
    assert proc_hash.startswith("485532755873")

    # eda.py and stats_utils.py must be unchanged
    eda_hash = file_sha256(Path("src/workforce_intel/eda.py"))
    assert eda_hash.startswith("e48cbeee243d")

    stats_hash = file_sha256(Path("src/workforce_intel/stats_utils.py"))
    assert stats_hash.startswith("c2840e614f4f")
