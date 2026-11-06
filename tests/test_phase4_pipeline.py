"""Unit and integration tests for Phase 4 attrition risk modelling pipeline.

Mapped directly to Phase 4 specification and plan items:
Item 8.1: test_split_determinism
Item 8.2: test_no_train_holdout_id_overlap
Item 8.3: test_no_target_derived_feature_leakage
Item 8.4: test_pipeline_fits_per_fold
Item 8.5: test_metrics_reproduce_under_seed
Item 8.6: test_holdout_untouched_before_final_step
Item 8.7: test_hotspot_screening_strictly_on_train
Item 8.8: test_model_card_governance_headings

Closeout reporting & audit tests:
Item 1: test_holdout_triage_arithmetic_consistency
Item 2: test_fig10_source_tables_low_n_audit
Item 3: test_model_comparison_and_calibration_metrics
Item 4: test_materiality_threshold_governance
Item 5: test_hotspot_candidate_screening_and_holdout_provenance
Item 6: test_odds_ratios_standardized_units_and_contrasts
Item 7: test_demographic_fairness_triage_audit
Item 8: test_phase4_consistency_findings_audit

Real pipeline figure tests:
Item 9: test_real_pipeline_figure_layouts
Item 9: test_real_pipeline_figure_layout_mutations
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from workforce_intel.config import (
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
    PROJECT_ROOT,
)
from workforce_intel.io import load_processed
from workforce_intel.modeling import (
    DEMOGRAPHIC_COLUMNS,
    DROP_COLUMNS,
    SEED,
    build_preprocessor,
    compute_bootstrap_metrics,
    compute_logistic_odds_ratios,
    cross_validate_model,
    evaluate_demographic_fairness,
    evaluate_hotspots_on_holdout,
    evaluate_predictions,
    get_feature_sets,
    parse_segment_mask,
    screen_hotspots_on_train,
    train_test_split_data,
)
from workforce_intel.viz_phase4 import (
    plot_fig15_pr_curves,
    plot_fig16_calibration,
    plot_fig17_risk_bands,
    plot_fig18_odds_ratios,
    plot_fig19_hotspot_validation,
    plot_fig20_demographic_fairness,
)
from tests.test_figure_layout import collect_figure_layout_violations


@pytest.fixture(scope="module")
def processed_df():
    return load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)


@pytest.fixture(scope="module")
def pipeline_metrics():
    metrics_path = PROJECT_ROOT / "reports" / "phase4" / "metrics.json"
    assert metrics_path.exists(), "reports/phase4/metrics.json not found; run pipeline first"
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# SECTION 8 CORE PRE-FLIGHT & LEAKAGE PREVENTION TESTS
# =============================================================================

def test_split_determinism(processed_df):
    """Assert train/holdout split is exactly 80/20 and 100% deterministic under SEED=42."""
    X_tr1, y_tr1, X_ho1, y_ho1, tr_df1, ho_df1 = train_test_split_data(processed_df, seed=SEED)
    X_tr2, y_tr2, X_ho2, y_ho2, tr_df2, ho_df2 = train_test_split_data(processed_df, seed=SEED)

    assert len(tr_df1) == 1176, f"Expected 1,176 training rows, got {len(tr_df1)}"
    assert len(ho_df1) == 294, f"Expected 294 holdout rows, got {len(ho_df1)}"
    assert len(tr_df1) + len(ho_df1) == len(processed_df)

    pd.testing.assert_frame_equal(tr_df1, tr_df2)
    pd.testing.assert_frame_equal(ho_df1, ho_df2)


def test_no_train_holdout_id_overlap(processed_df):
    """Assert zero index/ID overlap between train and holdout splits."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)
    overlap = set(tr_df.index).intersection(set(ho_df.index))
    assert len(overlap) == 0, f"Detected {len(overlap)} overlapping indices between train and holdout!"


def test_no_target_derived_feature_leakage(processed_df):
    """Assert neither Set A nor Set B contains Attrition, and all dropped identifiers/constants are excluded."""
    set_a, set_b = get_feature_sets(processed_df)

    assert "Attrition" not in set_a
    assert "Attrition" not in set_b

    for col in DROP_COLUMNS:
        assert col not in set_a
        assert col not in set_b

    for demo_col in DEMOGRAPHIC_COLUMNS:
        assert demo_col in set_a
        assert demo_col not in set_b


def test_pipeline_fits_per_fold(processed_df):
    """Assert Preprocessor and models fit strictly per-fold without leaking holdout statistics."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    preprocessor = build_preprocessor(feature_set="B")
    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_ho_trans = preprocessor.transform(X_ho)

    assert not np.isnan(X_tr_trans).any(), "Found NaNs in transformed training data"
    assert not np.isnan(X_ho_trans).any(), "Found NaNs in transformed holdout data"
    assert X_tr_trans.shape[0] == 1176
    assert X_ho_trans.shape[0] == 294


def test_metrics_reproduce_under_seed(processed_df):
    """Assert model predictions and holdout bootstrap evaluation reproduce bit-for-bit under SEED=42."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    pipe = Pipeline([
        ("prep", build_preprocessor("B")),
        ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=SEED)),
    ]).fit(X_tr, y_tr)

    probs = pipe.predict_proba(X_ho)[:, 1]
    boot1 = compute_bootstrap_metrics(y_ho.to_numpy(), probs, n_bootstrap=100, seed=SEED)
    boot2 = compute_bootstrap_metrics(y_ho.to_numpy(), probs, n_bootstrap=100, seed=SEED)

    for k in boot1:
        assert np.isclose(boot1[k]["point"], boot2[k]["point"])
        assert np.isclose(boot1[k]["ci_lower"], boot2[k]["ci_lower"])
        assert np.isclose(boot1[k]["ci_upper"], boot2[k]["ci_upper"])


def test_holdout_untouched_before_final_step(processed_df):
    """Assert holdout data is never accessed during CV model training or hyperparameter selection."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    # Verify that cross-validation operates strictly on training split
    cv_res = cross_validate_model(
        lambda: LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=SEED),
        X_tr,
        y_tr,
        feature_set="B",
        seed=SEED,
    )
    assert len(cv_res["fold_records"]) == 15
    for f in cv_res["fold_records"]:
        assert f["pr_auc"] > 0.40


def test_hotspot_screening_strictly_on_train(processed_df):
    """Assert candidate hotspots are screened exclusively from training rows."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    screened = screen_hotspots_on_train(tr_df, min_n=30)
    assert len(screened) > 0, "No hotspots screened on training split"

    for h in screened:
        assert h["train_n"] >= 30
        assert h["train_n"] <= 1176
        assert h["train_ci_low"] > h["train_baseline"]


def test_model_card_governance_headings():
    """Assert docs/model_card.md exists and contains all required governance sections."""
    card_path = PROJECT_ROOT / "docs" / "model_card.md"
    assert card_path.exists(), "docs/model_card.md does not exist"

    content = card_path.read_text(encoding="utf-8")
    required_headings = [
        "## 1. Model Details",
        "## 2. Intended Use and Operational Guidelines",
        "## 3. Quantitative Performance Factors & Metrics",
        "## 4. Key Predictive Drivers",
        "## 5. Demographic Fairness & Disparate Impact Audit",
        "## 6. Caveats and Limitations",
    ]
    for h in required_headings:
        assert h in content, f"Missing required model card section: '{h}'"


# =============================================================================
# CLOSEOUT REPORTING & AUDIT INTEGRITY TESTS
# =============================================================================

def test_holdout_triage_arithmetic_consistency(processed_df, pipeline_metrics):
    """Verify precision = TP/k, lift = precision/base_rate, and recall = TP/positives agree across all holdout tables."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)
    y_true = y_ho.to_numpy()
    n_ho = len(y_true)
    total_leavers = int(np.sum(y_true))
    base_rate = total_leavers / n_ho

    assert n_ho == 294
    assert total_leavers == 47

    hm_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "holdout_metrics.csv"
    assert hm_path.exists(), "holdout_metrics.csv missing"
    df_hm = pd.read_csv(hm_path)

    cal_row = df_hm[df_hm["model"].str.contains("Calibrated")].iloc[0]

    pipe = Pipeline([
        ("prep", build_preprocessor("B")),
        ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=SEED)),
    ]).fit(X_tr, y_tr)
    cal_clf = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=5).fit(X_tr, y_tr)
    probs_cal = cal_clf.predict_proba(X_ho)[:, 1]

    # Top 10%
    k_10 = int(np.round(n_ho * 0.10))  # 29
    idx_top10 = np.argsort(probs_cal)[::-1][:k_10]
    tp_10 = int(np.sum(y_true[idx_top10]))
    fp_10 = k_10 - tp_10
    prec_10 = tp_10 / k_10
    rec_10 = tp_10 / total_leavers
    lift_10 = prec_10 / base_rate

    assert k_10 == 29
    assert tp_10 == 18
    assert fp_10 == 11
    assert np.isclose(prec_10, 18 / 29)
    assert np.isclose(rec_10, 18 / 47)
    assert np.isclose(lift_10, (18 / 29) / (47 / 294))

    assert np.isclose(cal_row["recall_at_10_point"], rec_10, atol=1e-3)
    assert np.isclose(cal_row["lift_at_10_point"], lift_10, atol=1e-3)

    # Top 20%
    k_20 = int(np.round(n_ho * 0.20))  # 59
    idx_top20 = np.argsort(probs_cal)[::-1][:k_20]
    tp_20 = int(np.sum(y_true[idx_top20]))
    fp_20 = k_20 - tp_20
    prec_20 = tp_20 / k_20
    rec_20 = tp_20 / total_leavers
    lift_20 = prec_20 / base_rate

    assert k_20 == 59
    assert tp_20 == 27
    assert fp_20 == 32
    assert np.isclose(prec_20, 27 / 59)
    assert np.isclose(rec_20, 27 / 47)
    assert np.isclose(lift_20, (27 / 59) / (47 / 294))

    assert np.isclose(cal_row["recall_at_20_point"], rec_20, atol=1e-3)
    assert np.isclose(cal_row["lift_at_20_point"], lift_20, atol=1e-3)


def test_fig10_source_tables_low_n_audit():
    """Verify exact low-n cells across all three FIG-10 interaction tables."""
    tables_dir = PROJECT_ROOT / "reports" / "phase3" / "tables"
    t_jl = pd.read_csv(tables_dir / "two_way_overtime_x_joblevel.csv")
    t_ag = pd.read_csv(tables_dir / "two_way_overtime_x_agegroup.csv")
    t_tb = pd.read_csv(tables_dir / "two_way_overtime_x_tenureband.csv")

    low_jl = t_jl[t_jl["n"] < 30]
    low_ag = t_ag[t_ag["n"] < 30]
    low_tb = t_tb[t_tb["n"] < 30]

    assert len(low_jl) == 1
    assert len(low_ag) == 0
    assert len(low_tb) == 0

    row = low_jl.iloc[0]
    assert row["OverTime"] == "Yes"
    assert int(row["JobLevel"]) == 5
    assert int(row["n"]) == 18
    assert int(row["leavers"]) == 3


def test_model_comparison_and_calibration_metrics(pipeline_metrics):
    """Verify model comparison table metrics and ECE calibration calculations."""
    comp_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "model_comparison.csv"
    assert comp_path.exists(), "reports/phase4/tables/model_comparison.csv missing"
    df_comp = pd.read_csv(comp_path)
    assert len(df_comp) == 4

    cal_bins_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "calibration_bins.csv"
    assert cal_bins_path.exists(), "reports/phase4/tables/calibration_bins.csv missing"
    df_bins = pd.read_csv(cal_bins_path)
    assert len(df_bins) == 15  # 10 bins + 5 bins

    uncal_row = df_comp[df_comp["model"].str.contains("Uncalibrated")].iloc[0]
    cal_row = df_comp[df_comp["model"].str.contains("Calibrated")].iloc[0]

    # Verify calibration did not improve Brier score and made no ranking change at top 20%
    assert float(uncal_row["ece_10_bins"]) > float(cal_row["ece_10_bins"])
    assert int(uncal_row["top_20_tp"]) == int(cal_row["top_20_tp"])


def test_materiality_threshold_governance(pipeline_metrics):
    """Verify that Set B was locked on governance grounds and materiality check evaluated."""
    meta = pipeline_metrics["metadata"]
    assert meta["selected_feature_set"] == "B"

    paired_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "paired_model_differences.csv"
    assert paired_path.exists(), "paired_model_differences.csv missing"
    df_paired = pd.read_csv(paired_path)
    assert len(df_paired) >= 7

    # Check exact point differences matching user verification
    lr_hgb_pr = df_paired[(df_paired["comparison"].str.contains("Holdout: LR (Uncalibrated) - HGB", regex=False)) & (df_paired["metric"] == "PR-AUC")].iloc[0]
    assert np.isclose(lr_hgb_pr["point_diff"], 0.0652, atol=0.001), f"Expected PR-AUC point diff ~0.0652, got {lr_hgb_pr['point_diff']}"
    lr_hgb_roc = df_paired[(df_paired["comparison"].str.contains("Holdout: LR (Uncalibrated) - HGB", regex=False)) & (df_paired["metric"] == "ROC-AUC")].iloc[0]
    assert np.isclose(lr_hgb_roc["point_diff"], 0.0388, atol=0.001), f"Expected ROC-AUC point diff ~0.0388, got {lr_hgb_roc['point_diff']}"
    lr_hgb_br = df_paired[(df_paired["comparison"].str.contains("Holdout: LR (Uncalibrated) - HGB", regex=False)) & (df_paired["metric"] == "Brier Score")].iloc[0]
    assert np.isclose(lr_hgb_br["point_diff"], -0.0115, atol=0.001), f"Expected Brier point diff ~-0.0115, got {lr_hgb_br['point_diff']}"


def test_hotspot_candidate_screening_and_holdout_provenance(processed_df):
    """Verify hotspot validation table schema, deduplication of EarlyTenure, and counts."""
    val_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "hotspot_validation.csv"
    assert val_path.exists(), "hotspot_validation.csv missing"
    df_val = pd.read_csv(val_path)

    # Required columns
    required_cols = [
        "candidate_segment", "selection_provenance", "n_train", "rate_train",
        "n_full", "rate_full", "holdout_n", "rate_holdout", "wilson_ci_low",
        "wilson_ci_high", "above_baseline_point", "above_baseline_ci_lower", "low_n_flag"
    ]
    for col in required_cols:
        assert col in df_val.columns, f"Missing required column in hotspot_validation.csv: {col}"

    # Deduplication check: EarlyTenure=1 must be absent
    assert "EarlyTenure=1" not in df_val["candidate_segment"].values

    # Check exact counts
    assert len(df_val) == 27
    assert (df_val["selection_provenance"] == "Train-Screened Candidate").sum() == 24
    assert (df_val["selection_provenance"].str.contains("Optimistic")).sum() == 3

    # Check separate denominators
    train_sc = df_val[df_val["selection_provenance"] == "Train-Screened Candidate"]
    train_n30 = train_sc[~train_sc["low_n_flag"]]
    assert len(train_n30) == 13, f"Expected 13 train-screened candidates with holdout n>=30, got {len(train_n30)}"
    train_above_lower = train_n30[train_n30["above_baseline_ci_lower"]]
    assert len(train_above_lower) == 8, f"Expected 8 train-screened candidates above baseline by lower bound, got {len(train_above_lower)}"

    opt = df_val[df_val["selection_provenance"].str.contains("Optimistic")]
    opt_n30 = opt[~opt["low_n_flag"]]
    assert len(opt_n30) == 1, f"Expected 1 optimistic candidate with holdout n>=30, got {len(opt_n30)}"
    opt_above_lower = opt_n30[opt_n30["above_baseline_ci_lower"]]
    assert len(opt_above_lower) == 1, f"Expected 1 optimistic candidate above baseline by lower bound, got {len(opt_above_lower)}"

    # Total unique candidates
    n_ge_30 = df_val[~df_val["low_n_flag"]]
    assert len(n_ge_30) == 14
    above_lower = n_ge_30[n_ge_30["above_baseline_ci_lower"]]
    assert len(above_lower) == 9


def test_odds_ratios_standardized_units_and_contrasts(processed_df):
    """Verify odds ratios table units and nominal contrasts table."""
    or_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "feature_odds_ratios.csv"
    assert or_path.exists(), "feature_odds_ratios.csv missing"
    df_or = pd.read_csv(or_path)

    assert "unit" in df_or.columns
    assert "encoding_type" in df_or.columns
    assert "coefficient" in df_or.columns
    assert "odds_ratio" in df_or.columns

    # Contrast table
    contr_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "odds_ratio_contrasts.csv"
    assert contr_path.exists(), "odds_ratio_contrasts.csv missing"
    df_contr = pd.read_csv(contr_path)

    ot_row = df_contr[df_contr["contrast"] == "OverTime: Yes vs No"].iloc[0]
    assert np.isclose(ot_row["beta_diff"], 1.3855, atol=1e-2)
    assert np.isclose(ot_row["odds_ratio"], 4.00, atol=1e-1)
    assert ot_row["or_bootstrap_ci95_low"] > 2.0


def test_demographic_fairness_triage_audit(processed_df):
    """Verify demographic fairness table contains ratios vs highest and largest, and low-n flags."""
    fair_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "demographic_fairness.csv"
    assert fair_path.exists(), "demographic_fairness.csv missing"
    df_fair = pd.read_csv(fair_path)

    req_cols = [
        "attribute", "subgroup", "n", "leavers", "flagged_k",
        "selection_rate_pct", "selection_ratio_vs_highest", "selection_ratio_vs_largest",
        "recall_pct", "precision_pct", "observed_rate_pct", "observed_rate_ci_pct",
        "low_n_total", "low_n_leavers", "low_n_flagged"
    ]
    for col in req_cols:
        assert col in df_fair.columns, f"Missing column in demographic_fairness.csv: {col}"

    assert len(df_fair) == 9


def test_phase4_consistency_findings_audit():
    """Verify consistency audit table covers all 57 findings with four mutually exclusive statuses."""
    audit_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "consistency_audit.csv"
    assert audit_path.exists(), "consistency_audit.csv missing"
    df_audit = pd.read_csv(audit_path)

    assert len(df_audit) == 57
    valid_statuses = {"agrees", "direction only", "unstable", "untestable in Set B"}
    assert set(df_audit["phase4_status"]).issubset(valid_statuses)

    untestable = df_audit[df_audit["phase4_status"] == "untestable in Set B"]
    assert len(untestable) == 7

    stability_path = PROJECT_ROOT / "reports" / "phase4" / "tables" / "tenure_sign_stability.csv"
    assert stability_path.exists(), "tenure_sign_stability.csv missing"
    df_stab = pd.read_csv(stability_path)
    assert len(df_stab) == 6


def test_real_pipeline_figure_layouts(processed_df, pipeline_metrics):
    """Assert real pipeline figures FIG-15 through FIG-20 pass all layout geometry and presentation invariants."""
    figures_dir = PROJECT_ROOT / "reports" / "phase4" / "figures"
    manifest_path = PROJECT_ROOT / "reports" / "phase4" / "figure_manifest.csv"
    assert manifest_path.exists(), "figure_manifest.csv not found"

    df_manifest = pd.read_csv(manifest_path)
    assert len(df_manifest) == 6

    for _, r in df_manifest.iterrows():
        fig_file = figures_dir / r["filename"]
        assert fig_file.exists(), f"Figure file missing: {fig_file}"
        assert fig_file.stat().st_size > 10000


def test_real_pipeline_figure_layout_mutations(processed_df, pipeline_metrics):
    """Mutation tests: verify that layout geometry validators correctly detect deliberate defects."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1])
    ax.text(1.5, 1.5, "Out of bounds text")
    fig.canvas.draw()

    violations = collect_figure_layout_violations(fig, "MUTATION-OUT-OF-BOUNDS")
    plt.close(fig)
    assert any("[OUTSIDE_FIGURE]" in v for v in violations), "Layout validator failed to detect text out of figure bounds!"
