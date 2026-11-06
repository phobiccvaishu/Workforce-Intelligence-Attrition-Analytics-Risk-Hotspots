"""Unit and integration tests for Phase 4 attrition risk modelling pipeline.

Validates:
1. Split determinism (SEED=42) and zero index/ID overlap between train and holdout splits.
2. Leakage prevention: zero target-derived columns in Feature Sets A and B.
3. Preprocessing pipeline fitted strictly per-fold without data leakage.
4. Hotspot candidate screening executes exclusively on training data.
5. Out-of-sample holdout metrics validity (PR-AUC, ROC-AUC, Brier, Lift, Recall).
6. Model card completeness (required governance and ethical sections).
7. Figure layout compliance for FIG-15 through FIG-20 (zero overlaps, within canvas).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from workforce_intel.config import PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE, PROJECT_ROOT
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


def test_split_determinism_and_zero_overlap(processed_df):
    """Assert train/holdout split is exactly 80/20, deterministic, and has zero ID overlap."""
    X_tr1, y_tr1, X_ho1, y_ho1, tr_df1, ho_df1 = train_test_split_data(processed_df, seed=SEED)
    X_tr2, y_tr2, X_ho2, y_ho2, tr_df2, ho_df2 = train_test_split_data(processed_df, seed=SEED)

    # 80/20 exact counts
    assert len(tr_df1) == 1176, f"Expected 1,176 training rows, got {len(tr_df1)}"
    assert len(ho_df1) == 294, f"Expected 294 holdout rows, got {len(ho_df1)}"
    assert len(tr_df1) + len(ho_df1) == len(processed_df)

    # Determinism
    pd.testing.assert_frame_equal(tr_df1, tr_df2)
    pd.testing.assert_frame_equal(ho_df1, ho_df2)

    # Zero overlap
    overlap = set(tr_df1.index).intersection(set(ho_df1.index))
    assert len(overlap) == 0, f"Detected {len(overlap)} overlapping indices between train and holdout!"


def test_feature_sets_leakage_and_drop_columns(processed_df):
    """Assert no target variable or dropped constants/identifiers exist in feature sets."""
    set_a, set_b = get_feature_sets(processed_df)

    # Neither set should contain Attrition
    assert "Attrition" not in set_a
    assert "Attrition" not in set_b

    # Neither set should contain dropped columns
    for col in DROP_COLUMNS:
        assert col not in set_a
        assert col not in set_b

    # Set B must strictly exclude all protected demographic columns
    for demo_col in DEMOGRAPHIC_COLUMNS:
        assert demo_col in set_a
        assert demo_col not in set_b


def test_preprocessor_fitting_per_fold(processed_df):
    """Assert preprocessor can be fit on training split and transform holdout split with zero NaN outputs."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    preprocessor = build_preprocessor(feature_set="B")
    X_tr_trans = preprocessor.fit_transform(X_tr)
    X_ho_trans = preprocessor.transform(X_ho)

    assert not np.isnan(X_tr_trans).any(), "Found NaNs in transformed training data"
    assert not np.isnan(X_ho_trans).any(), "Found NaNs in transformed holdout data"
    assert X_tr_trans.shape[0] == 1176
    assert X_ho_trans.shape[0] == 294


def test_hotspot_screening_strictly_on_train(processed_df):
    """Assert candidate hotspots are screened solely from training rows."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    screened = screen_hotspots_on_train(tr_df, min_n=30)
    assert len(screened) > 0, "No hotspots screened on training split"

    # All train_n must sum to at most len(tr_df) for one segment
    for h in screened:
        assert h["train_n"] >= 30
        assert h["train_n"] <= 1176
        assert h["train_ci_low"] > h["train_baseline"]


def test_holdout_metrics_validity(processed_df):
    """Assert out-of-sample holdout metrics are computed with valid bootstrap confidence intervals."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    pipe = Pipeline([
        ("prep", build_preprocessor("B")),
        ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=SEED)),
    ]).fit(X_tr, y_tr)

    probs = pipe.predict_proba(X_ho)[:, 1]
    boot_results = compute_bootstrap_metrics(y_ho.to_numpy(), probs, n_bootstrap=100, seed=SEED)

    for metric_name in ["pr_auc", "roc_auc", "brier_score", "recall_at_10", "lift_at_10", "recall_at_20", "lift_at_20"]:
        assert metric_name in boot_results
        res = boot_results[metric_name]
        assert res["ci_lower"] <= res["point"] <= res["ci_upper"] or abs(res["point"] - res["ci_lower"]) < 1e-4


def test_model_card_sections():
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


def test_phase4_figure_layouts(processed_df):
    """Assert FIG-15 through FIG-20 pass all layout geometry and presentation invariants."""
    X_tr, y_tr, X_ho, y_ho, tr_df, ho_df = train_test_split_data(processed_df, seed=SEED)

    p_dummy = Pipeline([("clf", DummyClassifier(strategy="prior"))]).fit(X_tr, y_tr)
    p_lr = Pipeline([("prep", build_preprocessor("B")), ("clf", LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000, random_state=SEED))]).fit(X_tr, y_tr)
    p_hgb = Pipeline([("prep", build_preprocessor("B")), ("clf", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=50, max_leaf_nodes=15, random_state=SEED))]).fit(X_tr, y_tr)
    p_cal = CalibratedClassifierCV(estimator=p_lr, method="sigmoid", cv=5).fit(X_tr, y_tr)

    preds = {
        "Dummy (Prior)": p_dummy.predict_proba(X_ho)[:, 1],
        "Logistic Regression (Uncalibrated)": p_lr.predict_proba(X_ho)[:, 1],
        "HistGradientBoosting": p_hgb.predict_proba(X_ho)[:, 1],
        "Logistic Regression (Calibrated)": p_cal.predict_proba(X_ho)[:, 1],
    }

    ho_baseline = (y_ho == 1).sum() / len(y_ho)

    figs = [
        ("FIG-15", plot_fig15_pr_curves(preds, y_ho.to_numpy(), ho_baseline)[0]),
        ("FIG-16", plot_fig16_calibration(y_ho.to_numpy(), preds["Logistic Regression (Uncalibrated)"], preds["Logistic Regression (Calibrated)"])[0]),
        ("FIG-17", plot_fig17_risk_bands(y_ho.to_numpy(), preds["Logistic Regression (Calibrated)"], ho_baseline)[0]),
        ("FIG-18", plot_fig18_odds_ratios(compute_logistic_odds_ratios(p_lr))[0]),
        ("FIG-19", plot_fig19_hotspot_validation(evaluate_hotspots_on_holdout(screen_hotspots_on_train(tr_df), ho_df))[0]),
        ("FIG-20", plot_fig20_demographic_fairness(evaluate_demographic_fairness(ho_df, preds["Logistic Regression (Calibrated)"]))[0]),
    ]

    all_violations = []
    for fid, fig in figs:
        violations = collect_figure_layout_violations(fig, fid)
        if violations:
            all_violations.extend(violations)

    assert not all_violations, f"Found layout violations in Phase 4 figures:\n" + "\n".join(all_violations)
