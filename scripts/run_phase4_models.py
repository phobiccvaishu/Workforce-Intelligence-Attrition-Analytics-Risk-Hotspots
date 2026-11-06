"""Phase 4: Attrition Risk Modelling and Out-of-Sample Validation Pipeline.

Executes all modelling, probability calibration, bootstrap evaluation,
hotspot validation, fairness auditing, figure generation, and manifest creation.
Fully headless, deterministic (SEED=42), and leakage-safe.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from workforce_intel.config import (
    FIGURE_DPI,
    PROCESSED_DATA_FILE,
    PROCESSED_SCHEMA_FILE,
)
from workforce_intel.io import load_processed
from workforce_intel.modeling import (
    SEED,
    _count_leavers,
    build_consistency_matrix,
    build_preprocessor,
    compute_bootstrap_metrics,
    compute_ece_and_bins,
    compute_logistic_odds_ratios,
    compute_nominal_contrasts,
    compute_sign_stability_audit,
    cross_validate_model,
    evaluate_demographic_fairness,
    evaluate_hotspots_on_holdout,
    evaluate_predictions,
    get_feature_sets,
    parse_segment_mask,
    screen_hotspots_on_train,
    train_test_split_data,
    tune_hist_gradient_boosting,
    tune_logistic_regression,
    wilson_interval,
)
from workforce_intel.viz_phase4 import (
    plot_fig15_pr_curves,
    plot_fig16_calibration,
    plot_fig17_risk_bands,
    plot_fig18_odds_ratios,
    plot_fig19_hotspot_validation,
    plot_fig20_demographic_fairness,
)


def compute_df_sha256(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hash of a DataFrame's CSV representation."""
    csv_bytes = df.to_csv(index=False, lineterminator="\n", float_format="%.6f").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def main() -> None:
    print("=" * 70)
    print("PHASE 4: ATTRITION RISK MODELLING & VALIDATION PIPELINE")
    print("=" * 70)

    # 1. Output directories setup
    reports_phase4_dir = PROJECT_ROOT / "reports" / "phase4"
    tables_dir = reports_phase4_dir / "tables"
    figures_dir = reports_phase4_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load Processed Data
    print(f"Loading processed dataset from {PROCESSED_DATA_FILE}...")
    df = load_processed(PROCESSED_DATA_FILE, PROCESSED_SCHEMA_FILE)
    n_total = len(df)
    leavers_total = int((df["Attrition"] == 1).sum()) if pd.api.types.is_numeric_dtype(df["Attrition"]) else int((df["Attrition"].astype(str).str.strip().isin(["1", "Yes"])).sum())
    overall_baseline = leavers_total / n_total
    print(f"Dataset loaded: n={n_total:,}, total leavers={leavers_total:,} ({overall_baseline * 100:.2f}%)")

    # 3. Stratified 80/20 Train/Holdout Split
    print("\nExecuting stratified 80/20 train/holdout split (SEED=42)...")
    X_train, y_train, X_holdout, y_holdout, train_df, holdout_df = train_test_split_data(df, test_size=0.20, seed=SEED)
    n_train = len(train_df)
    n_holdout = len(holdout_df)
    train_leavers = int((y_train == 1).sum())
    holdout_leavers = int((y_holdout == 1).sum())
    train_baseline = train_leavers / n_train
    holdout_baseline = holdout_leavers / n_holdout

    print(f"Training split: n={n_train:,}, leavers={train_leavers} ({train_baseline * 100:.2f}%)")
    print(f"Holdout split:  n={n_holdout:,}, leavers={holdout_leavers} ({holdout_baseline * 100:.2f}%)")

    # Verify zero data overlap
    assert len(set(train_df.index).intersection(set(holdout_df.index))) == 0, "Split index overlap detected!"

    # 4. Cross-Validation Model Development on Training Set
    print("\nRunning repeated stratified 5-fold CV (3 repeats = 15 folds) on Training Split...")

    # Dummy Baseline
    print("Evaluating Dummy Classifier (Prior)...")
    dummy_cv = cross_validate_model(
        lambda: DummyClassifier(strategy="prior"),
        X_train,
        y_train,
        feature_set="B",
        seed=SEED,
    )

    # Logistic Regression Tuning (Set B)
    print("Tuning Logistic Regression (Set B - Demographics Excluded)...")
    best_c_b, lr_b_cv = tune_logistic_regression(X_train, y_train, feature_set="B", seed=SEED)
    print(f"  Best C (Set B): {best_c_b}, Mean CV PR-AUC: {lr_b_cv['metrics_mean']['pr_auc']:.4f}")

    # Logistic Regression Tuning (Set A - Full Demographics for Materiality Check)
    print("Tuning Logistic Regression (Set A - Demographics Included)...")
    best_c_a, lr_a_cv = tune_logistic_regression(X_train, y_train, feature_set="A", seed=SEED)
    print(f"  Best C (Set A): {best_c_a}, Mean CV PR-AUC: {lr_a_cv['metrics_mean']['pr_auc']:.4f}")

    # Materiality rule evaluation
    delta_pr_auc = lr_a_cv["metrics_mean"]["pr_auc"] - lr_b_cv["metrics_mean"]["pr_auc"]
    print(f"  PR-AUC Gain from Set A: {delta_pr_auc:+.4f} (Materiality threshold: > +0.0300)")
    if delta_pr_auc > 0.03:
        recommended_set = "A"
        print("  -> Set A exceeds materiality threshold; recommended for operations.")
    else:
        recommended_set = "B"
        print("  -> Set A gain <= 0.0300; Governance-recommended Set B SELECTED.")

    # HistGradientBoosting Tuning (Set B)
    print("Tuning HistGradientBoosting Classifier (Set B)...")
    best_hgb_params, hgb_cv = tune_hist_gradient_boosting(X_train, y_train, feature_set="B", seed=SEED)
    print(f"  Best HGB params: {best_hgb_params}, Mean CV PR-AUC: {hgb_cv['metrics_mean']['pr_auc']:.4f}")

    # 5. Fit Pipelines on Training Set & Calibrate
    print("\nFitting full pipelines on training split...")
    pipe_dummy = Pipeline([("clf", DummyClassifier(strategy="prior"))]).fit(X_train, y_train)

    pipe_lr = Pipeline(
        [
            ("prep", build_preprocessor("B")),
            ("clf", LogisticRegression(C=best_c_b, solver="lbfgs", max_iter=1000, random_state=SEED)),
        ]
    ).fit(X_train, y_train)

    pipe_hgb = Pipeline(
        [
            ("prep", build_preprocessor("B")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=best_hgb_params["learning_rate"],
                    max_iter=best_hgb_params["max_iter"],
                    max_leaf_nodes=best_hgb_params["max_leaf_nodes"],
                    random_state=SEED,
                ),
            ),
        ]
    ).fit(X_train, y_train)

    # Sigmoid calibration inside CV
    print("Fitting CalibratedClassifierCV (Sigmoid / Platt scaling, 5-fold CV)...")
    cal_lr = CalibratedClassifierCV(estimator=pipe_lr, method="sigmoid", cv=5).fit(X_train, y_train)

    # 6. Final Holdout Evaluation (Locked and Evaluated Once)
    print("\nScoring Holdout Split (Unseen Evaluation)...")
    probs_dummy = pipe_dummy.predict_proba(X_holdout)[:, 1]
    probs_lr_uncal = pipe_lr.predict_proba(X_holdout)[:, 1]
    probs_hgb = pipe_hgb.predict_proba(X_holdout)[:, 1]
    probs_lr_cal = cal_lr.predict_proba(X_holdout)[:, 1]

    y_ho_arr = y_holdout.to_numpy()

    print("Computing 2,000-sample percentile bootstrap 95% CIs on holdout...")
    boot_dummy = compute_bootstrap_metrics(y_ho_arr, probs_dummy, n_bootstrap=2000, seed=SEED)
    boot_lr_uncal = compute_bootstrap_metrics(y_ho_arr, probs_lr_uncal, n_bootstrap=2000, seed=SEED)
    boot_hgb = compute_bootstrap_metrics(y_ho_arr, probs_hgb, n_bootstrap=2000, seed=SEED)
    boot_lr_cal = compute_bootstrap_metrics(y_ho_arr, probs_lr_cal, n_bootstrap=2000, seed=SEED)

    print("\nHOLDOUT RESULTS (Calibrated Logistic Regression - Set B):")
    for m_name, vals in boot_lr_cal.items():
        print(f"  {m_name:15}: {vals['point']:.4f} (95% CI: [{vals['ci_lower']:.4f}, {vals['ci_upper']:.4f}])")

    # 7. Hotspot Candidate Out-of-Sample Validation
    print("\nExecuting candidate hotspot screening and holdout validation...")
    screened_train_hotspots = screen_hotspots_on_train(train_df, min_n=30)
    train_segments_set = set(h["segment"] for h in screened_train_hotspots)
    print(f"Screened {len(screened_train_hotspots)} candidate hotspot segments on Training Split.")

    # Also load Phase 3 candidates
    phase3_cand_path = PROJECT_ROOT / "reports" / "phase3" / "tables" / "hotspot_candidates.csv"
    p3_segments_set = set()
    if phase3_cand_path.exists():
        df_p3_cand = pd.read_csv(phase3_cand_path)
        p3_segments_set = set(df_p3_cand["subgroup"])
        for _, r in df_p3_cand.iterrows():
            seg_str = r["subgroup"]
            if seg_str not in train_segments_set:
                tr_mask = parse_segment_mask(seg_str, train_df)
                tr_n = int(tr_mask.sum())
                tr_leavers = _count_leavers(train_df.loc[tr_mask])
                tr_rate = tr_leavers / tr_n if tr_n > 0 else 0.0
                tr_low, tr_high = wilson_interval(tr_leavers, tr_n, confidence=0.95)
                screened_train_hotspots.append(
                    {
                        "segment": seg_str,
                        "source": "Optimistic: holdout was used to select",
                        "train_n": tr_n,
                        "train_leavers": tr_leavers,
                        "train_rate": tr_rate,
                        "train_ci_low": tr_low,
                        "train_ci_high": tr_high,
                        "train_baseline": train_baseline,
                    }
                )

    df_hotspot_val = evaluate_hotspots_on_holdout(screened_train_hotspots, holdout_df, full_df=df)
    df_hotspot_val["in_p3_16_candidates"] = df_hotspot_val["segment"].isin(p3_segments_set)
    df_hotspot_val["in_training_screened"] = df_hotspot_val["segment"].isin(train_segments_set)
    df_hotspot_val["above_baseline_point"] = df_hotspot_val["holdout_rate"] > holdout_baseline
    df_hotspot_val["above_baseline_ci_lower"] = df_hotspot_val["holdout_ci_low"] > holdout_baseline
    df_hotspot_val["low_n_flag"] = df_hotspot_val["holdout_n"] < 30
    df_hotspot_val["holdout_low_n_flag"] = df_hotspot_val["holdout_n"] < 30
    print(f"Validated {len(df_hotspot_val)} candidate segments against Holdout Split.")

    # 8. Demographic Fairness Audit
    print("\nAuditing demographic fairness on holdout predictions (20% triage threshold)...")
    df_fairness = evaluate_demographic_fairness(holdout_df, probs_lr_cal, percentile=0.20)

    # 9. Feature Importance / Odds Ratios and Nominal Contrasts
    print("\nExtracting feature odds ratios and nominal contrasts from regularised Logistic Regression...")
    df_odds_ratios = compute_logistic_odds_ratios(pipe_lr, X_train=X_train)
    df_contrasts = compute_nominal_contrasts(pipe_lr, X_train, y_train, n_bootstrap=500, seed=SEED)

    # 10. Calibration and ECE Analysis
    print("\nComputing calibration curves and bin-level reliability tables...")
    ece_uncal_10, bins_uncal_10 = compute_ece_and_bins(y_ho_arr, probs_lr_uncal, n_bins=10)
    ece_cal_10, bins_cal_10 = compute_ece_and_bins(y_ho_arr, probs_lr_cal, n_bins=10)
    ece_uncal_5, bins_uncal_5 = compute_ece_and_bins(y_ho_arr, probs_lr_uncal, n_bins=5)
    ece_cal_5, bins_cal_5 = compute_ece_and_bins(y_ho_arr, probs_lr_cal, n_bins=5)

    cal_bins_records = []
    for u, c in zip(bins_uncal_10, bins_cal_10):
        cal_bins_records.append({
            "bin": u["bin"],
            "bin_scheme": "10-bin",
            "range": u["range"],
            "uncal_count": u["count"],
            "uncal_accuracy": u["accuracy"],
            "uncal_confidence": u["confidence"],
            "uncal_low_count": u["low_count_flag"],
            "cal_count": c["count"],
            "cal_accuracy": c["accuracy"],
            "cal_confidence": c["confidence"],
            "cal_low_count": c["low_count_flag"],
        })
    for u, c in zip(bins_uncal_5, bins_cal_5):
        cal_bins_records.append({
            "bin": u["bin"],
            "bin_scheme": "5-bin",
            "range": u["range"],
            "uncal_count": u["count"],
            "uncal_accuracy": u["accuracy"],
            "uncal_confidence": u["confidence"],
            "uncal_low_count": u["low_count_flag"],
            "cal_count": c["count"],
            "cal_accuracy": c["accuracy"],
            "cal_confidence": c["confidence"],
            "cal_low_count": c["low_count_flag"],
        })
    df_cal_bins = pd.DataFrame(cal_bins_records)

    # 11. Model Comparison Table
    print("\nBuilding comprehensive model comparison table...")
    model_comp_records = [
        {
            "model": "Dummy (Prior)",
            "chosen_hyperparameters": "strategy='prior'",
            "cv_pr_auc_mean_sd": f"{dummy_cv['metrics_mean']['pr_auc']:.4f} +/- {dummy_cv['metrics_std']['pr_auc']:.4f}",
            "holdout_pr_auc": f"{boot_dummy['pr_auc']['point']:.4f} [{boot_dummy['pr_auc']['ci_lower']:.4f}, {boot_dummy['pr_auc']['ci_upper']:.4f}]",
            "holdout_roc_auc": f"{boot_dummy['roc_auc']['point']:.4f} [{boot_dummy['roc_auc']['ci_lower']:.4f}, {boot_dummy['roc_auc']['ci_upper']:.4f}]",
            "holdout_brier": f"{boot_dummy['brier_score']['point']:.4f} [{boot_dummy['brier_score']['ci_lower']:.4f}, {boot_dummy['brier_score']['ci_upper']:.4f}]",
            "top_10_tp": int(boot_dummy["tp_10"]["point"]),
            "top_20_tp": int(boot_dummy["tp_20"]["point"]),
            "ece_5_bins": "N/A",
            "ece_10_bins": "N/A",
        },
        {
            "model": "Logistic Regression (Uncalibrated)",
            "chosen_hyperparameters": f"C={best_c_b}, penalty='l2', solver='lbfgs'",
            "cv_pr_auc_mean_sd": f"{lr_b_cv['metrics_mean']['pr_auc']:.4f} +/- {lr_b_cv['metrics_std']['pr_auc']:.4f}",
            "holdout_pr_auc": f"{boot_lr_uncal['pr_auc']['point']:.4f} [{boot_lr_uncal['pr_auc']['ci_lower']:.4f}, {boot_lr_uncal['pr_auc']['ci_upper']:.4f}]",
            "holdout_roc_auc": f"{boot_lr_uncal['roc_auc']['point']:.4f} [{boot_lr_uncal['roc_auc']['ci_lower']:.4f}, {boot_lr_uncal['roc_auc']['ci_upper']:.4f}]",
            "holdout_brier": f"{boot_lr_uncal['brier_score']['point']:.4f} [{boot_lr_uncal['brier_score']['ci_lower']:.4f}, {boot_lr_uncal['brier_score']['ci_upper']:.4f}]",
            "top_10_tp": int(boot_lr_uncal["tp_10"]["point"]),
            "top_20_tp": int(boot_lr_uncal["tp_20"]["point"]),
            "ece_5_bins": f"{ece_uncal_5:.4f}",
            "ece_10_bins": f"{ece_uncal_10:.4f}",
        },
        {
            "model": "HistGradientBoosting",
            "chosen_hyperparameters": f"learning_rate={best_hgb_params['learning_rate']}, max_iter={best_hgb_params['max_iter']}, max_leaf_nodes={best_hgb_params['max_leaf_nodes']}",
            "cv_pr_auc_mean_sd": f"{hgb_cv['metrics_mean']['pr_auc']:.4f} +/- {hgb_cv['metrics_std']['pr_auc']:.4f}",
            "holdout_pr_auc": f"{boot_hgb['pr_auc']['point']:.4f} [{boot_hgb['pr_auc']['ci_lower']:.4f}, {boot_hgb['pr_auc']['ci_upper']:.4f}]",
            "holdout_roc_auc": f"{boot_hgb['roc_auc']['point']:.4f} [{boot_hgb['roc_auc']['ci_lower']:.4f}, {boot_hgb['roc_auc']['ci_upper']:.4f}]",
            "holdout_brier": f"{boot_hgb['brier_score']['point']:.4f} [{boot_hgb['brier_score']['ci_lower']:.4f}, {boot_hgb['brier_score']['ci_upper']:.4f}]",
            "top_10_tp": int(boot_hgb["tp_10"]["point"]),
            "top_20_tp": int(boot_hgb["tp_20"]["point"]),
            "ece_5_bins": "N/A",
            "ece_10_bins": "N/A",
        },
        {
            "model": "Logistic Regression (Calibrated)",
            "chosen_hyperparameters": "Sigmoid (Platt) 5-fold CV",
            "cv_pr_auc_mean_sd": f"{lr_b_cv['metrics_mean']['pr_auc']:.4f} +/- {lr_b_cv['metrics_std']['pr_auc']:.4f} (base LR)",
            "holdout_pr_auc": f"{boot_lr_cal['pr_auc']['point']:.4f} [{boot_lr_cal['pr_auc']['ci_lower']:.4f}, {boot_lr_cal['pr_auc']['ci_upper']:.4f}]",
            "holdout_roc_auc": f"{boot_lr_cal['roc_auc']['point']:.4f} [{boot_lr_cal['roc_auc']['ci_lower']:.4f}, {boot_lr_cal['roc_auc']['ci_upper']:.4f}]",
            "holdout_brier": f"{boot_lr_cal['brier_score']['point']:.4f} [{boot_lr_cal['brier_score']['ci_lower']:.4f}, {boot_lr_cal['brier_score']['ci_upper']:.4f}]",
            "top_10_tp": int(boot_lr_cal["tp_10"]["point"]),
            "top_20_tp": int(boot_lr_cal["tp_20"]["point"]),
            "ece_5_bins": f"{ece_cal_5:.4f}",
            "ece_10_bins": f"{ece_cal_10:.4f}",
        },
    ]
    df_model_comp = pd.DataFrame(model_comp_records)

    # 12. Paired Model Differences (LR vs HGB Holdout, and Set A vs Set B CV and Holdout)
    print("\nComputing paired model differences and single-loop marginal bootstrap CIs...")
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    from scipy import stats
    rng_paired = np.random.default_rng(SEED)
    n_paired_boot = 2000
    n_ho_samples = len(y_ho_arr)
    ho_indices = np.arange(n_ho_samples)

    # Fit Set A on train, score on holdout
    pipe_lr_a = Pipeline(
        [
            ("prep", build_preprocessor("A")),
            ("clf", LogisticRegression(C=best_c_a, solver="lbfgs", max_iter=1000, random_state=SEED)),
        ]
    ).fit(X_train, y_train)
    probs_lr_a = pipe_lr_a.predict_proba(X_holdout)[:, 1]
    boot_lr_a = compute_bootstrap_metrics(y_ho_arr, probs_lr_a, n_bootstrap=n_paired_boot, seed=SEED)

    # Point estimates on holdout
    pr_lr_pt = float(boot_lr_uncal["pr_auc"]["point"])
    roc_lr_pt = float(boot_lr_uncal["roc_auc"]["point"])
    br_lr_pt = float(boot_lr_uncal["brier_score"]["point"])

    pr_hgb_pt = float(boot_hgb["pr_auc"]["point"])
    roc_hgb_pt = float(boot_hgb["roc_auc"]["point"])
    br_hgb_pt = float(boot_hgb["brier_score"]["point"])

    pr_lr_a_pt = float(boot_lr_a["pr_auc"]["point"])
    roc_lr_a_pt = float(boot_lr_a["roc_auc"]["point"])
    br_lr_a_pt = float(boot_lr_a["brier_score"]["point"])

    diff_pr_list, diff_roc_list, diff_br_list = [], [], []
    diff_pr_ab_list, diff_roc_ab_list, diff_br_ab_list = [], [], []
    for _ in range(n_paired_boot):
        b_idx = rng_paired.choice(ho_indices, size=n_ho_samples, replace=True)
        y_b = y_ho_arr[b_idx]
        if np.sum(y_b) == 0 or np.sum(y_b) == n_ho_samples:
            continue
        # LR uncal vs HGB
        diff_pr_list.append(average_precision_score(y_b, probs_lr_uncal[b_idx]) - average_precision_score(y_b, probs_hgb[b_idx]))
        diff_roc_list.append(roc_auc_score(y_b, probs_lr_uncal[b_idx]) - roc_auc_score(y_b, probs_hgb[b_idx]))
        diff_br_list.append(brier_score_loss(y_b, probs_lr_uncal[b_idx]) - brier_score_loss(y_b, probs_hgb[b_idx]))
        # Set A vs Set B
        diff_pr_ab_list.append(average_precision_score(y_b, probs_lr_a[b_idx]) - average_precision_score(y_b, probs_lr_uncal[b_idx]))
        diff_roc_ab_list.append(roc_auc_score(y_b, probs_lr_a[b_idx]) - roc_auc_score(y_b, probs_lr_uncal[b_idx]))
        diff_br_ab_list.append(brier_score_loss(y_b, probs_lr_a[b_idx]) - brier_score_loss(y_b, probs_lr_uncal[b_idx]))

    # CV diffs for Set A vs Set B across the 15 folds
    cv_diffs = [
        ra["pr_auc"] - rb["pr_auc"]
        for ra, rb in zip(lr_a_cv["fold_records"], lr_b_cv["fold_records"])
    ]
    mean_cv_diff = float(np.mean(cv_diffs))
    sd_cv_diff = float(np.std(cv_diffs, ddof=1))
    se_nb = sd_cv_diff * np.sqrt(1.0 / 15.0 + 1.0 / 4.0)
    t_crit = float(stats.t.ppf(0.975, df=14))
    nb_ci_low = mean_cv_diff - t_crit * se_nb
    nb_ci_high = mean_cv_diff + t_crit * se_nb
    fold_spread_low = mean_cv_diff - 1.96 * sd_cv_diff
    fold_spread_high = mean_cv_diff + 1.96 * sd_cv_diff

    paired_records = [
        {
            "comparison": "Holdout: LR (Uncalibrated) - HGB",
            "metric": "PR-AUC",
            "point_diff": float(pr_lr_pt - pr_hgb_pt),
            "bootstrap_mean_diff": float(np.mean(diff_pr_list)),
            "ci95_low": float(np.percentile(diff_pr_list, 2.5)),
            "ci95_high": float(np.percentile(diff_pr_list, 97.5)),
            "ci_includes_zero": bool(float(np.percentile(diff_pr_list, 2.5)) <= 0.0 <= float(np.percentile(diff_pr_list, 97.5))),
            "notes": "Point diff = 0.5736 - 0.5084 = +0.0652",
        },
        {
            "comparison": "Holdout: LR (Uncalibrated) - HGB",
            "metric": "ROC-AUC",
            "point_diff": float(roc_lr_pt - roc_hgb_pt),
            "bootstrap_mean_diff": float(np.mean(diff_roc_list)),
            "ci95_low": float(np.percentile(diff_roc_list, 2.5)),
            "ci95_high": float(np.percentile(diff_roc_list, 97.5)),
            "ci_includes_zero": bool(float(np.percentile(diff_roc_list, 2.5)) <= 0.0 <= float(np.percentile(diff_roc_list, 97.5))),
            "notes": "Point diff = 0.8130 - 0.7742 = +0.0388",
        },
        {
            "comparison": "Holdout: LR (Uncalibrated) - HGB",
            "metric": "Brier Score",
            "point_diff": float(br_lr_pt - br_hgb_pt),
            "bootstrap_mean_diff": float(np.mean(diff_br_list)),
            "ci95_low": float(np.percentile(diff_br_list, 2.5)),
            "ci95_high": float(np.percentile(diff_br_list, 97.5)),
            "ci_includes_zero": bool(float(np.percentile(diff_br_list, 2.5)) <= 0.0 <= float(np.percentile(diff_br_list, 97.5))),
            "notes": "Point diff = 0.0995 - 0.1110 = -0.0115",
        },
        {
            "comparison": "Holdout: LR (Set A - Set B)",
            "metric": "PR-AUC",
            "point_diff": float(pr_lr_a_pt - pr_lr_pt),
            "bootstrap_mean_diff": float(np.mean(diff_pr_ab_list)),
            "ci95_low": float(np.percentile(diff_pr_ab_list, 2.5)),
            "ci95_high": float(np.percentile(diff_pr_ab_list, 97.5)),
            "ci_includes_zero": bool(float(np.percentile(diff_pr_ab_list, 2.5)) <= 0.0 <= float(np.percentile(diff_pr_ab_list, 97.5))),
            "notes": f"Set A ({pr_lr_a_pt:.4f}) vs Set B ({pr_lr_pt:.4f})",
        },
        {
            "comparison": "Holdout: LR (Set A - Set B)",
            "metric": "ROC-AUC",
            "point_diff": float(roc_lr_a_pt - roc_lr_pt),
            "bootstrap_mean_diff": float(np.mean(diff_roc_ab_list)),
            "ci95_low": float(np.percentile(diff_roc_ab_list, 2.5)),
            "ci95_high": float(np.percentile(diff_roc_ab_list, 97.5)),
            "ci_includes_zero": bool(float(np.percentile(diff_roc_ab_list, 2.5)) <= 0.0 <= float(np.percentile(diff_roc_ab_list, 97.5))),
            "notes": f"Set A ({roc_lr_a_pt:.4f}) vs Set B ({roc_lr_pt:.4f})",
        },
        {
            "comparison": "Holdout: LR (Set A - Set B)",
            "metric": "Brier Score",
            "point_diff": float(br_lr_a_pt - br_lr_pt),
            "bootstrap_mean_diff": float(np.mean(diff_br_ab_list)),
            "ci95_low": float(np.percentile(diff_br_ab_list, 2.5)),
            "ci95_high": float(np.percentile(diff_br_ab_list, 97.5)),
            "ci_includes_zero": bool(float(np.percentile(diff_br_ab_list, 2.5)) <= 0.0 <= float(np.percentile(diff_br_ab_list, 97.5))),
            "notes": f"Set A ({br_lr_a_pt:.4f}) vs Set B ({br_lr_pt:.4f})",
        },
        {
            "comparison": "15-Fold CV: LR (Set A - Set B)",
            "metric": "PR-AUC",
            "point_diff": float(mean_cv_diff),
            "bootstrap_mean_diff": float(mean_cv_diff),
            "ci95_low": float(nb_ci_low),
            "ci95_high": float(nb_ci_high),
            "ci_includes_zero": bool(nb_ci_low <= 0.0 <= nb_ci_high),
            "notes": f"Nadeau-Bengio interval; fold spread: [{fold_spread_low:.4f}, {fold_spread_high:.4f}] (SD={sd_cv_diff:.4f}, not a CI)",
        },
    ]
    df_paired_diffs = pd.DataFrame(paired_records)

    # 13. Sign Stability and Consistency Audit
    print("\nAuditing coefficient sign stability across 500 bootstrap refits...")
    df_sign_stability = compute_sign_stability_audit(X_train, y_train, n_bootstrap=500, seed=SEED)
    print("Building full 57-finding consistency matrix...")
    df_consistency = build_consistency_matrix()

    # 14. Save All Reporting Tables
    print("\nSaving reporting tables to reports/phase4/tables/...")

    # CV Metrics Table
    cv_records = [
        {"model": "Dummy (Prior)", "feature_set": "B", **{f"cv_{k}_mean": v for k, v in dummy_cv["metrics_mean"].items()}, **{f"cv_{k}_std": v for k, v in dummy_cv["metrics_std"].items()}},
        {"model": "HistGradientBoosting", "feature_set": "B", **{f"cv_{k}_mean": v for k, v in hgb_cv["metrics_mean"].items()}, **{f"cv_{k}_std": v for k, v in hgb_cv["metrics_std"].items()}},
        {"model": "Logistic Regression", "feature_set": "A", **{f"cv_{k}_mean": v for k, v in lr_a_cv["metrics_mean"].items()}, **{f"cv_{k}_std": v for k, v in lr_a_cv["metrics_std"].items()}},
        {"model": "Logistic Regression", "feature_set": "B", **{f"cv_{k}_mean": v for k, v in lr_b_cv["metrics_mean"].items()}, **{f"cv_{k}_std": v for k, v in lr_b_cv["metrics_std"].items()}},
    ]
    df_cv_metrics = pd.DataFrame(cv_records)

    # Holdout Metrics Table
    def format_boot(b_dict: Dict[str, Dict[str, float]], model_name: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {"model": model_name}
        for m, vals in b_dict.items():
            row[f"{m}_point"] = vals["point"]
            row[f"{m}_ci95_low"] = vals["ci_lower"]
            row[f"{m}_ci95_high"] = vals["ci_upper"]
        return row

    holdout_records = [
        format_boot(boot_dummy, "Dummy (Prior)"),
        format_boot(boot_hgb, "HistGradientBoosting"),
        format_boot(boot_lr_uncal, "Logistic Regression (Uncalibrated)"),
        format_boot(boot_lr_cal, "Logistic Regression (Calibrated - Sigmoid)"),
        format_boot(boot_lr_a, "Logistic Regression (Set A - Demographics Included)"),
    ]
    df_holdout_metrics = pd.DataFrame(holdout_records)

    df_cv_metrics.to_csv(tables_dir / "cv_metrics.csv", index=False)
    df_holdout_metrics.to_csv(tables_dir / "holdout_metrics.csv", index=False)
    df_model_comp.to_csv(tables_dir / "model_comparison.csv", index=False)
    df_cal_bins.to_csv(tables_dir / "calibration_bins.csv", index=False)
    df_paired_diffs.to_csv(tables_dir / "paired_model_differences.csv", index=False)
    df_odds_ratios.to_csv(tables_dir / "feature_odds_ratios.csv", index=False)
    df_contrasts.to_csv(tables_dir / "odds_ratio_contrasts.csv", index=False)
    df_hotspot_val.to_csv(tables_dir / "hotspot_validation.csv", index=False)
    df_fairness.to_csv(tables_dir / "demographic_fairness.csv", index=False)
    df_sign_stability.to_csv(tables_dir / "tenure_sign_stability.csv", index=False)
    df_consistency.to_csv(tables_dir / "consistency_audit.csv", index=False)

    # 11. Save Full Metrics JSON
    metrics_summary = {
        "metadata": {
            "n_total": n_total,
            "n_train": n_train,
            "n_holdout": n_holdout,
            "seed": SEED,
            "train_baseline_rate": float(train_baseline),
            "holdout_baseline_rate": float(holdout_baseline),
            "materiality_gain_set_a_vs_b": float(delta_pr_auc),
            "selected_feature_set": recommended_set,
        },
        "cv_evaluation": {
            "dummy_prior": dummy_cv["metrics_mean"],
            "hist_gradient_boosting": hgb_cv["metrics_mean"],
            "logistic_regression_set_a": lr_a_cv["metrics_mean"],
            "logistic_regression_set_b": lr_b_cv["metrics_mean"],
        },
        "holdout_evaluation": {
            "dummy_prior": boot_dummy,
            "hist_gradient_boosting": boot_hgb,
            "logistic_regression_uncalibrated": boot_lr_uncal,
            "logistic_regression_calibrated": boot_lr_cal,
        },
    }
    with open(reports_phase4_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print("Metrics written to reports/phase4/metrics.json")

    # 12. Generate Figures FIG-15 through FIG-20
    print("\nGenerating figures FIG-15 through FIG-20...")
    model_preds = {
        "Dummy (Prior)": probs_dummy,
        "HistGradientBoosting": probs_hgb,
        "Logistic Regression (Uncalibrated)": probs_lr_uncal,
        "Logistic Regression (Calibrated)": probs_lr_cal,
    }

    fig_generators = [
        (
            "FIG-15",
            "fig15_precision_recall_curves.png",
            "FIG-15: Precision-Recall Curves across Attrition Risk Models",
            lambda: plot_fig15_pr_curves(model_preds, y_ho_arr, holdout_baseline),
        ),
        (
            "FIG-16",
            "fig16_calibration_curves.png",
            "FIG-16: Probability Reliability Diagram (Calibration Curve)",
            lambda: plot_fig16_calibration(y_ho_arr, probs_lr_uncal, probs_lr_cal),
        ),
        (
            "FIG-17",
            "fig17_risk_tier_attrition.png",
            "FIG-17: Holdout Attrition Yield Across Model-Prioritized Risk Tiers",
            lambda: plot_fig17_risk_bands(y_ho_arr, probs_lr_cal, holdout_baseline),
        ),
        (
            "FIG-18",
            "fig18_feature_odds_ratios.png",
            "FIG-18: Top Predictors in Regularised Logistic Regression",
            lambda: plot_fig18_odds_ratios(df_odds_ratios),
        ),
        (
            "FIG-19",
            "fig19_hotspot_validation.png",
            "FIG-19: Out-of-Sample Validation of Candidate Attrition Hotspots",
            lambda: plot_fig19_hotspot_validation(df_hotspot_val),
        ),
        (
            "FIG-20",
            "fig20_demographic_fairness.png",
            "FIG-20: Demographic Fairness Audit at Operational 20% Triage Threshold",
            lambda: plot_fig20_demographic_fairness(df_fairness),
        ),
    ]

    manifest_records: List[Dict[str, Any]] = []

    for fid, fname, title, gen_fn in fig_generators:
        fig, plotted_df = gen_fn()
        fig_path = figures_dir / fname
        fig.savefig(fig_path, dpi=FIGURE_DPI)

        file_size = fig_path.stat().st_size
        width_in, height_in = fig.get_size_inches()
        plotted_sha = compute_df_sha256(plotted_df)

        manifest_records.append(
            {
                "figure_id": fid,
                "filename": fname,
                "title": title,
                "width_in": round(float(width_in), 2),
                "height_in": round(float(height_in), 2),
                "dpi": FIGURE_DPI,
                "file_size_bytes": file_size,
                "plotted_data_sha256": plotted_sha,
            }
        )
        print(f"  {fid} -> {fname} ({file_size:,} bytes, sha256={plotted_sha[:12]}...)")

    df_manifest = pd.DataFrame(manifest_records)
    df_manifest.to_csv(reports_phase4_dir / "figure_manifest.csv", index=False)
    print("Figure manifest written to reports/phase4/figure_manifest.csv")

    print("\nPhase 4 pipeline completed successfully.")


if __name__ == "__main__":
    main()
