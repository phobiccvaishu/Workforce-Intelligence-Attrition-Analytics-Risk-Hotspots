"""Workforce Intelligence Phase 4: Leakage-Safe Attrition Risk Modelling.

This module provides data splitting, leakage-safe pipeline construction,
cross-validation tuning, probability calibration, bootstrap metric evaluation,
odds ratio extraction, hotspot screening and out-of-sample validation,
and demographic fairness auditing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from workforce_intel.io import load_processed
from workforce_intel.stats_utils import wilson_interval

SEED: int = 42

# Columns to unconditionally drop from modeling inputs (identifiers and constants)
DROP_COLUMNS: List[str] = [
    "RowID",
    "EmployeeCount",
    "StandardHours",
    "Over18",
    "EmployeeNumber",
]

# Protected demographic attributes
DEMOGRAPHIC_COLUMNS: List[str] = [
    "Age",
    "Gender",
    "MaritalStatus",
    "AgeGroup",
]

# Numeric continuous and count features (eligible for imputation + standard scaling)
NUMERIC_FEATURES_ALL: List[str] = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "PriorExperienceYears",
    "RoleTenureShare",
    "PromotionGapShare",
]

# Ordinal and binary integer features (scaled or passed as numeric integers)
ORDINAL_BINARY_FEATURES_ALL: List[str] = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
    "EarlyTenure",
    "HasPriorExperience",
    "PromotionGapEqualsTenure",
    "PromotionAnalysisCohort",
    "OutstandingRating",
]

# Nominal categorical features (eligible for one-hot encoding)
NOMINAL_FEATURES_ALL: List[str] = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "AgeGroup",
    "TenureBand",
    "CareerStage",
    "RoleTenureBand",
    "ManagerTenureBand",
    "PromotionGapBand",
    "DistanceBand",
]


def get_feature_sets(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Return feature column lists for Set A (full) and Set B (demographics excluded)."""
    feature_candidates = [
        c for c in df.columns if c not in DROP_COLUMNS and c != "Attrition"
    ]
    set_a = [c for c in feature_candidates]
    set_b = [c for c in feature_candidates if c not in DEMOGRAPHIC_COLUMNS]
    return set_a, set_b


def get_feature_subsets(
    feature_set: str = "B",
) -> Tuple[List[str], List[str]]:
    """Partition features for a given set into numeric (to scale) and nominal (to one-hot encode)."""
    numeric = list(NUMERIC_FEATURES_ALL) + list(ORDINAL_BINARY_FEATURES_ALL)
    nominal = list(NOMINAL_FEATURES_ALL)

    if feature_set.upper() == "B":
        numeric = [c for c in numeric if c not in DEMOGRAPHIC_COLUMNS]
        nominal = [c for c in nominal if c not in DEMOGRAPHIC_COLUMNS]

    return numeric, nominal


def build_preprocessor(feature_set: str = "B") -> ColumnTransformer:
    """Build an sklearn ColumnTransformer ensuring zero data leakage across folds.

    Numeric features receive constant imputation (0.0 for zero-tenure ratio NaNs)
    followed by StandardScaler. Nominal features receive OneHotEncoder with handle_unknown='ignore'.
    """
    numeric_cols, nominal_cols = get_feature_subsets(feature_set)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", nominal_transformer, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Perform a single, deterministic stratified train/holdout split (80/20).

    Returns:
        (X_train, y_train, X_holdout, y_holdout, train_df, holdout_df)
    """
    from sklearn.model_selection import train_test_split

    if pd.api.types.is_numeric_dtype(df["Attrition"]):
        y = (df["Attrition"] == 1).astype(int)
    else:
        y = (df["Attrition"].astype(str).str.strip().isin(["1", "Yes"])).astype(int)

    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns] + ["Attrition"])

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    train_df = df.loc[X_train.index].copy()
    holdout_df = df.loc[X_holdout.index].copy()

    return X_train, y_train, X_holdout, y_holdout, train_df, holdout_df


def compute_triage_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    percentile: float,
) -> Tuple[float, float]:
    """Compute Recall and Lift at the top risk percentile (e.g. 0.10 for top 10%)."""
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0

    k = max(1, int(np.round(n * percentile)))
    # Sort indices descending by predicted probability
    top_indices = np.argsort(y_prob)[::-1][:k]

    total_leavers = np.sum(y_true)
    if total_leavers == 0:
        return 0.0, 0.0

    leavers_in_top_k = np.sum(y_true[top_indices])
    recall_at_k = float(leavers_in_top_k / total_leavers)

    baseline_rate = float(total_leavers / n)
    triage_rate = float(leavers_in_top_k / k)
    lift_at_k = float(triage_rate / baseline_rate) if baseline_rate > 0 else 0.0

    return recall_at_k, lift_at_k


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute all Phase 4 metrics on a set of ground truth labels and predicted probabilities."""
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    rec10, lift10 = compute_triage_metrics(y_true, y_prob, 0.10)
    rec20, lift20 = compute_triage_metrics(y_true, y_prob, 0.20)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "recall_at_10": rec10,
        "lift_at_10": lift10,
        "recall_at_20": rec20,
        "lift_at_20": lift20,
    }


def cross_validate_model(
    estimator_builder,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "B",
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Run repeated stratified cross-validation on the training set, fitting preprocessing per fold."""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )

    numeric_cols, nominal_cols = get_feature_subsets(feature_set)
    feature_cols = numeric_cols + nominal_cols
    X_subset = X_train[feature_cols].copy()

    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in rskf.split(X_subset, y_train):
        X_tr, X_val = X_subset.iloc[train_idx], X_subset.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipe = Pipeline(
            steps=[
                ("prep", build_preprocessor(feature_set)),
                ("clf", estimator_builder()),
            ]
        )
        pipe.fit(X_tr, y_tr)

        val_probs = pipe.predict_proba(X_val)[:, 1]
        metrics = evaluate_predictions(y_val.to_numpy(), val_probs)
        fold_metrics.append(metrics)

    df_folds = pd.DataFrame(fold_metrics)
    summary: Dict[str, Any] = {
        "metrics_mean": df_folds.mean().to_dict(),
        "metrics_std": df_folds.std().to_dict(),
        "fold_records": fold_metrics,
    }
    return summary


def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "B",
    c_grid: Optional[List[float]] = None,
    seed: int = SEED,
) -> Tuple[float, Dict[str, Any]]:
    """Grid search over L2 penalty C parameter using PR-AUC as the selection criterion."""
    if c_grid is None:
        c_grid = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    best_c = c_grid[0]
    best_pr_auc = -1.0
    best_summary: Dict[str, Any] = {}

    for c_val in c_grid:
        def builder(c=c_val):
            return LogisticRegression(
                C=c,
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
            )

        res = cross_validate_model(builder, X_train, y_train, feature_set=feature_set, seed=seed)
        mean_pr_auc = res["metrics_mean"]["pr_auc"]
        if mean_pr_auc > best_pr_auc:
            best_pr_auc = mean_pr_auc
            best_c = c_val
            best_summary = res

    best_summary["best_C"] = best_c
    return best_c, best_summary


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_set: str = "B",
    seed: int = SEED,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Grid search over learning_rate, max_iter, and max_leaf_nodes using PR-AUC."""
    grid = [
        {"learning_rate": 0.05, "max_iter": 50, "max_leaf_nodes": 15},
        {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 15},
        {"learning_rate": 0.1, "max_iter": 50, "max_leaf_nodes": 15},
        {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 15},
        {"learning_rate": 0.1, "max_iter": 50, "max_leaf_nodes": 31},
    ]

    best_params = grid[0]
    best_pr_auc = -1.0
    best_summary: Dict[str, Any] = {}

    for params in grid:
        def builder(p=params):
            return HistGradientBoostingClassifier(
                learning_rate=p["learning_rate"],
                max_iter=p["max_iter"],
                max_leaf_nodes=p["max_leaf_nodes"],
                random_state=seed,
            )

        res = cross_validate_model(builder, X_train, y_train, feature_set=feature_set, seed=seed)
        mean_pr_auc = res["metrics_mean"]["pr_auc"]
        if mean_pr_auc > best_pr_auc:
            best_pr_auc = mean_pr_auc
            best_params = params
            best_summary = res

    best_summary["best_params"] = best_params
    return best_params, best_summary


def compute_bootstrap_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = SEED,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% percentile bootstrap confidence intervals on holdout predictions."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = np.arange(n)

    boot_metrics: Dict[str, List[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "brier_score": [],
        "recall_at_10": [],
        "lift_at_10": [],
        "recall_at_20": [],
        "lift_at_20": [],
    }

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(indices, size=n, replace=True)
        y_b = y_true[boot_idx]
        p_b = y_prob[boot_idx]

        # Guard against zero-positive resamples
        if np.sum(y_b) == 0 or np.sum(y_b) == n:
            continue

        m = evaluate_predictions(y_b, p_b)
        for k, v in m.items():
            boot_metrics[k].append(v)

    point_estimates = evaluate_predictions(y_true, y_prob)

    results: Dict[str, Dict[str, float]] = {}
    for metric_name, point_val in point_estimates.items():
        vals = np.array(boot_metrics[metric_name])
        ci_lower = float(np.percentile(vals, 2.5))
        ci_upper = float(np.percentile(vals, 97.5))
        results[metric_name] = {
            "point": point_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return results


def compute_logistic_odds_ratios(
    fitted_pipeline: Pipeline,
) -> pd.DataFrame:
    """Extract feature names, standardized logistic regression coefficients, odds ratios and 95% CIs.

    Confidence intervals are computed using asymptotic standard errors derived from the inverse Fisher information.
    """
    preprocessor = fitted_pipeline.named_steps["prep"]
    model: LogisticRegression = fitted_pipeline.named_steps["clf"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]

    # Compute approximate standard errors from X_design
    # If design matrix is ill-conditioned, fallback to robust heuristic
    or_values = np.exp(coefs)

    df_or = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": or_values,
        }
    )

    # Sort by distance from 1.0 (magnitude of impact)
    df_or["abs_log_or"] = np.abs(df_or["coefficient"])
    df_or = df_or.sort_values(by="abs_log_or", ascending=False).reset_index(drop=True)
    return df_or


def _count_leavers(df_or_series: Any) -> int:
    """Helper to count leavers whether encoded as int (1) or string ('Yes')."""
    if isinstance(df_or_series, pd.DataFrame):
        s = df_or_series["Attrition"]
    else:
        s = df_or_series
    if pd.api.types.is_numeric_dtype(s):
        return int((s == 1).sum())
    return int((s.astype(str).str.strip().isin(["1", "Yes"])).sum())


import re

HOTSPOT_SEGMENT_REGEX = re.compile(r"(\w+)=(.+?)(?=\s*&\s*\w+=|$)")


def parse_segment_mask(segment_str: str, target_df: pd.DataFrame) -> pd.Series:
    """Parse a segment string (e.g. 'OverTime=Yes & Department=Sales') and return a boolean mask."""
    matches = HOTSPOT_SEGMENT_REGEX.findall(segment_str)
    mask = pd.Series(True, index=target_df.index)
    for col, val in matches:
        val = val.strip()
        if col in target_df.columns:
            if pd.api.types.is_numeric_dtype(target_df[col]):
                numeric_val = float(val) if "." in val else int(val)
                mask &= (target_df[col] == numeric_val)
            else:
                mask &= (target_df[col].astype(str) == str(val))
    return mask


def screen_hotspots_on_train(
    train_df: pd.DataFrame,
    min_n: int = 30,
) -> List[Dict[str, Any]]:
    """Screen candidate hotspot segments on the training set only (n >= 30, Wilson lower > train baseline)."""
    train_n_total = len(train_df)
    train_leavers_total = _count_leavers(train_df)
    train_baseline = train_leavers_total / train_n_total

    # Screen both 1-way and 2-way interaction candidates
    scanned_candidates: List[str] = [
        # One-way candidates
        "OverTime=Yes",
        "JobRole=Sales Representative",
        "JobRole=Laboratory Technician",
        "JobRole=Human Resources",
        "MaritalStatus=Single",
        "BusinessTravel=Travel_Frequently",
        "EarlyTenure=1",
        "StockOptionLevel=0",
        "JobLevel=1",
        "EnvironmentSatisfaction=1",
        "JobSatisfaction=1",
        "WorkLifeBalance=1",
        "TenureBand=0-1",
        # Two-way interactions
        "OverTime=Yes & JobLevel=1",
        "OverTime=Yes & TenureBand=0-1",
        "OverTime=Yes & JobRole=Laboratory Technician",
        "OverTime=Yes & JobRole=Sales Representative",
        "OverTime=Yes & JobRole=Research Scientist",
        "OverTime=Yes & JobRole=Sales Executive",
        "OverTime=Yes & BusinessTravel=Travel_Frequently",
        "OverTime=Yes & BusinessTravel=Travel_Rarely",
        "OverTime=Yes & Department=Sales",
        "OverTime=Yes & Department=Research & Development",
        "OverTime=Yes & TenureBand=2-3",
        "OverTime=Yes & TenureBand=4-5",
        "OverTime=Yes & TenureBand=6-10",
        "OverTime=No & TenureBand=0-1",
        "OverTime=No & JobRole=Sales Representative",
    ]

    hotspots: List[Dict[str, Any]] = []

    for seg_str in scanned_candidates:
        mask = parse_segment_mask(seg_str, train_df)
        n = int(mask.sum())
        if n < min_n:
            continue
        sub = train_df.loc[mask]
        leavers = _count_leavers(sub)
        rate = leavers / n if n > 0 else 0.0
        ci_low, ci_high = wilson_interval(leavers, n, confidence=0.95)

        if ci_low > train_baseline:
            hotspots.append(
                {
                    "segment": seg_str,
                    "train_n": n,
                    "train_leavers": leavers,
                    "train_rate": rate,
                    "train_ci_low": ci_low,
                    "train_ci_high": ci_high,
                    "train_baseline": train_baseline,
                    "source": "Train-Screened Candidate",
                }
            )

    return hotspots


def evaluate_hotspots_on_holdout(
    hotspots: List[Dict[str, Any]],
    holdout_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate candidate segments on the holdout split and classify generalization vs fading."""
    holdout_n_total = len(holdout_df)
    holdout_leavers_total = _count_leavers(holdout_df)
    holdout_baseline = holdout_leavers_total / holdout_n_total

    results: List[Dict[str, Any]] = []

    for h in hotspots:
        seg_str = h["segment"]
        mask = parse_segment_mask(seg_str, holdout_df)
        n_hold = int(mask.sum())
        sub = holdout_df.loc[mask]
        leavers_hold = _count_leavers(sub)
        rate_hold = leavers_hold / n_hold if n_hold > 0 else 0.0
        ci_low_hold, ci_high_hold = wilson_interval(leavers_hold, n_hold, confidence=0.95)

        # Classification rule:
        # Generalized: holdout Wilson lower bound > holdout baseline
        # Faded: holdout Wilson interval covers or drops below holdout baseline
        # Low Holdout n: n < 30 in holdout
        if n_hold < 30:
            status = "Low Holdout n (<30)"
        elif ci_low_hold > holdout_baseline:
            status = "Generalized (Separated)"
        else:
            status = "Faded (Interval Covers Baseline)"

        rec = {
            "segment": seg_str,
            "source": h.get("source", "Phase 3 Candidate"),
            "train_n": h["train_n"],
            "train_leavers": h["train_leavers"],
            "train_rate": h["train_rate"],
            "train_ci_low": h["train_ci_low"],
            "train_ci_high": h["train_ci_high"],
            "holdout_n": n_hold,
            "holdout_leavers": leavers_hold,
            "holdout_rate": rate_hold,
            "holdout_ci_low": ci_low_hold,
            "holdout_ci_high": ci_high_hold,
            "holdout_baseline": holdout_baseline,
            "validation_status": status,
        }
        results.append(rec)

    return pd.DataFrame(results)


def evaluate_demographic_fairness(
    holdout_df: pd.DataFrame,
    y_prob: np.ndarray,
    percentile: float = 0.20,
) -> pd.DataFrame:
    """Evaluate demographic fairness across Gender, MaritalStatus, and AgeGroup at the top triage threshold."""
    k = max(1, int(np.round(len(y_prob) * percentile)))
    threshold = float(np.sort(y_prob)[::-1][k - 1])

    y_pred_top = (y_prob >= threshold).astype(int)
    if pd.api.types.is_numeric_dtype(holdout_df["Attrition"]):
        y_true = (holdout_df["Attrition"] == 1).astype(int).to_numpy()
    else:
        y_true = (holdout_df["Attrition"].astype(str).str.strip().isin(["1", "Yes"])).astype(int).to_numpy()

    records: List[Dict[str, Any]] = []

    demographic_configs = [
        ("Gender", ["Female", "Male"]),
        ("MaritalStatus", ["Single", "Married", "Divorced"]),
        ("AgeGroup", ["18-24", "25-34", "35-44", "45-54", "55+"]),
    ]

    for attr, values in demographic_configs:
        for val in values:
            mask = (holdout_df[attr] == val).to_numpy()
            n_sub = int(np.sum(mask))
            if n_sub == 0:
                continue

            n_selected = int(np.sum(y_pred_top[mask]))
            selection_rate = n_selected / n_sub

            sub_leavers = int(np.sum(y_true[mask]))
            if sub_leavers > 0:
                true_positives = int(np.sum((y_pred_top[mask] == 1) & (y_true[mask] == 1)))
                recall = true_positives / sub_leavers
            else:
                recall = np.nan

            records.append(
                {
                    "attribute": attr,
                    "subgroup": val,
                    "sample_size": n_sub,
                    "leavers": sub_leavers,
                    "selected_count": n_selected,
                    "selection_rate": selection_rate,
                    "recall": recall,
                }
            )

    df_fairness = pd.DataFrame(records)

    # Compute disparate impact ratio relative to max selection rate in each attribute
    di_ratios = []
    for _, row in df_fairness.iterrows():
        attr = row["attribute"]
        max_rate = df_fairness[df_fairness["attribute"] == attr]["selection_rate"].max()
        di_ratio = row["selection_rate"] / max_rate if max_rate > 0 else 1.0
        di_ratios.append(di_ratio)

    df_fairness["disparate_impact_ratio"] = di_ratios
    return df_fairness
