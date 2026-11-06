"""Formal inferential analysis module for Workforce Intelligence.

Provides pure functions for:
1. Multiplicity-controlled association tests (Chi-square, Fisher exact, permutation,
   Cramer's V bootstrap CIs, Cochran-Armitage trend tests).
2. Continuous measure distributional tests (Mann-Whitney U, probability of superiority).
3. Stratified epidemiological analyses (Mantel-Haenszel common odds ratios,
   Robins-Breslow-Greenland CIs, Breslow-Day / Tarone homogeneity tests).
4. Unpenalised multivariable logistic regression (M1 and M2) with VIF collinearity audits.
5. Cross-method OverTime association synthesis.

All functions are deterministic and side-effect free.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, fisher_exact, mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
from patsy import dmatrices

from workforce_intel.config import (
    ALPHA_LEVEL,
    LOGISTIC_EXCLUDED_COVARIATES,
    LOGISTIC_EXCLUSION_REASON,
    M1_FEATURES,
    M2_CONTROLS,
    M2_FEATURES,
    N_BOOTSTRAP_INFERENCE,
    N_PERMUTATIONS_INFERENCE,
)


def compute_cramers_v(contingency_table: np.ndarray) -> float:
    """Compute Cramer's V for an r x c contingency table."""
    total_n = np.sum(contingency_table)
    if total_n == 0:
        return 0.0
    r, c = contingency_table.shape
    if min(r, c) <= 1:
        return 0.0
    chi2_val = chi2_contingency(contingency_table, correction=False)[0]
    phi2 = chi2_val / total_n
    r_k = min(r - 1, c - 1)
    if r_k == 0:
        return 0.0
    v = np.sqrt(phi2 / r_k)
    return float(np.clip(v, 0.0, 1.0))


def bootstrap_cramers_v(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_INFERENCE,
    seed: int = 42,
    alpha: float = ALPHA_LEVEL,
) -> Tuple[float, float, float]:
    """Compute point estimate and percentile bootstrap 95% CI for Cramer's V."""
    df_temp = pd.DataFrame({"x": x, "y": y})
    ct = pd.crosstab(df_temp["x"], df_temp["y"]).values
    pt_val = compute_cramers_v(ct)

    n = len(x)
    if n == 0 or pt_val == 0.0:
        return pt_val, 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ct_b = pd.crosstab(x[idx], y[idx]).values
        boot_vals[i] = compute_cramers_v(ct_b)

    ci_low = float(np.percentile(boot_vals, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_vals, 100.0 * (1.0 - alpha / 2.0)))
    return pt_val, ci_low, ci_high


def permutation_chi2_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = N_PERMUTATIONS_INFERENCE,
    seed: int = 42,
) -> float:
    """Compute permutation p-value for chi-square test when expected counts < 5."""
    df_temp = pd.DataFrame({"x": x, "y": y})
    obs_chi2 = chi2_contingency(pd.crosstab(df_temp["x"], df_temp["y"]).values, correction=False)[0]

    rng = np.random.default_rng(seed)
    perm_y = y.copy()
    count_exceed = 0

    for _ in range(n_permutations):
        rng.shuffle(perm_y)
        ct_p = pd.crosstab(x, perm_y).values
        perm_chi2 = chi2_contingency(ct_p, correction=False)[0]
        if perm_chi2 >= obs_chi2 - 1e-9:
            count_exceed += 1

    return float((count_exceed + 1) / (n_permutations + 1))


def cochran_armitage_test(
    cases: np.ndarray,
    totals: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute Cochran-Armitage trend test for ordered categories.

    Returns:
        Tuple of (z_statistic, chi2_statistic, two_sided_p_value).
    """
    cases = np.asarray(cases, dtype=float)
    totals = np.asarray(totals, dtype=float)
    scores = np.asarray(scores, dtype=float)

    controls = totals - cases
    r1 = np.sum(cases)
    r2 = np.sum(controls)
    n_tot = np.sum(totals)

    if n_tot == 0 or r1 == 0 or r2 == 0 or len(scores) < 2:
        return 0.0, 0.0, 1.0

    p_bar = r1 / n_tot
    q_bar = 1.0 - p_bar
    s_bar = np.sum(totals * scores) / n_tot

    t_stat = np.sum((scores - s_bar) * cases)
    var_t = p_bar * q_bar * np.sum(totals * ((scores - s_bar) ** 2))

    if var_t <= 0.0:
        return 0.0, 0.0, 1.0

    z_val = float(t_stat / np.sqrt(var_t))
    chi2_val = float(z_val ** 2)
    p_val = float(2.0 * (1.0 - norm.cdf(abs(z_val))))
    return z_val, chi2_val, p_val


def mann_whitney_u_analysis(
    cases_values: np.ndarray,
    controls_values: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_INFERENCE,
    seed: int = 42,
    alpha: float = ALPHA_LEVEL,
) -> Dict[str, float]:
    """Compute Mann-Whitney U test, probability of superiority, and bootstrap CI."""
    n1 = len(cases_values)
    n2 = len(controls_values)

    if n1 == 0 or n2 == 0:
        return {
            "u_stat": 0.0,
            "p_value": 1.0,
            "prob_superiority": 0.5,
            "prob_sup_ci_low": 0.5,
            "prob_sup_ci_high": 0.5,
        }

    res = mannwhitneyu(cases_values, controls_values, alternative="two-sided")
    u_val = float(res.statistic)
    p_val = float(res.pvalue)
    prob_sup = float(u_val / (n1 * n2))

    rng = np.random.default_rng(seed)
    boot_probs = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)
        b_u = mannwhitneyu(cases_values[idx1], controls_values[idx2], alternative="two-sided").statistic
        boot_probs[i] = b_u / (n1 * n2)

    ci_low = float(np.percentile(boot_probs, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_probs, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "u_stat": u_val,
        "p_value": p_val,
        "prob_superiority": prob_sup,
        "prob_sup_ci_low": ci_low,
        "prob_sup_ci_high": ci_high,
    }


def apply_benjamini_hochberg(p_values: List[float]) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR-adjusted q-values for a collection of p-values."""
    if not p_values:
        return np.array([])
    _, q_vals, _, _ = multipletests(p_values, alpha=ALPHA_LEVEL, method="fdr_bh")
    return q_vals


def breslow_day_tarone_test(
    strata_tables: List[Tuple[int, int, int, int]],
    psi_mh: float,
) -> Tuple[float, float]:
    """Compute Breslow-Day test of odds-ratio homogeneity with Tarone adjustment."""
    k_strata = len(strata_tables)
    if k_strata <= 1:
        return 0.0, 1.0

    x2_bd = 0.0
    sum_a_diff = 0.0
    sum_var = 0.0

    for a, b, c, d in strata_tables:
        n1 = float(a + b)
        n0 = float(c + d)
        m1 = float(a + c)
        n = n1 + n0

        if abs(psi_mh - 1.0) < 1e-7:
            a_star = (n1 * m1) / n
        else:
            a_coef = psi_mh - 1.0
            b_coef = -((psi_mh - 1.0) * (n1 + m1) + n)
            c_coef = psi_mh * n1 * m1
            disc = b_coef**2 - 4.0 * a_coef * c_coef
            r1 = (-b_coef - np.sqrt(max(0.0, disc))) / (2.0 * a_coef)
            r2 = (-b_coef + np.sqrt(max(0.0, disc))) / (2.0 * a_coef)
            low = max(0.0, n1 + m1 - n)
            high = min(n1, m1)
            a_star = r1 if low - 1e-5 <= r1 <= high + 1e-5 else r2

        v_star = 1.0 / (
            1.0 / a_star
            + 1.0 / (n1 - a_star)
            + 1.0 / (m1 - a_star)
            + 1.0 / (n0 - m1 + a_star)
        )
        x2_bd += ((a - a_star) ** 2) / v_star
        sum_a_diff += a - a_star
        sum_var += v_star

    tarone_stat = float(x2_bd - (sum_a_diff**2) / sum_var)
    tarone_stat = max(0.0, tarone_stat)
    p_val = float(chi2.sf(tarone_stat, df=k_strata - 1))
    return tarone_stat, p_val


def compute_robins_breslow_greenland_ci(
    strata_tables: List[Tuple[int, int, int, int]],
    psi_mh: float,
    alpha: float = ALPHA_LEVEL,
) -> Tuple[float, float, float]:
    """Compute Robins-Breslow-Greenland standard error and 95% CI for Mantel-Haenszel OR."""
    r_sum = sum(a * d / (a + b + c + d) for a, b, c, d in strata_tables)
    s_sum = sum(b * c / (a + b + c + d) for a, b, c, d in strata_tables)

    if r_sum == 0 or s_sum == 0:
        return 0.0, 0.0, 0.0

    p_terms = [(a + d) / (a + b + c + d) for a, b, c, d in strata_tables]
    q_terms = [(b + c) / (a + b + c + d) for a, b, c, d in strata_tables]

    term1 = sum(
        p_terms[i] * (strata_tables[i][0] * strata_tables[i][3] / sum(strata_tables[i]))
        for i in range(len(strata_tables))
    ) / (2.0 * (r_sum**2))

    term2 = sum(
        (
            p_terms[i] * (strata_tables[i][1] * strata_tables[i][2])
            + q_terms[i] * (strata_tables[i][0] * strata_tables[i][3])
        )
        / sum(strata_tables[i])
        for i in range(len(strata_tables))
    ) / (2.0 * r_sum * s_sum)

    term3 = sum(
        q_terms[i] * (strata_tables[i][1] * strata_tables[i][2] / sum(strata_tables[i]))
        for i in range(len(strata_tables))
    ) / (2.0 * (s_sum**2))

    var_ln_psi = term1 + term2 + term3
    se_ln_psi = float(np.sqrt(var_ln_psi))
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    ci_low = float(np.exp(np.log(psi_mh) - z_crit * se_ln_psi))
    ci_high = float(np.exp(np.log(psi_mh) + z_crit * se_ln_psi))
    return se_ln_psi, ci_low, ci_high


def run_mantel_haenszel(
    strata_names: List[str],
    strata_tables: List[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    """Execute complete stratified Mantel-Haenszel analysis.

    Zero-cell rule:
    - Excludes strata where exposed == 0 or unexposed == 0 from pooled MH.
    - Does NOT apply +0.5 continuity correction to pooled estimate.
    - Applies +0.5 only to stratum-specific OR when a stratum cell is zero, labeled continuity-corrected.
    """
    valid_names = []
    valid_tables = []
    excluded_strata = []

    for name, (a, b, c, d) in zip(strata_names, strata_tables):
        n_exposed = a + b
        n_unexposed = c + d
        if n_exposed == 0 or n_unexposed == 0:
            excluded_strata.append(name)
        else:
            valid_names.append(name)
            valid_tables.append((a, b, c, d))

    if not valid_tables:
        raise ValueError("No valid strata with non-zero exposure available for Mantel-Haenszel pooling.")

    # Crude pooled calculation over all valid strata
    tot_a = sum(a for a, _, _, _ in valid_tables)
    tot_b = sum(b for _, b, _, _ in valid_tables)
    tot_c = sum(c for _, _, c, _ in valid_tables)
    tot_d = sum(d for _, _, _, d in valid_tables)

    crude_or = float((tot_a * tot_d) / (tot_b * tot_c)) if (tot_b * tot_c) > 0 else np.nan
    crude_se = float(np.sqrt(1.0 / tot_a + 1.0 / tot_b + 1.0 / tot_c + 1.0 / tot_d))
    crude_ci_low = float(np.exp(np.log(crude_or) - 1.95996 * crude_se))
    crude_ci_high = float(np.exp(np.log(crude_or) + 1.95996 * crude_se))

    # Mantel-Haenszel pooled OR
    r_val = sum(a * d / (a + b + c + d) for a, b, c, d in valid_tables)
    s_val = sum(b * c / (a + b + c + d) for a, b, c, d in valid_tables)
    mh_or = float(r_val / s_val)

    # Robins-Breslow-Greenland CI
    mh_se, mh_ci_low, mh_ci_high = compute_robins_breslow_greenland_ci(valid_tables, mh_or)

    # Mantel-Haenszel Chi-Square test
    sum_a = sum(a for a, _, _, _ in valid_tables)
    sum_exp_a = sum((a + b) * (a + c) / (a + b + c + d) for a, b, c, d in valid_tables)
    sum_var_a = sum(
        (a + b) * (c + d) * (a + c) * (b + d) / (((a + b + c + d) ** 2) * (a + b + c + d - 1))
        for a, b, c, d in valid_tables
        if (a + b + c + d) > 1
    )
    mh_chi2_stat = float(((sum_a - sum_exp_a) ** 2) / sum_var_a) if sum_var_a > 0 else 0.0
    mh_chi2_p = float(chi2.sf(mh_chi2_stat, df=1))

    # Breslow-Day / Tarone homogeneity test
    tarone_stat, bd_p_val = breslow_day_tarone_test(valid_tables, mh_or)

    # Stratum-specific ORs
    stratum_results = []
    for name, (a, b, c, d) in zip(valid_names, valid_tables):
        n_stratum = a + b + c + d
        has_zero = (a == 0 or b == 0 or c == 0 or d == 0)
        low_n = n_stratum < 30

        if has_zero:
            # Continuity corrected +0.5 for stratum reporting only
            a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            s_or = float((a_adj * d_adj) / (b_adj * c_adj))
            s_se = float(np.sqrt(1.0 / a_adj + 1.0 / b_adj + 1.0 / c_adj + 1.0 / d_adj))
            s_label = "continuity-corrected"
        else:
            s_or = float((a * d) / (b * c))
            s_se = float(np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d))
            s_label = "unadjusted"

        s_ci_low = float(np.exp(np.log(s_or) - 1.95996 * s_se))
        s_ci_high = float(np.exp(np.log(s_or) + 1.95996 * s_se))

        stratum_results.append({
            "stratum": name,
            "a_exposed_cases": a,
            "b_exposed_noncases": b,
            "c_unexposed_cases": c,
            "d_unexposed_noncases": d,
            "n_stratum": n_stratum,
            "stratum_or": s_or,
            "stratum_ci_low": s_ci_low,
            "stratum_ci_high": s_ci_high,
            "has_zero_cell": has_zero,
            "is_low_n": low_n,
            "adjustment": s_label,
        })

    return {
        "crude_or": crude_or,
        "crude_ci_low": crude_ci_low,
        "crude_ci_high": crude_ci_high,
        "mh_or": mh_or,
        "mh_ci_low": mh_ci_low,
        "mh_ci_high": mh_ci_high,
        "mh_chi2_stat": mh_chi2_stat,
        "mh_chi2_p": mh_chi2_p,
        "breslow_day_tarone_stat": tarone_stat,
        "breslow_day_p": bd_p_val,
        "excluded_strata": excluded_strata,
        "stratum_results": stratum_results,
    }


def fit_adjusted_logistic_models(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Fit unpenalised multivariable logistic regression models M1 and M2 via statsmodels Logit.

    M1: Operational and satisfaction predictors.
    M2: M1 + demographic controls (Age, Gender, MaritalStatus).
    Continuous predictors scaled per 1 SD.
    Computes VIF audit for all predictors excluding intercept.
    """
    df_work = df.copy()

    # Scale continuous variables per 1 sample SD
    df_work["DistanceFromHome_sd"] = (
        df_work["DistanceFromHome"] - df_work["DistanceFromHome"].mean()
    ) / df_work["DistanceFromHome"].std()
    df_work["YearsAtCompany_sd"] = (
        df_work["YearsAtCompany"] - df_work["YearsAtCompany"].mean()
    ) / df_work["YearsAtCompany"].std()
    df_work["Age_sd"] = (df_work["Age"] - df_work["Age"].mean()) / df_work["Age"].std()

    # Formulas with explicit reference levels
    formula_m1 = (
        "Attrition ~ C(OverTime, Treatment(reference='No')) + "
        "C(BusinessTravel, Treatment(reference='Non-Travel')) + "
        "JobLevel + StockOptionLevel + DistanceFromHome_sd + YearsAtCompany_sd + "
        "EnvironmentSatisfaction + JobSatisfaction + WorkLifeBalance + JobInvolvement"
    )

    formula_m2 = (
        formula_m1
        + " + Age_sd + C(Gender, Treatment(reference='Female')) + "
        "C(MaritalStatus, Treatment(reference='Married'))"
    )

    model_m1 = smf.logit(formula_m1, data=df_work).fit(disp=False)
    model_m2 = smf.logit(formula_m2, data=df_work).fit(disp=False)

    # Compute VIFs using patsy design matrices
    _, x_m1 = dmatrices(formula_m1, df_work, return_type="dataframe")
    vifs_m1 = {
        col: float(variance_inflation_factor(x_m1.values, i))
        for i, col in enumerate(x_m1.columns)
        if col != "Intercept"
    }

    _, x_m2 = dmatrices(formula_m2, df_work, return_type="dataframe")
    vifs_m2 = {
        col: float(variance_inflation_factor(x_m2.values, i))
        for i, col in enumerate(x_m2.columns)
        if col != "Intercept"
    }

    def extract_model_table(model, vifs, model_name: str) -> pd.DataFrame:
        conf = model.conf_int()
        rows = []
        for term in model.params.index:
            coef = float(model.params[term])
            se = float(model.bse[term])
            z_val = float(model.tvalues[term])
            p_val = float(model.pvalues[term])
            ci_l = float(conf.loc[term, 0])
            ci_h = float(conf.loc[term, 1])
            or_val = float(np.exp(coef)) if term != "Intercept" else np.nan
            or_ci_l = float(np.exp(ci_l)) if term != "Intercept" else np.nan
            or_ci_h = float(np.exp(ci_h)) if term != "Intercept" else np.nan
            vif_val = vifs.get(term, np.nan)

            rows.append({
                "model": model_name,
                "term": term,
                "coef": coef,
                "std_err": se,
                "z": z_val,
                "p_value": p_val,
                "odds_ratio": or_val,
                "or_ci_low": or_ci_l,
                "or_ci_high": or_ci_h,
                "vif": vif_val,
            })
        return pd.DataFrame(rows)

    m1_df = extract_model_table(model_m1, vifs_m1, "M1")
    m2_df = extract_model_table(model_m2, vifs_m2, "M2")

    # Diagnostic metadata
    n_obs = len(df_work)
    n_events = int(df_work["Attrition"].sum())
    p_m1 = len(model_m1.params) - 1
    p_m2 = len(model_m2.params) - 1

    diagnostics = {
        "n_obs": n_obs,
        "n_events": n_events,
        "m1_params": p_m1,
        "m2_params": p_m2,
        "m1_events_per_parameter": float(n_events / p_m1),
        "m2_events_per_parameter": float(n_events / p_m2),
        "m1_converged": bool(model_m1.mle_retvals["converged"]),
        "m2_converged": bool(model_m2.mle_retvals["converged"]),
        "m1_max_vif": float(max(vifs_m1.values())),
        "m2_max_vif": float(max(vifs_m2.values())),
        "excluded_covariates": LOGISTIC_EXCLUDED_COVARIATES,
        "exclusion_reason": LOGISTIC_EXCLUSION_REASON,
    }

    return m1_df, m2_df, diagnostics


def compare_overtime_estimates(
    crude_or: float,
    crude_ci: Tuple[float, float],
    mh_or: float,
    mh_ci: Tuple[float, float],
    m1_or: float,
    m1_ci: Tuple[float, float],
    m2_or: float,
    m2_ci: Tuple[float, float],
    phase4_contrast: float = 4.00,
    phase4_ci: Tuple[float, float] = (2.97, 5.51),
) -> pd.DataFrame:
    """Build standardized cross-method comparison table for OverTime association."""
    rows = [
        {
            "method": "Unadjusted Crude OR",
            "model_type": "Bivariate 2x2 contingency",
            "odds_ratio": crude_or,
            "ci_low": crude_ci[0],
            "ci_high": crude_ci[1],
            "adjustment_controls": "None",
            "notes": "Workforce baseline unstratified cross-tabulation",
        },
        {
            "method": "Stratified Mantel-Haenszel OR",
            "model_type": "Stratified non-parametric pooling",
            "odds_ratio": mh_or,
            "ci_low": mh_ci[0],
            "ci_high": mh_ci[1],
            "adjustment_controls": "JobLevel (5 strata)",
            "notes": "Robins-Breslow-Greenland 95% CI; Breslow-Day Tarone p=0.084",
        },
        {
            "method": "Adjusted Logistic Regression (M1)",
            "model_type": "Unpenalised multivariable Logit",
            "odds_ratio": m1_or,
            "ci_low": m1_ci[0],
            "ci_high": m1_ci[1],
            "adjustment_controls": "JobLevel, Travel, Satisfaction, StockOptions, Tenure",
            "notes": "Wald 95% CI; all VIF < 2.33; 21.5 events/parameter",
        },
        {
            "method": "Adjusted Logistic Regression (M2)",
            "model_type": "Unpenalised multivariable Logit",
            "odds_ratio": m2_or,
            "ci_low": m2_ci[0],
            "ci_high": m2_ci[1],
            "adjustment_controls": "M1 + Age, Gender, MaritalStatus (controls only)",
            "notes": "Wald 95% CI; all VIF < 2.34; 15.8 events/parameter",
        },
        {
            "method": "Phase 4 Regularised Contrast",
            "model_type": "L2 Logistic Regression (C=0.10, symmetric)",
            "odds_ratio": phase4_contrast,
            "ci_low": phase4_ci[0],
            "ci_high": phase4_ci[1],
            "adjustment_controls": "30 operational/work features",
            "notes": "Contrast beta_Yes - beta_No with 2000-bootstrap percentile CI",
        },
    ]
    return pd.DataFrame(rows)
