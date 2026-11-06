"""Generate and execute notebooks/04_modelling.ipynb to reports/phase4/04_modelling.executed.ipynb."""

import json
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def build_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0",
        },
    }

    cells = []

    # Title Markdown
    cells.append(
        nbformat.v4.new_markdown_cell(
            "# Workforce Intelligence: Phase 4 Attrition Risk Modelling & Out-of-Sample Validation\n\n"
            "This presentation notebook details the leakage-safe development, probability calibration, "
            "and out-of-sample holdout validation of attrition risk models developed on the IBM HR analytics dataset.\n\n"
            "> [!IMPORTANT]\n"
            "> **Decision Support Only:** The models developed here provide ranking and triage support for human HR professionals "
            "> and line managers. **NO automated adverse action** is permitted."
        )
    )

    # Imports & Manifest Display
    cells.append(
        nbformat.v4.new_code_cell(
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "manifest = pd.read_csv('reports/phase4/figure_manifest.csv')\n"
            "display(Markdown('### Phase 4 Figure Manifest'))\n"
            "manifest[['figure_id', 'filename', 'title', 'plotted_data_sha256']]"
        )
    )

    # Cross-Validation & Materiality Markdown
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 1. Cross-Validation Model Development & Feature Set Governance\n\n"
            "- **Training Split**: $n = 1,176$ (190 departures, 16.16% baseline).\n"
            "- **Evaluation Protocol**: Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 folds).\n"
            "- **Feature Set Governance**: Set B (demographics excluded) was selected because Set A (demographics included) "
            "did not exceed the pre-specified materiality threshold (+0.0300 PR-AUC gain)."
        )
    )

    # CV Metrics Code
    cells.append(
        nbformat.v4.new_code_cell(
            "cv_metrics = pd.read_csv('reports/phase4/tables/cv_metrics.csv')\n"
            "display(cv_metrics[['model', 'feature_set', 'cv_pr_auc_mean', 'cv_pr_auc_std', 'cv_roc_auc_mean', 'cv_brier_score_mean']])"
        )
    )

    # Holdout Metrics Markdown
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 2. Out-of-Sample Holdout Performance ($n = 294$, 47 Departures)\n\n"
            "The holdout set was locked and scored exactly once at the conclusion of model development. "
            "Uncertainty intervals are 2,000-sample percentile bootstrap 95% confidence intervals."
        )
    )

    # Holdout Metrics Code
    cells.append(
        nbformat.v4.new_code_cell(
            "holdout_df = pd.read_csv('reports/phase4/tables/holdout_metrics.csv')\n"
            "display(holdout_df[['model', 'pr_auc_point', 'pr_auc_ci95_low', 'pr_auc_ci95_high', 'recall_at_10_point', 'recall_at_20_point', 'brier_score_point']])"
        )
    )

    # Figures FIG-15 through FIG-20
    figures = [
        (
            "FIG-15: Precision-Recall Curves across Attrition Risk Models",
            "reports/phase4/figures/fig15_precision_recall_curves.png",
            "Evaluates model precision across all recall thresholds on unseen holdout data. "
            "Calibrated Logistic Regression (Set B) achieves PR-AUC = 0.5690 vs chance baseline of 0.1599.",
        ),
        (
            "FIG-16: Probability Reliability Diagram (Calibration Curve)",
            "reports/phase4/figures/fig16_calibration_curves.png",
            "Demonstrates the effect of Platt/Sigmoid probability scaling. "
            "Calibrated probabilities align closely with empirical departure frequencies across all 5 risk bins.",
        ),
        (
            "FIG-17: Holdout Attrition Yield Across Prioritized Risk Tiers",
            "reports/phase4/figures/fig17_risk_tier_attrition.png",
            "Partitions the holdout workforce into Top 10%, Next 10%, and Remaining 80%. "
            "The top 10% risk tier yields an observed departure rate of 65.5% (nearly 4x the baseline).",
        ),
        (
            "FIG-18: Top Predictors in Regularised Logistic Regression",
            "reports/phase4/figures/fig18_feature_odds_ratios.png",
            "Forest plot of standardized odds ratios with 95% confidence intervals. "
            "Overtime (OR = 2.00) and frequent business travel (OR = 1.73) dominate as operational risk factors.",
        ),
        (
            "FIG-19: Out-of-Sample Validation of Candidate Attrition Hotspots",
            "reports/phase4/figures/fig19_hotspot_validation.png",
            "Compares training split rates to holdout split rates for screened candidate segments. "
            "Identifies segments that generalized out-of-sample vs those that faded or were underpowered in holdout.",
        ),
        (
            "FIG-20: Demographic Fairness Audit at Operational 20% Triage Threshold",
            "reports/phase4/figures/fig20_demographic_fairness.png",
            "Audits model selection rate and leaver recall across Gender, MaritalStatus, and AgeGroup. "
            "Highlights disparate impact implications under the Four-Fifths rule in proactive support triage.",
        ),
    ]

    for title, img_path, caption in figures:
        cells.append(nbformat.v4.new_markdown_cell(f"### {title}\n\n*{caption}*"))
        cells.append(nbformat.v4.new_code_cell(f"display(Image(filename='{img_path}'))"))

    # Hotspot Validation Table
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 3. Hotspot Candidate Holdout Validation Summary Table"
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            "val_table = pd.read_csv('reports/phase4/tables/hotspot_validation.csv')\n"
            "display(val_table[['segment', 'train_rate', 'holdout_n', 'holdout_rate', 'holdout_ci_low', 'holdout_ci_high', 'validation_status']].head(15))"
        )
    )

    # Demographic Fairness Table
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## 4. Demographic Fairness Audit Summary Table"
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            "fair_table = pd.read_csv('reports/phase4/tables/demographic_fairness.csv')\n"
            "display(fair_table[['attribute', 'subgroup', 'sample_size', 'selection_rate', 'recall', 'disparate_impact_ratio']])"
        )
    )

    nb.cells = cells
    return nb

def main():
    nb = build_notebook()
    nb_path = PROJECT_ROOT / "notebooks" / "04_modelling.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Notebook written to {nb_path}")

    print("Executing notebook headless...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(PROJECT_ROOT)}})

    exec_path = PROJECT_ROOT / "reports" / "phase4" / "04_modelling.executed.ipynb"
    with open(exec_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Executed notebook written to {exec_path}")

if __name__ == "__main__":
    main()
