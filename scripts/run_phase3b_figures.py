"""Phase 3B execution script: Static figures, findings register, and thin notebook.

Idempotent pipeline that loads Phase 3A tables, generates 14 static figures,
creates the figure manifest, compiles the findings register and summary JSON,
and builds/executes the thin notebook. Zero re-computation or data modifications.
"""

from hashlib import sha256
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pandas as pd

from workforce_intel.config import (
    FIGURE_DPI,
    FIGURE_SPECS,
    FINDINGS_REGISTER_MD,
    FINDINGS_RULE,
    FINDINGS_SUMMARY_JSON,
    NOTEBOOK_FIGURES_IPYNB,
    EXECUTED_FIGURES_IPYNB,
    PHASE3_FIGURES_DIR,
    PHASE3_MANIFEST_CSV,
    PHASE3_SUMMARY_JSON,
    PHASE3_TABLES_DIR,
)
from workforce_intel.findings import (
    build_findings,
    compile_findings_summary,
    render_register_markdown,
)
import workforce_intel.viz as viz


def setup_logging() -> logging.Logger:
    """Configure console logging for Phase 3B pipeline."""
    logger = logging.getLogger("workforce_intel.phase3b")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_phase3_tables(tables_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all 46 CSV tables from Phase 3A directory."""
    if not tables_dir.exists():
        raise FileNotFoundError(f"Phase 3A tables directory not found at {tables_dir}")

    tables: Dict[str, pd.DataFrame] = {}
    csv_files = sorted(tables_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV tables found in {tables_dir}")

    for f in csv_files:
        df = pd.read_csv(f)
        tables[f.name] = df
        tables[f.stem] = df  # Allow access by filename or stem

    return tables


def generate_figures(
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Generate all 14 report-quality figures and record manifest data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_functions = [
        ("FIG-01", viz.plot_fig01_department_overview),
        ("FIG-02", viz.plot_fig02_jobrole_attrition),
        ("FIG-03", viz.plot_fig03_funnel_plot),
        ("FIG-04", viz.plot_fig04_leaver_concentration),
        ("FIG-05", viz.plot_fig05_age_sensitivity),
        ("FIG-06", viz.plot_fig06_tenure_sensitivity),
        ("FIG-07", viz.plot_fig07_promotion_gap_cohort),
        ("FIG-08", viz.plot_fig08_workload_mobility),
        ("FIG-09", viz.plot_fig09_satisfaction_small_multiples),
        ("FIG-10", viz.plot_fig10_overtime_interactions),
        ("FIG-11", viz.plot_fig11_hotspot_candidates),
        ("FIG-12", viz.plot_fig12_hotspot_overlap),
        ("FIG-13", viz.plot_fig13_numeric_distributions),
        ("FIG-14", viz.plot_fig14_demographics_background),
    ]

    manifest_rows: List[Dict[str, Any]] = []

    for fid, func in fig_functions:
        spec = FIGURE_SPECS[fid]
        file_name = spec["file"]
        target_path = output_dir / file_name

        fig, plotted_df = func(tables)

        # Save figure with clean deterministic metadata
        fig.savefig(
            target_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            metadata={"Software": "Matplotlib", "Author": "Workforce Intelligence Engine"},
        )
        plt.close(fig)

        # Compute SHA-256 hash of plotted DataFrame representation
        df_csv_bytes = plotted_df.to_csv(index=False).encode("utf-8")
        data_hash = sha256(df_csv_bytes).hexdigest()

        sources_str = "; ".join(spec["source_tables"])

        manifest_rows.append({
            "figure_id": fid,
            "file": file_name,
            "title": spec["title"],
            "source_tables": sources_str,
            "plotted_rows": len(plotted_df),
            "plotted_data_sha256": data_hash,
            "alt_text": spec["alt_text"],
        })

        logger.info(f"Generated {fid}: {file_name} ({len(plotted_df)} rows plotted)")

    return manifest_rows


def build_thin_notebook(
    manifest_rows: List[Dict[str, Any]],
    notebook_path: Path,
) -> nbformat.NotebookNode:
    """Build thin Jupyter notebook displaying figures and findings register."""
    nb = nbformat.v4.new_notebook()

    # Cell 1: Title and Header
    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            "# Workforce Intelligence: Exploratory Data Analysis & Static Visualizations\n\n"
            "This thin presentation notebook displays the 14 pre-specified static figures generated from "
            "the 46 Phase 3A exploratory rate tables.\n\n"
            "> [!NOTE]\n"
            "> **Descriptive Analysis Notice:** All visuals report observed subgroups, counts, and statistical intervals.\n"
            "> They represent descriptive associations in a public/synthetic HR dataset (provenance unverified). No causal claims.\n"
        )
    )

    # Cell 2: Manifest display code
    nb.cells.append(
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "manifest_df = pd.read_csv('reports/phase3/figure_manifest.csv')\n"
            "manifest_df[['figure_id', 'file', 'title', 'plotted_rows']]\n"
        )
    )

    # Cells for each figure
    for row in manifest_rows:
        fid = row["figure_id"]
        fname = row["file"]
        title = row["title"]
        alt = row["alt_text"]

        nb.cells.append(
            nbformat.v4.new_markdown_cell(
                f"### {fid}: {title}\n\n"
                f"*Alt Text: {alt}*"
            )
        )
        nb.cells.append(
            nbformat.v4.new_code_cell(
                f"display(Image(filename='reports/phase3/figures/{fname}'))"
            )
        )

    # Final cell: Findings register summary
    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            "## Findings Register Summary\n\n"
            "The formal mechanical interval-flagging findings register is documented in `docs/findings_register.md`.\n"
            "Formal statistical tests, multivariable regression models, and multiplicity adjustments are reserved for Phase 4.\n"
        )
    )

    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return nb


def execute_notebook(
    notebook_path: Path,
    executed_path: Path,
    logger: logging.Logger,
) -> Tuple[int, bool]:
    """Execute thin notebook headlessly and save executed version."""
    logger.info(f"Executing thin notebook {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(Path.cwd())}})
        has_error = False
    except Exception as exc:
        logger.error(f"Error during notebook execution: {exc}")
        has_error = True

    executed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(executed_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    cell_count = len(nb.cells)
    logger.info(f"Saved executed notebook to {executed_path} ({cell_count} cells, error={has_error})")
    return cell_count, has_error


def main() -> None:
    """Run the complete Phase 3B static visualization and findings pipeline."""
    logger = setup_logging()
    logger.info("Starting Phase 3B Static Visualization & Findings Engine...")

    # 1. Load Phase 3A Tables
    logger.info(f"Loading Phase 3A tables from {PHASE3_TABLES_DIR}...")
    tables = load_phase3_tables(PHASE3_TABLES_DIR)
    logger.info(f"Successfully loaded {len(tables) // 2} distinct tables.")

    # 2. Generate 14 Static Figures
    logger.info("Generating 14 static report-quality figures...")
    manifest_rows = generate_figures(tables, PHASE3_FIGURES_DIR, logger)

    # 3. Save Figure Manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(PHASE3_MANIFEST_CSV, index=False, lineterminator="\n")
    logger.info(f"Saved figure manifest to {PHASE3_MANIFEST_CSV}")

    # 4. Build Findings Register & Summary JSON
    logger.info("Evaluating mechanical interval-flagging rule across dimensions...")
    findings_df = build_findings(tables, FINDINGS_RULE)
    eda_summary = None
    if PHASE3_SUMMARY_JSON.exists():
        with open(PHASE3_SUMMARY_JSON, "r", encoding="utf-8") as f:
            eda_summary = json.load(f)
    summary_dict = compile_findings_summary(tables, findings_df, eda_summary=eda_summary)

    FINDINGS_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_SUMMARY_JSON.write_text(
        json.dumps(summary_dict, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved findings summary JSON to {FINDINGS_SUMMARY_JSON}")

    register_md = render_register_markdown(findings_df, summary_dict)
    FINDINGS_REGISTER_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_REGISTER_MD.write_text(register_md, encoding="utf-8")
    logger.info(f"Saved findings register markdown to {FINDINGS_REGISTER_MD}")

    # 5. Build and Execute Thin Notebook
    logger.info("Building thin presentation notebook...")
    build_thin_notebook(manifest_rows, NOTEBOOK_FIGURES_IPYNB)
    execute_notebook(NOTEBOOK_FIGURES_IPYNB, EXECUTED_FIGURES_IPYNB, logger)

    logger.info("Phase 3B Static Visualization & Findings Pipeline completed successfully.")


if __name__ == "__main__":
    main()
