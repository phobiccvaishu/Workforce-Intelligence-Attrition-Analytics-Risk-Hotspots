"""CLI Runner for Final Deliverables (Research Paper, Executive Summary, Figures).

Builds:
- reports/final/paper.md
- reports/final/paper.docx
- reports/final/paper.pdf (if Word COM is available)
- reports/final/executive_summary.md
- reports/final/executive_summary.docx
- reports/final/executive_summary.pdf (if Word COM is available)
- reports/final/figures/* (copied canonical figures)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from workforce_intel.final_docs import build_all_docs


def main() -> int:
    try:
        build_all_docs()
        return 0
    except Exception as exc:
        print(f"ERROR: Final deliverables build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
