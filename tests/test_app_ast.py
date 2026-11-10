"""AST Invariant Enforcement Tests for Workforce Intelligence Streamlit Application.

Parses every Python file under the app/ directory using Python's ast module and strictly
verifies that:
1. No statistical computation libraries (scipy, statsmodels, sklearn) are imported.
2. No banned calculation methods (groupby, value_counts) are invoked.
3. No internal inference or exploratory modules are directly imported (pure presentation rule).
"""

import ast
from pathlib import Path
import pytest

from workforce_intel.config import PROJECT_ROOT

APP_DIR = PROJECT_ROOT / "app"
BANNED_MODULES = {"scipy", "statsmodels", "sklearn", "stats_utils", "inference", "hotspot_index", "eda"}
BANNED_METHODS = {"groupby", "value_counts"}


def get_app_python_files():
    """Retrieve all Python source files within app/ and app/pages/."""
    files = list(APP_DIR.rglob("*.py"))
    assert len(files) >= 5, f"Expected at least 5 app files, found {len(files)}"
    return files


def check_ast_compliance(source_code: str, filename: str = "<string>"):
    """Walk AST of source code and assert no banned imports or method calls exist."""
    tree = ast.parse(source_code, filename=filename)
    violations = []

    for node in ast.walk(tree):
        # Check standard imports (e.g. import scipy)
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in BANNED_MODULES:
                    violations.append(f"Banned import '{alias.name}' at line {node.lineno}")

        # Check from imports (e.g. from scipy.stats import norm)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in BANNED_MODULES:
                    violations.append(f"Banned from-import '{node.module}' at line {node.lineno}")

        # Check method calls (e.g. df.groupby(...) or df.value_counts(...))
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_METHODS:
                violations.append(f"Banned method call '.{node.attr}' at line {node.lineno}")

    return violations


def test_app_ast_no_banned_imports_or_methods():
    """Verify all files in app/ adhere to pure presentation constraints."""
    app_files = get_app_python_files()
    all_violations = {}

    for file_path in app_files:
        code = file_path.read_text(encoding="utf-8")
        violations = check_ast_compliance(code, filename=str(file_path))
        if violations:
            all_violations[file_path.name] = violations

    assert not all_violations, f"AST invariant violations found in app/: {all_violations}"


def test_mutation_ast_detects_banned_import():
    """Mutation test: Assert AST checker flags banned import 'scipy'."""
    bad_code = "import scipy.stats as stats\nx = 1"
    violations = check_ast_compliance(bad_code)
    assert any("scipy" in v for v in violations)


def test_mutation_ast_detects_banned_groupby():
    """Mutation test: Assert AST checker flags banned method '.groupby'."""
    bad_code = "result = df.groupby('Department').size()"
    violations = check_ast_compliance(bad_code)
    assert any("groupby" in v for v in violations)


def test_mutation_ast_detects_banned_value_counts():
    """Mutation test: Assert AST checker flags banned method '.value_counts'."""
    bad_code = "counts = df['Gender'].value_counts()"
    violations = check_ast_compliance(bad_code)
    assert any("value_counts" in v for v in violations)
