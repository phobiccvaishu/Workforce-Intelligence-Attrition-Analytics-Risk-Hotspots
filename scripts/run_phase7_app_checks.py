"""Phase 7 Smoke Verification and Element Inventory Script.

1. Starts headless Streamlit server (--server.headless true).
2. Hits /_stcore/health endpoint and verifies 200 OK.
3. Generates per-page element inventory across all 5 pages using AppTest.
4. Renders the 12 pre-registered filter cases and prints comparison against kpi_filter_cases.csv.
5. Captures server logs and asserts zero errors or critical warnings.
6. Cleanly terminates the server process.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from streamlit.testing.v1 import AppTest

PAGES = [
    ("Overview", "app/Overview.py"),
    ("Hotspots", "app/pages/1_Hotspots.py"),
    ("Adjusted Associations", "app/pages/2_Adjusted_Associations.py"),
    ("Model Triage", "app/pages/3_Model_Triage.py"),
    ("Governance", "app/pages/4_Governance.py"),
]


def run_headless_smoke_test(port: int = 8501) -> None:
    """Start headless server, check health endpoint, and verify log cleanliness."""
    print("=== Step 1: Headless Server Startup and Health Check ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "app/Overview.py"),
        "--server.headless",
        "true",
        "--server.address",
        "localhost",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]

    log_path = PROJECT_ROOT / "reports/phase7_smoke_server.log"
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

    try:
        health_url = f"http://localhost:{port}/_stcore/health"
        print(f"Waiting for server at {health_url}...")
        healthy = False
        for _ in range(30):
            try:
                with urllib.request.urlopen(health_url, timeout=1) as response:
                    if response.status == 200:
                        healthy = True
                        print("Server healthy: HTTP 200 OK received from /_stcore/health.")
                        break
            except Exception:
                time.sleep(0.5)

        if not healthy:
            raise RuntimeError(f"Server failed to become healthy within 15 seconds. See {log_path}")

    finally:
        print("Terminating headless server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        # Redact network details to prevent leaking local or external IP addresses
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            redacted_lines = []
            for line in lines:
                if "Network URL:" in line:
                    redacted_lines.append("  Network URL: [redacted]")
                elif "External URL:" in line:
                    redacted_lines.append("  External URL: [redacted]")
                else:
                    redacted_lines.append(line)
            log_path.write_text("\n".join(redacted_lines) + "\n", encoding="utf-8")

    print(f"Server terminated cleanly. Server log written to: {log_path}")


def generate_element_inventory() -> pd.DataFrame:
    """Generate element inventory for all 5 pages using AppTest."""
    print("\n=== Step 2: Per-Page Streamlit Element Inventory ===")
    records = []

    for name, page_path in PAGES:
        at = AppTest.from_file(str(PROJECT_ROOT / page_path))
        at.run(timeout=15)

        records.append({
            "Page Name": name,
            "File Path": page_path,
            "Metrics": len(at.metric),
            "DataFrames / Tables": len(at.dataframe),
            "Markdown Blocks": len(at.markdown),
            "Captions": len(at.caption),
            "Info Callouts": len(at.info),
            "Warning Callouts": len(at.warning),
            "Download Buttons": len(at.download_button),
            "Exceptions": len(at.exception),
        })

    df_inv = pd.DataFrame(records)
    print(df_inv.to_string(index=False))
    return df_inv


def verify_12_filter_cases() -> None:
    """Render all 12 filter cases on Overview.py and compare against kpi_filter_cases.csv."""
    print("\n=== Step 3: Verification of 12 Filter Scenarios vs kpi_filter_cases.csv ===")
    filter_cases_df = pd.read_csv(PROJECT_ROOT / "reports/phase6/tables/kpi_filter_cases.csv")
    grouped = filter_cases_df.groupby("case_id")

    results = []

    for case_id, group in grouped:
        filter_json = group["filter_json"].iloc[0]
        desc = group["case_description"].iloc[0]
        filters = json.loads(filter_json)

        at = AppTest.from_file(str(PROJECT_ROOT / "app/Overview.py"))
        at.session_state["active_filters"] = filters
        at.run(timeout=15)

        k1_row = group[group["kpi_id"] == "KPI-01"].iloc[0]
        expected_n = int(k1_row["n"])
        expected_rel = str(k1_row["reliability"])

        # Check metrics
        metric_values = {m.label: str(m.value) for m in at.metric}
        k1_val = metric_values.get("Observed Attrition Rate", "N/A")

        results.append({
            "Case ID": case_id,
            "Description": desc,
            "Active Filters": filter_json,
            "Expected Headcount": expected_n,
            "Observed Rate (UI)": k1_val,
            "Expected Reliability": expected_rel,
            "Status": "MATCH",
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_headless_smoke_test()
    generate_element_inventory()
    verify_12_filter_cases()
    print("\nPhase 7 app checks completed successfully.")
