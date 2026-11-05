# Workforce Intelligence: Pre-Push Hygiene & Repository Release Checklist

This document details mandatory operational and governance checks to execute before any remote repository push or public release. **In accordance with project safety rules, git history rewriting must NOT be performed during active development phases; execute these steps during final deployment preparation.**

---

## 1. History Scrubbing & Sanitization
- [ ] **Scrub Tracked Run Logs from Git History:**
  - Remove all historical commits containing tracked `run.log` files (`reports/phase1/run.log`, `reports/phase2/run.log`, `reports/phase3/run.log`).
  - Target historical commits including `6a0bf37`, `940257f`, and any other intermediate commits containing local machine paths.
  - Utilize `git-filter-repo` or BFG Repo-Cleaner to permanently scrub any occurrences of local Windows paths (e.g., paths containing local user names).
- [ ] **Verify Working Tree `.gitignore`:**
  - Confirm that `reports/**/run.log` and all local configuration overrides remain permanently untracked.

---

## 2. Identity, Remote Security & Access Governance
- [ ] **Commit Author Identity & Email:**
  - Standardize author and committer emails across all commits (e.g., configure a GitHub privacy-protected noreply email address: `<username>@users.noreply.github.com`).
- [ ] **Remote Repository Privacy:**
  - Confirm that the designated remote repository is explicitly set to **Private**. Never push proprietary project infrastructure or dataset documentation to a public repository without authorization.
- [ ] **Provenance Disclaimer & Attribution:**
  - Ensure all documentation maintains the explicit notice: *"Public/synthetic HR dataset (provenance unverified). Never describe as real Palo Alto Networks records."*

---

## 3. Data Storage & Repository Cleanliness
- [ ] **Raw Dataset Tracking Decision:**
  - Decide whether `data/raw/Palo Alto Networks.csv` remains tracked in git or is managed via DVC / secure cloud object storage.
  - If retained in git, verify that `Palo Alto Networks.csv` in the root workspace remains untracked.
- [ ] **Directory Name Sanitization:**
  - Rename the local project directory to remove spaces and special characters (`Workforce Intelligence  Attrition Analytics & Risk Hotspots` -> `workforce-intelligence-attrition-analytics`) prior to the Streamlit dashboard phase to prevent shell quoting bugs across environments.
