#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if [[ ! -x .venv/bin/python ]]; then "$PYTHON_BIN" -m venv .venv; fi
.venv/bin/python -m pip install -r requirements-lock.txt
mkdir -p artifacts
.venv/bin/python src/download.py
.venv/bin/python src/prepare.py
.venv/bin/python src/audit_annotations.py > artifacts/annotation_audit.log
.venv/bin/python src/check_contract.py > artifacts/contract_check.log
.venv/bin/python -u src/train.py --variant baseline > artifacts/baseline_train.log 2>&1
.venv/bin/python -u src/train.py --variant skip > artifacts/skip_train.log 2>&1
.venv/bin/python src/evaluate.py
.venv/bin/python src/figures.py
.venv/bin/python src/build_evidence.py
