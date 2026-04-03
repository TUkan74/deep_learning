#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/labs${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" labs/06/bboxes_utils.py
