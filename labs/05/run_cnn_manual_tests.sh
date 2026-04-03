#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/labs${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-1-1
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-3-1
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-3-2
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-3-2,10-3-2
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=30-1-1,20-3-2
