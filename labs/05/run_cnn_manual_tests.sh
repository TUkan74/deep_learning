#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
AIC_VENV_PYTHON="${HOME}/venvs/npfl138-2526/bin/python"
PYTHON311="/opt/python/3.11.7/bin/python3"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$AIC_VENV_PYTHON" ]]; then
    PYTHON_BIN="$AIC_VENV_PYTHON"
  elif [[ -x "$PYTHON311" ]]; then
    PYTHON_BIN="$PYTHON311"
  else
    PYTHON_BIN="python3"
  fi
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/labs${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-1-1
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-3-1
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-3-2
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=5-3-2,10-3-2
"$PYTHON_BIN" labs/05/cnn_manual.py --recodex --epochs=1 --cnn=30-1-1,20-3-2
