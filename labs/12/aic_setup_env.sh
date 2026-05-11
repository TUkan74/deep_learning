#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON311="${PYTHON311:-/opt/python/3.11.7/bin/python3}"

if [[ ! -d "$PROJECT_DIR/labs" ]]; then
  echo "Expected repository at $PROJECT_DIR." >&2
  echo "Override with PROJECT_DIR=/path/to/deep_learning" >&2
  exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON311" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -e "$PROJECT_DIR/labs"

echo "Environment ready:"
echo "  PROJECT_DIR=$PROJECT_DIR"
echo "  VENV_DIR=$VENV_DIR"
