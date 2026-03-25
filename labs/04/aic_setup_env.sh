#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"

if [[ ! -x "$PROJECT_DIR/labs/03/aic_setup_env.sh" ]]; then
  echo "Missing shared environment setup script at $PROJECT_DIR/labs/03/aic_setup_env.sh." >&2
  exit 1
fi

"$PROJECT_DIR/labs/03/aic_setup_env.sh"
