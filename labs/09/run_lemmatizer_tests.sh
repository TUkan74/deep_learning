#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="${PYTHON_BIN:-}"

cd "$PROJECT_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export PYTHONPATH="$PROJECT_DIR/labs:${PYTHONPATH:-}"

run() {
  echo
  echo "$*"
  "$@"
}

run "$PYTHON_BIN" labs/09/lemmatizer_noattn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=64 --rnn_dim=32
run "$PYTHON_BIN" labs/09/lemmatizer_noattn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings

run "$PYTHON_BIN" labs/09/lemmatizer_attn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=64 --rnn_dim=32
run "$PYTHON_BIN" labs/09/lemmatizer_attn.py --recodex --epochs=1 --max_sentences=500 --batch_size=2 --cle_dim=32 --rnn_dim=32 --tie_embeddings
