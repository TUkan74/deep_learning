#!/bin/bash
#SBATCH -J tagtrans
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o tagtrans_%j.out
#SBATCH -e tagtrans_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
ARGS="${ARGS:---epochs 5 --batch_size 10 --max_sentences 5000 --transformer_heads 4 --transformer_layers 2 --transformer_dropout 0.1}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/10/aic_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/labs:${PYTHONPATH:-}"

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Arguments: $ARGS"

srun "$PYTHON_BIN" labs/10/tagger_transformer.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS
