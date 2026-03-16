#!/bin/bash
#SBATCH -J uppercase
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o uppercase_%j.out
#SBATCH -e uppercase_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
ARGS="${ARGS:---epochs 30 --batch_size 2048 --window 10 --embedding_dim 96 --hidden_layers 768 384 --dropout 0.2 --weight_decay 1e-4 --learning_rate 1e-3 --patience 5}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/03/aic_setup_env.sh first." >&2
  exit 1
fi

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Arguments: $ARGS"

srun "$PYTHON_BIN" labs/03/uppercase.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS

LATEST_PREDICTION="$(find logs -name uppercase_test.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [[ -n "${LATEST_PREDICTION:-}" ]]; then
  echo "Latest prediction file: $LATEST_PREDICTION"
fi
