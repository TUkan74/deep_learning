#!/bin/bash
#SBATCH -J rein-base
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o reinforce_baseline_%j.out
#SBATCH -e reinforce_baseline_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
ARGS="${ARGS:---episodes 200 --batch_size 5 --learning_rate 0.01 --value_learning_rate 0.01 --hidden_layer_size 64}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/11/aic_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/labs:${PYTHONPATH:-}"

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Arguments: $ARGS"

srun "$PYTHON_BIN" labs/11/reinforce_baseline.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS
