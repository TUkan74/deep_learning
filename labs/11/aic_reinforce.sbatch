#!/bin/bash
#SBATCH -J reinforce
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o reinforce_%j.out
#SBATCH -e reinforce_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
TASK="${TASK:-all}"
REINFORCE_ARGS="${REINFORCE_ARGS:---episodes 1000 --batch_size 10 --learning_rate 0.01 --hidden_layer_size 64}"
BASELINE_ARGS="${BASELINE_ARGS:---episodes 200 --batch_size 5 --learning_rate 0.01 --value_learning_rate 0.01 --hidden_layer_size 64}"
PIXELS_ARGS="${PIXELS_ARGS:---controller 0.2 1.0 10.0 1.0}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/11/aic_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/labs:${PYTHONPATH:-}"

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Task selector: $TASK"

if [[ "$TASK" == "all" || "$TASK" == "reinforce" ]]; then
  echo "Running reinforce.py $REINFORCE_ARGS"
  srun "$PYTHON_BIN" labs/11/reinforce.py --threads "${SLURM_CPUS_PER_TASK:-1}" $REINFORCE_ARGS
fi

if [[ "$TASK" == "all" || "$TASK" == "baseline" ]]; then
  echo "Running reinforce_baseline.py $BASELINE_ARGS"
  srun "$PYTHON_BIN" labs/11/reinforce_baseline.py --threads "${SLURM_CPUS_PER_TASK:-1}" $BASELINE_ARGS
fi

if [[ "$TASK" == "all" || "$TASK" == "pixels" ]]; then
  echo "Running reinforce_pixels.py $PIXELS_ARGS"
  srun "$PYTHON_BIN" labs/11/reinforce_pixels.py --threads "${SLURM_CPUS_PER_TASK:-1}" $PIXELS_ARGS
fi
