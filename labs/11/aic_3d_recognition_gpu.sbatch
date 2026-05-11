#!/bin/bash
#SBATCH -J modelnet
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o modelnet_%j.out
#SBATCH -e modelnet_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
MODELNET_DIM="${MODELNET_DIM:-32}"
ARGS="${ARGS:---modelnet $MODELNET_DIM --epochs 60 --batch_size 64 --learning_rate 1e-3 --patience 10 --workers 2}"

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

srun "$PYTHON_BIN" labs/11/3d_recognition.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS

LATEST_DEV="$(find logs -name 3d_recognition_dev.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
LATEST_TEST="$(find logs -name 3d_recognition.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"

if [[ -n "${LATEST_DEV:-}" ]]; then
  echo "Latest dev prediction file: $LATEST_DEV"
  "$PYTHON_BIN" -m npfl138.datasets.modelnet --evaluate "$LATEST_DEV" --dataset dev --dim "$MODELNET_DIM"
fi

if [[ -n "${LATEST_TEST:-}" ]]; then
  echo "Latest test prediction file: $LATEST_TEST"
fi
