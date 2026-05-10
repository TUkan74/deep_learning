#!/bin/bash
#SBATCH -J readcomp
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o reading_%j.out
#SBATCH -e reading_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
ARGS="${ARGS:---epochs 3 --batch_size 8 --learning_rate 3e-5 --max_length 384 --stride 128 --warmup_ratio 0.1 --weight_decay 0.01}"

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

srun "$PYTHON_BIN" labs/10/reading_comprehension.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS

LATEST_DEV="$(find logs -name reading_comprehension_dev.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
LATEST_TEST="$(find logs -name reading_comprehension.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"

if [[ -n "${LATEST_DEV:-}" ]]; then
  echo "Latest dev prediction file: $LATEST_DEV"
  "$PYTHON_BIN" -m npfl138.datasets.reading_comprehension_dataset --evaluate "$LATEST_DEV" --dataset dev
fi

if [[ -n "${LATEST_TEST:-}" ]]; then
  echo "Latest test prediction file: $LATEST_TEST"
fi
