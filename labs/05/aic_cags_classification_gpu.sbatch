#!/bin/bash
#SBATCH -J cags-cls
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -o cags_cls_%j.out
#SBATCH -e cags_cls_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
ARGS="${ARGS:---batch_size 64 --epochs 3 --finetune_epochs 12 --learning_rate 3e-3 --finetune_learning_rate 3e-5 --dropout 0.3 --weight_decay 1e-4 --patience 5}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/05/aic_setup_env.sh first." >&2
  exit 1
fi

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Arguments: $ARGS"

srun "$PYTHON_BIN" labs/05/cags_classification.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS

LATEST_DEV="$(find logs -name cags_classification_dev.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
LATEST_TEST="$(find logs -name cags_classification.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"

if [[ -n "${LATEST_DEV:-}" ]]; then
  echo "Latest dev prediction file: $LATEST_DEV"
  "$PYTHON_BIN" -m npfl138.datasets.cags --evaluate_classification "$LATEST_DEV" --dataset dev
fi

if [[ -n "${LATEST_TEST:-}" ]]; then
  echo "Latest test prediction file: $LATEST_TEST"
fi
