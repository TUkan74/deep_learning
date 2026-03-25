#!/bin/bash
#SBATCH -J cifar
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH -o cifar_%j.out
#SBATCH -e cifar_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
ARGS="${ARGS:---epochs 60 --batch_size 256 --channels 96 --dropout 0.15 --classifier_dropout 0.35 --learning_rate 7e-4 --weight_decay 2e-4 --label_smoothing 0.1}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/04/aic_setup_env.sh first." >&2
  exit 1
fi

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Arguments: $ARGS"

srun "$PYTHON_BIN" labs/04/cifar_competition.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS

LATEST_DEV="$(find logs -name cifar_competition_dev.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
LATEST_TEST="$(find logs -name cifar_competition_test.txt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"

if [[ -n "${LATEST_DEV:-}" ]]; then
  echo "Latest dev prediction file: $LATEST_DEV"
  "$PYTHON_BIN" -m npfl138.datasets.cifar10 --evaluate "$LATEST_DEV" --dataset dev
fi

if [[ -n "${LATEST_TEST:-}" ]]; then
  echo "Latest test prediction file: $LATEST_TEST"
fi
