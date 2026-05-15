#!/bin/bash
#SBATCH -J flow-att
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -o flow_matching_attention_%j.out
#SBATCH -e flow_matching_attention_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
ARGS="${ARGS:---recodex --epochs=1 --epoch_batches=16 --batch_size=8 --stages=2 --stage_blocks=1 --channels=6 --ema=0.9 --sampling_steps=8 --attention_stages=0 --attention_heads=4}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/13/aic_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/labs:${PYTHONPATH:-}"

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Arguments: $ARGS"

srun "$PYTHON_BIN" labs/13/flow_matching_attention.py --threads "${SLURM_CPUS_PER_TASK:-1}" $ARGS
