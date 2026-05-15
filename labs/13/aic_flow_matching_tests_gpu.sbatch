#!/bin/bash
#SBATCH -J flow-tests
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o flow_matching_tests_%j.out
#SBATCH -e flow_matching_tests_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/13/aic_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/labs:${PYTHONPATH:-}"

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"

run_test() {
  echo
  echo "== $* =="
  srun "$PYTHON_BIN" "$@" --threads "${SLURM_CPUS_PER_TASK:-1}"
}

run_test labs/13/flow_matching.py --recodex --epochs=1 --epoch_batches=16 --batch_size=8 --stages=2 --stage_blocks=2 --channels=8 --ema=0.9 --sampling_steps=8
run_test labs/13/flow_matching.py --recodex --epochs=1 --epoch_batches=10 --batch_size=12 --stages=3 --stage_blocks=1 --channels=12 --ema=0.8 --sampling_steps=7 --sigma_min=0.1

run_test labs/13/flow_matching_attention.py --recodex --epochs=1 --epoch_batches=16 --batch_size=8 --stages=2 --stage_blocks=1 --channels=6 --ema=0.9 --sampling_steps=8 --attention_stages=0 --attention_heads=4
run_test labs/13/flow_matching_attention.py --recodex --epochs=1 --epoch_batches=10 --batch_size=12 --stages=3 --stage_blocks=1 --channels=4 --ema=0.8 --sampling_steps=7 --attention_stages=1 --attention_heads=2

run_test labs/13/flow_matching_conditional.py --recodex --epochs=1 --epoch_batches=16 --batch_size=8 --stages=2 --stage_blocks=2 --channels=8 --ema=0.9 --sampling_steps=8
run_test labs/13/flow_matching_conditional.py --recodex --epochs=1 --epoch_batches=10 --batch_size=12 --stages=3 --stage_blocks=1 --channels=12 --ema=0.8 --sampling_steps=7 --sigma_min=0.1
