#!/bin/bash
#SBATCH -J gen-tests
#SBATCH -p gpu
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o gen_tests_%j.out
#SBATCH -e gen_tests_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/deep_learning}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/npfl138-2526}"
PYTHON_BIN="$VENV_DIR/bin/python"
TASK="${TASK:-all}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment at $VENV_DIR." >&2
  echo "Run labs/12/aic_setup_env.sh first." >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR/labs:${PYTHONPATH:-}"

echo "Running on $(hostname)"
echo "Project directory: $PROJECT_DIR"
echo "Task selector: $TASK"

run_task() {
  local name="$1"
  shift
  if [[ "$TASK" == "all" || "$TASK" == "$name" ]]; then
    echo "Running $name: $*"
    srun "$PYTHON_BIN" "$@"
  fi
}

run_task vae labs/12/vae.py --recodex --dataset=mnist --train_size=500 --epochs=3 --z_dim=2 --threads "${SLURM_CPUS_PER_TASK:-1}"
run_task vae labs/12/vae.py --recodex --dataset=mnist --train_size=500 --epochs=3 --z_dim=100 --threads "${SLURM_CPUS_PER_TASK:-1}"
run_task gan labs/12/gan.py --recodex --dataset=mnist --train_size=490 --epochs=5 --z_dim=2 --threads "${SLURM_CPUS_PER_TASK:-1}"
run_task gan labs/12/gan.py --recodex --dataset=mnist --train_size=490 --epochs=5 --z_dim=100 --threads "${SLURM_CPUS_PER_TASK:-1}"
run_task dcgan labs/12/dcgan.py --recodex --dataset=mnist --train_size=490 --epochs=2 --z_dim=2 --threads "${SLURM_CPUS_PER_TASK:-1}"
run_task dcgan labs/12/dcgan.py --recodex --dataset=mnist --train_size=490 --epochs=2 --z_dim=100 --threads "${SLURM_CPUS_PER_TASK:-1}"
