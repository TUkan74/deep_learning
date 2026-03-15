#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: neither python3 nor python is available in PATH." >&2
  exit 2
fi

SCRIPT="mnist_training.py"

NAMES=(
  "sgd_lr_0.01"
  "sgd_lr_0.01_momentum_0.9"
  "sgd_lr_0.1"
  "adam_lr_0.001"
  "adam_lr_0.01"
  "adam_linear_0.01_to_0.0001"
  "adam_exponential_0.01_to_0.001"
  "adam_cosine_0.01_to_0.0001"
)

CMDS=(
  "--recodex --epochs=1 --optimizer=SGD --learning_rate=0.01"
  "--recodex --epochs=1 --optimizer=SGD --learning_rate=0.01 --momentum=0.9"
  "--recodex --epochs=1 --optimizer=SGD --learning_rate=0.1"
  "--recodex --epochs=1 --optimizer=Adam --learning_rate=0.001"
  "--recodex --epochs=1 --optimizer=Adam --learning_rate=0.01"
  "--recodex --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=linear --learning_rate_final=0.0001"
  "--recodex --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=exponential --learning_rate_final=0.001"
  "--recodex --epochs=2 --optimizer=Adam --learning_rate=0.01 --decay=cosine --learning_rate_final=0.0001"
)

# Loose but useful checks (results vary by hardware and accelerator).
MIN_DEV_ACC=(
  "0.9000"
  "0.9400"
  "0.9400"
  "0.9550"
  "0.9550"
  "0.9700"
  "0.9700"
  "0.9700"
)

EXPECTED_NEXT_LR=(
  ""
  ""
  ""
  ""
  ""
  "0.0001"
  "0.001"
  "0.0001"
)

TOL_LR="1e-6"

pass=0
fail=0

echo "Running ${#CMDS[@]} mnist_training tests using: $PYTHON_BIN"
echo
printf "%-2s  %-34s %-6s %-11s %-11s\n" "#" "name" "status" "dev_acc" "next_lr"
printf -- "%.0s-" {1..72}
echo

for i in "${!CMDS[@]}"; do
  idx=$((i + 1))
  name="${NAMES[$i]}"
  cmd="${CMDS[$i]}"
  min_acc="${MIN_DEV_ACC[$i]}"
  expected_lr="${EXPECTED_NEXT_LR[$i]}"

  tmp="$(mktemp)"

  if "$PYTHON_BIN" "$SCRIPT" $cmd >"$tmp" 2>&1; then
    status="PASS"

    dev_acc="$(grep -Eo 'dev:accuracy=[0-9.]+' "$tmp" | tail -n1 | cut -d= -f2 || true)"
    if [[ -z "$dev_acc" ]]; then
      status="FAIL"
    else
      if ! awk -v got="$dev_acc" -v min="$min_acc" 'BEGIN { exit !(got + 0 >= min + 0) }'; then
        status="FAIL"
      fi
    fi

    shown_lr="-"
    if [[ -n "$expected_lr" ]]; then
      shown_lr="$(grep -Eo 'Next learning rate to be used: [0-9.eE+-]+' "$tmp" | awk '{print $NF}' | tail -n1 || true)"
      if [[ -z "$shown_lr" ]]; then
        status="FAIL"
        shown_lr="missing"
      else
        if ! awk -v got="$shown_lr" -v exp="$expected_lr" -v tol="$TOL_LR" \
          'BEGIN { d = got - exp; if (d < 0) d = -d; exit !(d <= tol) }'; then
          status="FAIL"
        fi
      fi
    fi

    if [[ "$status" == "PASS" ]]; then
      pass=$((pass + 1))
    else
      fail=$((fail + 1))
      cp "$tmp" "mnist_training_test_${idx}.log"
    fi

    printf "%-2s  %-34s %-6s %-11s %-11s\n" "$idx" "$name" "$status" "${dev_acc:--}" "$shown_lr"
  else
    fail=$((fail + 1))
    cp "$tmp" "mnist_training_test_${idx}.log"
    printf "%-2s  %-34s %-6s %-11s %-11s\n" "$idx" "$name" "FAIL" "-" "-"
  fi

  rm -f "$tmp"
done

printf -- "%.0s-" {1..72}
echo
printf "PASS: %d  FAIL: %d\n" "$pass" "$fail"

if [[ "$fail" -gt 0 ]]; then
  echo "Saved failing logs as mnist_training_test_<n>.log"
  exit 1
fi
