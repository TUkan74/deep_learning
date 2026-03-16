#!/bin/bash
set -euo pipefail

AIC_HOST="${AIC_HOST:-aic.ufal.mff.cuni.cz}"

if [[ $# -ge 1 && "$1" != -* ]]; then
  AIC_LOGIN="$1"
  shift
elif [[ -n "${AIC_LOGIN:-}" ]]; then
  :
else
  echo "Usage: $0 LOGIN [ssh options]" >&2
  echo "Or set AIC_LOGIN in the environment." >&2
  exit 1
fi

exec ssh "${AIC_LOGIN}@${AIC_HOST}" "$@"
