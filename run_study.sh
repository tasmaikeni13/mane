#!/usr/bin/env bash
# Main-study launcher: one 50k-step run per (optimizer, seed), one GPU each.
#
# Usage:
#   ./run_study.sh <optimizer> <lr> <weight_decay> <seed> <gpu> [extra args…]
# Examples:
#   ./run_study.sh adamw 6e-4  0.1  0 0
#   ./run_study.sh lion  1e-4  0.6  0 1
#   ./run_study.sh mane  1.4e-4 0.43 0 1 --budget-ratio 1.0 --trust-cap 3.0
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

OPT="${1:?optimizer}"; LR="${2:?lr}"; WD="${3:?wd}"; SEED="${4:-0}"; GPU="${5:-0}"
shift 5 || true

exec python3 src/train.py --optimizer "$OPT" --lr "$LR" --weight-decay "$WD" \
  --seed "$SEED" --gpu "$GPU" --steps 50000 "$@" \
  > "logs/main_${OPT}_s${SEED}.log" 2>&1
