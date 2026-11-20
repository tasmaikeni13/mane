#!/usr/bin/env bash
# Pilot run replicator for AdamW, Lion, and MANE.
# Runs 4k steps for validation checks.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

P="python3 src/train.py --steps 4000 --warmup 1000 --log-every 250 --diag-every 2000 --seed 0"

echo "Launching AdamW pilot on GPU 0..."
$P --optimizer adamw --lr 1e-3 --weight-decay 0.06 --gpu 0 --tag _pilot_lr1e-3 > logs/pilot_adamw_1e-3.log 2>&1 &

echo "Launching vanilla Lion pilot on GPU 0..."
$P --optimizer lion --lr 5e-5 --weight-decay 1.2 --gpu 0 --tag _pilot_lion_5e-5 > logs/pilot_lion_5e-5.log 2>&1 &

echo "Launching MANE pilot on GPU 1..."
$P --optimizer mane --lr 1.4e-4 --weight-decay 0.43 --budget-ratio 1.0 --trust-cap 3.0 --gpu 1 --tag _pilot_lr14e-5 > logs/pilot_mane_1.4e-4.log 2>&1 &

wait
echo "All pilots completed successfully."
