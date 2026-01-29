#!/bin/bash
set -euo pipefail

REPO="/Users/omer/Desktop/ag"
cd "$REPO"

mkdir -p runs/daily
RUN_TAG=$(date +%Y%m%d)

./.venv/bin/python scripts/aci_pipeline.py \
  --config configs/ablations/yahoo_big_lsq_smin_0.yaml \
  --checkpoint artifacts/yahoo_big_lsq_smin_0.pt \
  --split test \
  --preds_out runs/daily/lsq0_preds_${RUN_TAG}.npz \
  --aci_out runs/daily/lsq0_aci_${RUN_TAG}.npz \
  --aci_metrics_out runs/daily/lsq0_aci_metrics_${RUN_TAG}.npz \
  --coverage_mode per_horizon \
  --coverage_target 0.80 \
  --window 720 \
  --recent_window 720 \
  --vol_bins 4 \
  --state_in artifacts/aci_state_lsq0_v4.json \
  --state_out artifacts/aci_state_lsq0_v4.json \
  >> runs/daily/lsq0_aci_${RUN_TAG}.log 2>&1
