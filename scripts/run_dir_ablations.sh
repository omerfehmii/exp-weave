#!/usr/bin/env bash
set -euo pipefail

PY=".venv/bin/python"
HORIZONS="1,2,3,4,5,6,7,24"
SPLIT_PURGE=24
SPLIT_EMBARGO=24
OUT_ROOT="runs/dir_ablations"
SUMMARY="${OUT_ROOT}/summary.csv"

mkdir -p "${OUT_ROOT}"

if [ ! -f "${SUMMARY}" ]; then
  echo "run_name,config,checkpoint,forecast_mae,forecast_pinball,forecast_coverage80,forecast_width80,aci_mae,aci_coverage80,aci_width80,dir_wMCC,dir_wAUC,dir_acc_mean,dir_mcc_mean,dir_auc_mean,dir_tp_sum,dir_tn_sum,dir_fp_sum,dir_fn_sum,bin_max_acc,bin_max_mcc,bin_max_auc,boot_h1_acc,boot_h1_acc_lo,boot_h1_acc_hi,boot_h1_mcc,boot_h1_mcc_lo,boot_h1_mcc_hi,boot_h24_acc,boot_h24_acc_lo,boot_h24_acc_hi,boot_h24_mcc,boot_h24_mcc_lo,boot_h24_mcc_hi" > "${SUMMARY}"
fi

RUNS=(
  "mag|configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_mag.yaml|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_mag.pt"
  "pcgrad_mag|configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_mag_pcgrad.yaml|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_mag_pcgrad.pt"
  "pcgrad_nomag|configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_nomag.yaml|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_nomag.pt"
  "gradnorm_mag|configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_mag_gradnorm.yaml|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_mag_gradnorm.pt"
)

for entry in "${RUNS[@]}"; do
  IFS="|" read -r RUN_NAME CFG_PATH CKPT_PATH <<< "${entry}"
  OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
  mkdir -p "${OUT_DIR}"

  echo "=== ${RUN_NAME} ==="
  echo "train: ${CFG_PATH}"
  ${PY} train.py --config "${CFG_PATH}"

  echo "eval: forecast"
  ${PY} eval.py \
    --config "${CFG_PATH}" \
    --checkpoint "${CKPT_PATH}" \
    --split test \
    --use_cqr false \
    --out_npz "${OUT_DIR}/preds.npz" \
    --metrics_out "${OUT_DIR}/forecast_metrics.npz" \
    --out_csv "${OUT_DIR}/forecast_horizon.csv" \
    --split_purge "${SPLIT_PURGE}" \
    --split_embargo "${SPLIT_EMBARGO}"

  echo "eval: aci"
  ${PY} scripts/aci_backtest.py \
    --config "${CFG_PATH}" \
    --preds "${OUT_DIR}/preds.npz" \
    --out_npz "${OUT_DIR}/aci_preds.npz" \
    --out_metrics "${OUT_DIR}/aci_metrics.npz" \
    --coverage_mode per_horizon \
    --coverage_target 0.80

  echo "eval: direction head"
  ${PY} scripts/direction_head_eval.py \
    --config "${CFG_PATH}" \
    --checkpoint "${CKPT_PATH}" \
    --split test \
    --out_csv "${OUT_DIR}/dir_head.csv" \
    --out_metrics "${OUT_DIR}/dir_head_metrics.npz" \
    --delta_mode origin \
    --split_purge "${SPLIT_PURGE}" \
    --split_embargo "${SPLIT_EMBARGO}" \
    --horizons "${HORIZONS}"

  echo "eval: dir bins"
  ${PY} scripts/dir_delta_bins.py \
    --config "${CFG_PATH}" \
    --preds "${OUT_DIR}/preds.npz" \
    --out_csv "${OUT_DIR}/dir_bins.csv" \
    --out_metrics "${OUT_DIR}/dir_bins_metrics.npz" \
    --delta_mode origin \
    --horizons "${HORIZONS}" \
    --split_purge "${SPLIT_PURGE}" \
    --split_embargo "${SPLIT_EMBARGO}"

  echo "eval: bootstrap ci"
  ${PY} scripts/dir_bootstrap_ci.py \
    --config "${CFG_PATH}" \
    --preds "${OUT_DIR}/preds.npz" \
    --out_csv "${OUT_DIR}/dir_bootstrap_ci.csv" \
    --delta_mode origin \
    --epsilon_mode quantile \
    --epsilon_q 0.33 \
    --horizons "${HORIZONS}" \
    --bootstrap_runs 200 \
    --block_size 24 \
    --split_purge "${SPLIT_PURGE}" \
    --split_embargo "${SPLIT_EMBARGO}"

  RUN_NAME="${RUN_NAME}" CFG_PATH="${CFG_PATH}" CKPT_PATH="${CKPT_PATH}" OUT_DIR="${OUT_DIR}" SUMMARY="${SUMMARY}" ${PY} - <<'PY'
import csv
import os
import numpy as np

run_name = os.environ["RUN_NAME"]
cfg = os.environ["CFG_PATH"]
ckpt = os.environ["CKPT_PATH"]
out_dir = os.environ["OUT_DIR"]
summary = os.environ["SUMMARY"]

def _load_metrics(path):
    data = np.load(path, allow_pickle=True)
    metrics = data["metrics"]
    if isinstance(metrics, np.ndarray):
        return metrics.item()
    return metrics

forecast = _load_metrics(os.path.join(out_dir, "forecast_metrics.npz"))
aci = _load_metrics(os.path.join(out_dir, "aci_metrics.npz"))
dir_m = _load_metrics(os.path.join(out_dir, "dir_head_metrics.npz"))
bins = _load_metrics(os.path.join(out_dir, "dir_bins_metrics.npz"))

bin_summary = bins.get("bin_summary", {})
bin_key = None
if bin_summary:
    def _hi(k):
        try:
            return float(k.split("-")[1])
        except Exception:
            return -1.0
    bin_key = sorted(bin_summary.keys(), key=_hi)[-1]
    bin_m = bin_summary[bin_key]
else:
    bin_m = {}

boot_path = os.path.join(out_dir, "dir_bootstrap_ci.csv")
boot = {}
if os.path.exists(boot_path):
    with open(boot_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = int(row["h"])
            metric = row["metric"]
            if h in (1, 24) and metric in ("acc", "mcc"):
                boot[(h, metric, "estimate")] = row["estimate"]
                boot[(h, metric, "ci_lo")] = row["ci_lo"]
                boot[(h, metric, "ci_hi")] = row["ci_hi"]

row = [
    run_name,
    cfg,
    ckpt,
    forecast.get("mae"),
    forecast.get("pinball"),
    forecast.get("coverage80"),
    forecast.get("width80"),
    aci.get("mae"),
    aci.get("coverage80"),
    aci.get("width80"),
    dir_m.get("dirscore_wMCC"),
    dir_m.get("dirscore_wAUC"),
    dir_m.get("dir_acc_mean"),
    dir_m.get("dir_mcc_mean"),
    dir_m.get("dir_auc_mean"),
    dir_m.get("dir_tp_sum"),
    dir_m.get("dir_tn_sum"),
    dir_m.get("dir_fp_sum"),
    dir_m.get("dir_fn_sum"),
    bin_m.get("acc"),
    bin_m.get("mcc"),
    bin_m.get("auc"),
    boot.get((1, "acc", "estimate")),
    boot.get((1, "acc", "ci_lo")),
    boot.get((1, "acc", "ci_hi")),
    boot.get((1, "mcc", "estimate")),
    boot.get((1, "mcc", "ci_lo")),
    boot.get((1, "mcc", "ci_hi")),
    boot.get((24, "acc", "estimate")),
    boot.get((24, "acc", "ci_lo")),
    boot.get((24, "acc", "ci_hi")),
    boot.get((24, "mcc", "estimate")),
    boot.get((24, "mcc", "ci_lo")),
    boot.get((24, "mcc", "ci_hi")),
]

with open(summary, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)
PY
done
