#!/usr/bin/env bash
set -euo pipefail

PY=".venv/bin/python"
HORIZONS="1,2,3,4,5,6,7,24"
SPLIT_PURGE=24
SPLIT_EMBARGO=24
OUT_ROOT="runs/dir_ablations_seed3"
SUMMARY="${OUT_ROOT}/summary.csv"

mkdir -p "${OUT_ROOT}"

if [ ! -f "${SUMMARY}" ]; then
  echo "run_name,model,seed,config,checkpoint_best_loss,checkpoint_best_dir,checkpoint_last,best_loss_mae,best_loss_pinball,best_loss_coverage80,best_loss_width80,best_loss_aci_coverage80,best_loss_aci_width80,best_loss_dir_wMCC,best_loss_dir_wAUC,best_loss_dir_acc_mean,best_loss_dir_mcc_mean,best_loss_dir_auc_mean,best_loss_boot_h1_acc,best_loss_boot_h1_mcc,best_loss_boot_h24_acc,best_loss_boot_h24_mcc,best_dir_mae,best_dir_pinball,best_dir_coverage80,best_dir_width80,best_dir_aci_coverage80,best_dir_aci_width80,best_dir_dir_wMCC,best_dir_dir_wAUC,best_dir_dir_acc_mean,best_dir_dir_mcc_mean,best_dir_dir_auc_mean,best_dir_boot_h1_acc,best_dir_boot_h1_mcc,best_dir_boot_h24_acc,best_dir_boot_h24_mcc,last_mae,last_pinball,last_coverage80,last_width80,last_aci_coverage80,last_aci_width80,last_dir_wMCC,last_dir_wAUC,last_dir_acc_mean,last_dir_mcc_mean,last_dir_auc_mean,last_boot_h1_acc,last_boot_h1_mcc,last_boot_h24_acc,last_boot_h24_mcc" > "${SUMMARY}"
fi

if command -v rg >/dev/null 2>&1; then
  HAS_RG=1
else
  HAS_RG=0
fi

has_run() {
  local name="$1"
  if [ "${HAS_RG}" -eq 1 ]; then
    rg -q "^${name}," "${SUMMARY}"
  else
    grep -q "^${name}," "${SUMMARY}"
  fi
}

RUNS=(
  "pcgrad_mag_s7|pcgrad_mag|7|configs/ablations/seeds/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_mag_seed7.yaml|artifacts/seed_runs/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_mag_seed7.pt"
  "pcgrad_mag_s13|pcgrad_mag|13|configs/ablations/seeds/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_mag_seed13.yaml|artifacts/seed_runs/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_mag_seed13.pt"
  "pcgrad_mag_s23|pcgrad_mag|23|configs/ablations/seeds/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_mag_seed23.yaml|artifacts/seed_runs/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_mag_seed23.pt"
  "detach_s7|detach|7|configs/ablations/seeds/yahoo_big_lsq_smin_0_p3_dirhead_detach_seed7.yaml|artifacts/seed_runs/yahoo_big_lsq_smin_0_p3_dirhead_detach_seed7.pt"
  "detach_s13|detach|13|configs/ablations/seeds/yahoo_big_lsq_smin_0_p3_dirhead_detach_seed13.yaml|artifacts/seed_runs/yahoo_big_lsq_smin_0_p3_dirhead_detach_seed13.pt"
  "detach_s23|detach|23|configs/ablations/seeds/yahoo_big_lsq_smin_0_p3_dirhead_detach_seed23.yaml|artifacts/seed_runs/yahoo_big_lsq_smin_0_p3_dirhead_detach_seed23.pt"
)

for entry in "${RUNS[@]}"; do
  IFS="|" read -r RUN_NAME MODEL_NAME SEED CFG_PATH CKPT_PATH <<< "${entry}"
  OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
  mkdir -p "${OUT_DIR}"

  if has_run "${RUN_NAME}"; then
    echo "=== ${RUN_NAME} (skip: already in summary) ==="
    continue
  fi

  CKPT_BASE="${CKPT_PATH%.pt}"
  CKPT_BEST_LOSS="${CKPT_BASE}_best_loss.pt"
  CKPT_BEST_DIR="${CKPT_BASE}_best_dir.pt"
  CKPT_LAST="${CKPT_BASE}_last.pt"

  echo "=== ${RUN_NAME} ==="
  echo "train: ${CFG_PATH}"
  ${PY} train.py --config "${CFG_PATH}"

  for tag in best_loss best_dir last; do
    if [ "${tag}" = "best_loss" ]; then
      CKPT="${CKPT_BEST_LOSS}"
    elif [ "${tag}" = "best_dir" ]; then
      CKPT="${CKPT_BEST_DIR}"
    else
      CKPT="${CKPT_LAST}"
    fi
    SUB_DIR="${OUT_DIR}/${tag}"
    mkdir -p "${SUB_DIR}"

    echo "eval (${tag}): forecast"
    ${PY} eval.py \
      --config "${CFG_PATH}" \
      --checkpoint "${CKPT}" \
      --split test \
      --use_cqr false \
      --out_npz "${SUB_DIR}/preds.npz" \
      --metrics_out "${SUB_DIR}/forecast_metrics.npz" \
      --out_csv "${SUB_DIR}/forecast_horizon.csv" \
      --split_purge "${SPLIT_PURGE}" \
      --split_embargo "${SPLIT_EMBARGO}"

    echo "eval (${tag}): aci"
    ${PY} scripts/aci_backtest.py \
      --config "${CFG_PATH}" \
      --preds "${SUB_DIR}/preds.npz" \
      --out_npz "${SUB_DIR}/aci_preds.npz" \
      --out_metrics "${SUB_DIR}/aci_metrics.npz" \
      --coverage_mode per_horizon \
      --coverage_target 0.80

    echo "eval (${tag}): direction head"
    ${PY} scripts/direction_head_eval.py \
      --config "${CFG_PATH}" \
      --checkpoint "${CKPT}" \
      --split test \
      --out_csv "${SUB_DIR}/dir_head.csv" \
      --out_metrics "${SUB_DIR}/dir_head_metrics.npz" \
      --delta_mode origin \
      --split_purge "${SPLIT_PURGE}" \
      --split_embargo "${SPLIT_EMBARGO}" \
      --horizons "${HORIZONS}"

    echo "eval (${tag}): dir bins"
    ${PY} scripts/dir_delta_bins.py \
      --config "${CFG_PATH}" \
      --preds "${SUB_DIR}/preds.npz" \
      --out_csv "${SUB_DIR}/dir_bins.csv" \
      --out_metrics "${SUB_DIR}/dir_bins_metrics.npz" \
      --delta_mode origin \
      --horizons "${HORIZONS}" \
      --split_purge "${SPLIT_PURGE}" \
      --split_embargo "${SPLIT_EMBARGO}"

    echo "eval (${tag}): bootstrap ci"
    ${PY} scripts/dir_bootstrap_ci.py \
      --config "${CFG_PATH}" \
      --preds "${SUB_DIR}/preds.npz" \
      --out_csv "${SUB_DIR}/dir_bootstrap_ci.csv" \
      --delta_mode origin \
      --epsilon_mode quantile \
      --epsilon_q 0.33 \
      --horizons "${HORIZONS}" \
      --bootstrap_runs 200 \
      --block_size 24 \
      --split_purge "${SPLIT_PURGE}" \
      --split_embargo "${SPLIT_EMBARGO}"
  done

  RUN_NAME="${RUN_NAME}" MODEL_NAME="${MODEL_NAME}" SEED="${SEED}" CFG_PATH="${CFG_PATH}" CKPT_BEST_LOSS="${CKPT_BEST_LOSS}" CKPT_BEST_DIR="${CKPT_BEST_DIR}" CKPT_LAST="${CKPT_LAST}" OUT_DIR="${OUT_DIR}" SUMMARY="${SUMMARY}" ${PY} - <<'PY'
import csv
import json
import os
import numpy as np

run_name = os.environ["RUN_NAME"]
model = os.environ["MODEL_NAME"]
seed = os.environ["SEED"]
cfg = os.environ["CFG_PATH"]
ckpt_best_loss = os.environ["CKPT_BEST_LOSS"]
ckpt_best_dir = os.environ["CKPT_BEST_DIR"]
ckpt_last = os.environ["CKPT_LAST"]
out_dir = os.environ["OUT_DIR"]
summary = os.environ["SUMMARY"]


def _load_metrics(path):
    data = np.load(path, allow_pickle=True)
    metrics = data["metrics"]
    if isinstance(metrics, np.ndarray):
        return metrics.item()
    return metrics

def _load_pack(tag):
    sub = os.path.join(out_dir, tag)
    forecast = _load_metrics(os.path.join(sub, "forecast_metrics.npz"))
    aci = _load_metrics(os.path.join(sub, "aci_metrics.npz"))
    dir_m = _load_metrics(os.path.join(sub, "dir_head_metrics.npz"))
    boot_path = os.path.join(sub, "dir_bootstrap_ci.csv")
    boot = {}
    if os.path.exists(boot_path):
        with open(boot_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = int(row["h"])
                metric = row["metric"]
                if h in (1, 24) and metric in ("acc", "mcc"):
                    boot[(h, metric, "estimate")] = row["estimate"]
    return forecast, aci, dir_m, boot

fore_bl, aci_bl, dir_bl, boot_bl = _load_pack("best_loss")
fore_bd, aci_bd, dir_bd, boot_bd = _load_pack("best_dir")
fore_l, aci_l, dir_l, boot_l = _load_pack("last")

row = [
    run_name,
    model,
    seed,
    cfg,
    ckpt_best_loss,
    ckpt_best_dir,
    ckpt_last,
    fore_bl.get("mae"),
    fore_bl.get("pinball"),
    fore_bl.get("coverage80"),
    fore_bl.get("width80"),
    aci_bl.get("coverage80"),
    aci_bl.get("width80"),
    dir_bl.get("dirscore_wMCC"),
    dir_bl.get("dirscore_wAUC"),
    dir_bl.get("dir_acc_mean"),
    dir_bl.get("dir_mcc_mean"),
    dir_bl.get("dir_auc_mean"),
    boot_bl.get((1, "acc", "estimate")),
    boot_bl.get((1, "mcc", "estimate")),
    boot_bl.get((24, "acc", "estimate")),
    boot_bl.get((24, "mcc", "estimate")),
    fore_bd.get("mae"),
    fore_bd.get("pinball"),
    fore_bd.get("coverage80"),
    fore_bd.get("width80"),
    aci_bd.get("coverage80"),
    aci_bd.get("width80"),
    dir_bd.get("dirscore_wMCC"),
    dir_bd.get("dirscore_wAUC"),
    dir_bd.get("dir_acc_mean"),
    dir_bd.get("dir_mcc_mean"),
    dir_bd.get("dir_auc_mean"),
    boot_bd.get((1, "acc", "estimate")),
    boot_bd.get((1, "mcc", "estimate")),
    boot_bd.get((24, "acc", "estimate")),
    boot_bd.get((24, "mcc", "estimate")),
    fore_l.get("mae"),
    fore_l.get("pinball"),
    fore_l.get("coverage80"),
    fore_l.get("width80"),
    aci_l.get("coverage80"),
    aci_l.get("width80"),
    dir_l.get("dirscore_wMCC"),
    dir_l.get("dirscore_wAUC"),
    dir_l.get("dir_acc_mean"),
    dir_l.get("dir_mcc_mean"),
    dir_l.get("dir_auc_mean"),
    boot_l.get((1, "acc", "estimate")),
    boot_l.get((1, "mcc", "estimate")),
    boot_l.get((24, "acc", "estimate")),
    boot_l.get((24, "mcc", "estimate")),
]

with open(summary, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)

eval_summary = {
    "run_name": run_name,
    "model": model,
    "seed": seed,
    "entries": [
        {
            "tag": "best_loss",
            "checkpoint": ckpt_best_loss,
            "metrics": {
                "mae": fore_bl.get("mae"),
                "pinball": fore_bl.get("pinball"),
                "coverage80": fore_bl.get("coverage80"),
                "width80": fore_bl.get("width80"),
                "aci_coverage80": aci_bl.get("coverage80"),
                "aci_width80": aci_bl.get("width80"),
                "dir_wMCC": dir_bl.get("dirscore_wMCC"),
                "dir_wAUC": dir_bl.get("dirscore_wAUC"),
                "dir_acc_mean": dir_bl.get("dir_acc_mean"),
                "dir_mcc_mean": dir_bl.get("dir_mcc_mean"),
                "dir_auc_mean": dir_bl.get("dir_auc_mean"),
                "boot_h1_acc": boot_bl.get((1, "acc", "estimate")),
                "boot_h1_mcc": boot_bl.get((1, "mcc", "estimate")),
                "boot_h24_acc": boot_bl.get((24, "acc", "estimate")),
                "boot_h24_mcc": boot_bl.get((24, "mcc", "estimate")),
            },
        },
        {
            "tag": "best_dir",
            "checkpoint": ckpt_best_dir,
            "metrics": {
                "mae": fore_bd.get("mae"),
                "pinball": fore_bd.get("pinball"),
                "coverage80": fore_bd.get("coverage80"),
                "width80": fore_bd.get("width80"),
                "aci_coverage80": aci_bd.get("coverage80"),
                "aci_width80": aci_bd.get("width80"),
                "dir_wMCC": dir_bd.get("dirscore_wMCC"),
                "dir_wAUC": dir_bd.get("dirscore_wAUC"),
                "dir_acc_mean": dir_bd.get("dir_acc_mean"),
                "dir_mcc_mean": dir_bd.get("dir_mcc_mean"),
                "dir_auc_mean": dir_bd.get("dir_auc_mean"),
                "boot_h1_acc": boot_bd.get((1, "acc", "estimate")),
                "boot_h1_mcc": boot_bd.get((1, "mcc", "estimate")),
                "boot_h24_acc": boot_bd.get((24, "acc", "estimate")),
                "boot_h24_mcc": boot_bd.get((24, "mcc", "estimate")),
            },
        },
        {
            "tag": "last",
            "checkpoint": ckpt_last,
            "metrics": {
                "mae": fore_l.get("mae"),
                "pinball": fore_l.get("pinball"),
                "coverage80": fore_l.get("coverage80"),
                "width80": fore_l.get("width80"),
                "aci_coverage80": aci_l.get("coverage80"),
                "aci_width80": aci_l.get("width80"),
                "dir_wMCC": dir_l.get("dirscore_wMCC"),
                "dir_wAUC": dir_l.get("dirscore_wAUC"),
                "dir_acc_mean": dir_l.get("dir_acc_mean"),
                "dir_mcc_mean": dir_l.get("dir_mcc_mean"),
                "dir_auc_mean": dir_l.get("dir_auc_mean"),
                "boot_h1_acc": boot_l.get((1, "acc", "estimate")),
                "boot_h1_mcc": boot_l.get((1, "mcc", "estimate")),
                "boot_h24_acc": boot_l.get((24, "acc", "estimate")),
                "boot_h24_mcc": boot_l.get((24, "mcc", "estimate")),
            },
        },
    ],
}

with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
    json.dump(eval_summary, f, indent=2, sort_keys=True)
PY

done
