#!/usr/bin/env bash
set -euo pipefail

if [ -z "${PY:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python -u"
  else
    PY="python -u"
  fi
fi
export PYTHONUNBUFFERED=1
HORIZONS="1,2,3,4,5,6,7,24"
SPLIT_PURGE=24
SPLIT_EMBARGO=24
OUT_ROOT="runs/pcgrad_mag_sweep"
SUMMARY="${OUT_ROOT}/summary_sweep.csv"
BASE_CFG="configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_mag_pcgrad.yaml"
CFG_DIR="configs/ablations/sweep"
if [ -z "${DRIVE_ROOT:-}" ] && [ -d "/content/drive/MyDrive" ]; then
  DRIVE_ROOT="/content/drive/MyDrive/ag"
fi

mkdir -p "${OUT_ROOT}"
mkdir -p "${CFG_DIR}"
mkdir -p artifacts/sweep

if [ ! -f "${SUMMARY}" ]; then
  echo "run_name,seed,lambda_dir,mag_weight_power,checkpoint_tag,checkpoint,config,mae,pinball,coverage80,width80,aci_coverage80,aci_width80,dir_wMCC,dir_wAUC,dir_acc_mean,dir_ece_mean,dir_mcc_mean,dir_auc_mean,bin_max_acc,bin_max_mcc,bin_max_auc,boot_h1_acc,boot_h1_mcc,boot_h24_acc,boot_h24_mcc" > "${SUMMARY}"
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

copy_to_drive() {
  if [ -z "${DRIVE_ROOT:-}" ] || [ ! -d "${DRIVE_ROOT}" ]; then
    return 0
  fi
  local drive_out="${DRIVE_ROOT}/runs/pcgrad_mag_sweep"
  local drive_art="${DRIVE_ROOT}/artifacts/sweep"
  local drive_cfg="${DRIVE_ROOT}/configs/ablations/sweep"
  local train_log="artifacts/sweep/${RUN_NAME}_train_log.jsonl"
  mkdir -p "${drive_out}/${RUN_NAME}" "${drive_art}" "${drive_cfg}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${RUN_OUT}/" "${drive_out}/${RUN_NAME}/"
    rsync -a "${SUMMARY}" "${drive_out}/"
    rsync -a "${CFG_PATH}" "${drive_cfg}/"
    rsync -a "${CKPT_BEST_LOSS}" "${CKPT_BEST_DIR}" "${CKPT_LAST}" "${drive_art}/"
    if [ -f "${train_log}" ]; then
      rsync -a "${train_log}" "${drive_art}/"
    fi
  else
    cp -R "${RUN_OUT}/." "${drive_out}/${RUN_NAME}/"
    cp -f "${SUMMARY}" "${drive_out}/"
    cp -f "${CFG_PATH}" "${drive_cfg}/"
    cp -f "${CKPT_BEST_LOSS}" "${CKPT_BEST_DIR}" "${CKPT_LAST}" "${drive_art}/"
    if [ -f "${train_log}" ]; then
      cp -f "${train_log}" "${drive_art}/"
    fi
  fi
}

IFS=',' read -r -a SEEDS <<< "${SWEEP_SEEDS:-7}"
LAMBDA_DIRS=(0.03 0.1 0.3)
MAG_POWERS=(0.5 1.0)

for seed in "${SEEDS[@]}"; do
  for lambda_dir in "${LAMBDA_DIRS[@]}"; do
    for mag_power in "${MAG_POWERS[@]}"; do
      RUN_NAME="pcgrad_mag_ld${lambda_dir}_mp${mag_power}_s${seed}"
      if has_run "${RUN_NAME}"; then
        echo "=== ${RUN_NAME} (skip: already in summary) ==="
        continue
      fi

      CFG_PATH="${CFG_DIR}/${RUN_NAME}.yaml"
      RUN_OUT="${OUT_ROOT}/${RUN_NAME}"
      mkdir -p "${RUN_OUT}"

      RUN_NAME="${RUN_NAME}" CFG_PATH="${CFG_PATH}" BASE_CFG="${BASE_CFG}" SEED="${seed}" LAMBDA_DIR="${lambda_dir}" MAG_POWER="${mag_power}" ${PY} - <<'PY'
import os
from pathlib import Path
import yaml

base_cfg = Path(os.environ["BASE_CFG"])
cfg_path = Path(os.environ["CFG_PATH"])
seed = int(os.environ["SEED"])
lambda_dir = float(os.environ["LAMBDA_DIR"])
mag_power = float(os.environ["MAG_POWER"])
run_name = os.environ["RUN_NAME"]

with base_cfg.open() as f:
    cfg = yaml.safe_load(f)

training = cfg.setdefault("training", {})
training["seed"] = seed
training["epochs"] = 8
training.setdefault("early_stopping", {})
training["early_stopping"]["metric"] = "direction_mcc"
training["early_stopping"]["patience"] = 5

training["output_path"] = f"artifacts/sweep/{run_name}.pt"
training["output_path_best_loss"] = f"artifacts/sweep/{run_name}_best_loss.pt"
training["output_path_best_dir"] = f"artifacts/sweep/{run_name}_best_dir.pt"
training["output_path_last"] = f"artifacts/sweep/{run_name}_last.pt"
training["save_last"] = True
training["log_path"] = f"artifacts/sweep/{run_name}_train_log.jsonl"

direction = training.setdefault("direction", {})
direction["loss_weight"] = lambda_dir
direction["mag_weighting"] = True
direction["mag_weight_power"] = mag_power

cfg_path.parent.mkdir(parents=True, exist_ok=True)
with cfg_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

      echo "=== ${RUN_NAME} ==="
      echo "train: ${CFG_PATH}"
      ${PY} train.py --config "${CFG_PATH}"

      CKPT_BEST_LOSS="artifacts/sweep/${RUN_NAME}_best_loss.pt"
      CKPT_BEST_DIR="artifacts/sweep/${RUN_NAME}_best_dir.pt"
      CKPT_LAST="artifacts/sweep/${RUN_NAME}_last.pt"

      for tag in best_loss best_dir last; do
        if [ "${tag}" = "best_loss" ]; then
          CKPT="${CKPT_BEST_LOSS}"
        elif [ "${tag}" = "best_dir" ]; then
          CKPT="${CKPT_BEST_DIR}"
        else
          CKPT="${CKPT_LAST}"
        fi
        SUB_DIR="${RUN_OUT}/${tag}"
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

      RUN_NAME="${RUN_NAME}" SEED="${seed}" LAMBDA_DIR="${lambda_dir}" MAG_POWER="${mag_power}" CFG_PATH="${CFG_PATH}" RUN_OUT="${RUN_OUT}" CKPT_BEST_LOSS="${CKPT_BEST_LOSS}" CKPT_BEST_DIR="${CKPT_BEST_DIR}" CKPT_LAST="${CKPT_LAST}" SUMMARY="${SUMMARY}" ${PY} - <<'PY'
import csv
import os
import numpy as np

run_name = os.environ["RUN_NAME"]
seed = os.environ["SEED"]
lambda_dir = os.environ["LAMBDA_DIR"]
mag_power = os.environ["MAG_POWER"]
cfg = os.environ["CFG_PATH"]
run_out = os.environ["RUN_OUT"]
summary = os.environ["SUMMARY"]
ckpt_best_loss = os.environ["CKPT_BEST_LOSS"]
ckpt_best_dir = os.environ["CKPT_BEST_DIR"]
ckpt_last = os.environ["CKPT_LAST"]

ckpt_map = {
    "best_loss": ckpt_best_loss,
    "best_dir": ckpt_best_dir,
    "last": ckpt_last,
}

def _load_metrics(path):
    data = np.load(path, allow_pickle=True)
    metrics = data["metrics"]
    if isinstance(metrics, np.ndarray):
        return metrics.item()
    return metrics


def _bin_metrics(path):
    data = np.load(path, allow_pickle=True)
    metrics = data["metrics"]
    if isinstance(metrics, np.ndarray):
        metrics = metrics.item()
    bin_summary = metrics.get("bin_summary", {})
    if not bin_summary:
        return None, None, None
    def _hi(k):
        try:
            return float(k.split("-")[1])
        except Exception:
            return -1.0
    key = sorted(bin_summary.keys(), key=_hi)[-1]
    m = bin_summary[key]
    return m.get("acc"), m.get("mcc"), m.get("auc")


def _boot_metrics(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = int(row["h"])
            metric = row["metric"]
            if h in (1, 24) and metric in ("acc", "mcc"):
                out[(h, metric)] = row["estimate"]
    return out

rows = []
for tag in ("best_loss", "best_dir", "last"):
    sub = os.path.join(run_out, tag)
    fore = _load_metrics(os.path.join(sub, "forecast_metrics.npz"))
    aci = _load_metrics(os.path.join(sub, "aci_metrics.npz"))
    dir_m = _load_metrics(os.path.join(sub, "dir_head_metrics.npz"))
    bin_acc, bin_mcc, bin_auc = _bin_metrics(os.path.join(sub, "dir_bins_metrics.npz"))
    boot = _boot_metrics(os.path.join(sub, "dir_bootstrap_ci.csv"))
    rows.append([
        run_name,
        seed,
        lambda_dir,
        mag_power,
        tag,
        ckpt_map[tag],
        cfg,
        fore.get("mae"),
        fore.get("pinball"),
        fore.get("coverage80"),
        fore.get("width80"),
        aci.get("coverage80"),
        aci.get("width80"),
        dir_m.get("dirscore_wMCC"),
        dir_m.get("dirscore_wAUC"),
        dir_m.get("dir_acc_mean"),
        dir_m.get("dir_ece_mean"),
        dir_m.get("dir_mcc_mean"),
        dir_m.get("dir_auc_mean"),
        bin_acc,
        bin_mcc,
        bin_auc,
        boot.get((1, "acc")),
        boot.get((1, "mcc")),
        boot.get((24, "acc")),
        boot.get((24, "mcc")),
    ])

with open(summary, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
PY
      copy_to_drive
    done
  done

done
