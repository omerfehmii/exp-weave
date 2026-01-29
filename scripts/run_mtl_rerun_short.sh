#!/usr/bin/env bash
set -euo pipefail

PY=".venv/bin/python"
HORIZONS="1,2,3,4,5,6,7,24"
SPLIT_PURGE=24
SPLIT_EMBARGO=24
OUT_ROOT="runs/dir_ablations_mtlshort"
SUMMARY="${OUT_ROOT}/summary.csv"

mkdir -p "${OUT_ROOT}"

if [ ! -f "${SUMMARY}" ]; then
  echo "run_name,config,checkpoint_best,checkpoint_last,best_epoch,last_epoch,preds_hash_best,preds_hash_last,best_mae,best_pinball_q10,best_pinball_q50,best_pinball_q90,best_coverage80,best_width80,best_aci_coverage80,best_aci_width80,best_dir_wMCC,best_dir_wAUC,best_dir_acc_mean,best_dir_mcc_mean,best_dir_auc_mean,best_boot_h1_acc,best_boot_h1_mcc,best_boot_h24_acc,best_boot_h24_mcc,last_mae,last_pinball_q10,last_pinball_q50,last_pinball_q90,last_coverage80,last_width80,last_aci_coverage80,last_aci_width80,last_dir_wMCC,last_dir_wAUC,last_dir_acc_mean,last_dir_mcc_mean,last_dir_auc_mean,last_boot_h1_acc,last_boot_h1_mcc,last_boot_h24_acc,last_boot_h24_mcc" > "${SUMMARY}"
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
  "mag|configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_mag_mtlshort.yaml|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_mag_mtlshort_best.pt|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_mag_mtlshort_last.pt"
  "pcgrad_nomag|configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_nomag_mtlshort.yaml|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_nomag_mtlshort_best.pt|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_pcgrad_nomag_mtlshort_last.pt"
  "gradnorm_mag|configs/ablations/yahoo_big_lsq_smin_0_p3_dirhead_mag_gradnorm_mtlshort.yaml|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_mag_gradnorm_mtlshort_best.pt|artifacts/yahoo_big_lsq_smin_0_p3_dirhead_mag_gradnorm_mtlshort_last.pt"
)

for entry in "${RUNS[@]}"; do
  IFS="|" read -r RUN_NAME CFG_PATH CKPT_BEST CKPT_LAST <<< "${entry}"
  OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
  mkdir -p "${OUT_DIR}"

  if has_run "${RUN_NAME}"; then
    echo "=== ${RUN_NAME} (skip: already in summary) ==="
    continue
  fi

  echo "=== ${RUN_NAME} ==="
  echo "train: ${CFG_PATH}"
  ${PY} train.py --config "${CFG_PATH}"

  for tag in best last; do
    if [ "${tag}" = "best" ]; then
      CKPT="${CKPT_BEST}"
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

  RUN_NAME="${RUN_NAME}" CFG_PATH="${CFG_PATH}" CKPT_BEST="${CKPT_BEST}" CKPT_LAST="${CKPT_LAST}" OUT_DIR="${OUT_DIR}" SUMMARY="${SUMMARY}" ${PY} - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

run_name = os.environ["RUN_NAME"]
cfg = os.environ["CFG_PATH"]
ckpt_best = os.environ["CKPT_BEST"]
ckpt_last = os.environ["CKPT_LAST"]
out_dir = Path(os.environ["OUT_DIR"])
summary = Path(os.environ["SUMMARY"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_metrics(path: Path):
    data = np.load(path, allow_pickle=True)
    metrics = data["metrics"]
    if isinstance(metrics, np.ndarray):
        return metrics.item()
    return metrics


def _pinball(y: np.ndarray, q: np.ndarray, alpha: float, mask: np.ndarray) -> float:
    diff = y - q
    loss = np.where(diff >= 0, alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss[mask]))


def _pinball_from_preds(preds_path: Path):
    preds = np.load(preds_path)
    y = preds["y"]
    q10 = preds["q10"]
    q50 = preds["q50"]
    q90 = preds["q90"]
    mask = preds["mask"].astype(np.float32)
    valid = np.isfinite(y) & (mask > 0)
    return {
        "q10": _pinball(y, q10, 0.1, valid),
        "q50": _pinball(y, q50, 0.5, valid),
        "q90": _pinball(y, q90, 0.9, valid),
    }


def _boot_extract(csv_path: Path):
    out = {}
    if not csv_path.exists():
        return out
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = int(row["h"])
            metric = row["metric"]
            if h in (1, 24) and metric in ("acc", "mcc"):
                out[(h, metric)] = row["estimate"]
    return out


def _load_pack(tag: str):
    sub = out_dir / tag
    fore = _load_metrics(sub / "forecast_metrics.npz")
    aci = _load_metrics(sub / "aci_metrics.npz")
    dir_m = _load_metrics(sub / "dir_head_metrics.npz")
    pins = _pinball_from_preds(sub / "preds.npz")
    boot = _boot_extract(sub / "dir_bootstrap_ci.csv")
    return fore, aci, dir_m, pins, boot, _sha256(sub / "preds.npz")


def _epochs_from_log(log_path: Path):
    if not log_path.exists():
        return None, None
    val_records = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") == "val":
                val_records.append(rec)
    if not val_records:
        return None, None
    def _score(r):
        if "dir_loss" in r:
            return r["dir_loss"]
        return r.get("loss")
    best = min(val_records, key=_score)
    best_epoch = best.get("epoch")
    last_epoch = max(r.get("epoch", -1) for r in val_records)
    return best_epoch, last_epoch


log_path = None
with Path(cfg).open() as f:
    for line in f:
        if line.strip().startswith("log_path:"):
            log_path = line.split(":", 1)[1].strip()
            break
best_epoch, last_epoch = _epochs_from_log(Path(log_path)) if log_path else (None, None)

fore_b, aci_b, dir_b, pins_b, boot_b, hash_b = _load_pack("best")
fore_l, aci_l, dir_l, pins_l, boot_l, hash_l = _load_pack("last")

row = [
    run_name,
    cfg,
    ckpt_best,
    ckpt_last,
    best_epoch,
    last_epoch,
    hash_b,
    hash_l,
    fore_b.get("mae"),
    pins_b.get("q10"),
    pins_b.get("q50"),
    pins_b.get("q90"),
    fore_b.get("coverage80"),
    fore_b.get("width80"),
    aci_b.get("coverage80"),
    aci_b.get("width80"),
    dir_b.get("dirscore_wMCC"),
    dir_b.get("dirscore_wAUC"),
    dir_b.get("dir_acc_mean"),
    dir_b.get("dir_mcc_mean"),
    dir_b.get("dir_auc_mean"),
    boot_b.get((1, "acc")),
    boot_b.get((1, "mcc")),
    boot_b.get((24, "acc")),
    boot_b.get((24, "mcc")),
    fore_l.get("mae"),
    pins_l.get("q10"),
    pins_l.get("q50"),
    pins_l.get("q90"),
    fore_l.get("coverage80"),
    fore_l.get("width80"),
    aci_l.get("coverage80"),
    aci_l.get("width80"),
    dir_l.get("dirscore_wMCC"),
    dir_l.get("dirscore_wAUC"),
    dir_l.get("dir_acc_mean"),
    dir_l.get("dir_mcc_mean"),
    dir_l.get("dir_auc_mean"),
    boot_l.get((1, "acc")),
    boot_l.get((1, "mcc")),
    boot_l.get((24, "acc")),
    boot_l.get((24, "mcc")),
]

with summary.open("a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(row)
PY
done
