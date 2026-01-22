from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from train import default_horizon_weights
from scripts.dir_head_utils import load_dir_head_arrays


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    scores = scores.astype(float)
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores), dtype=float) + 1
    unique, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inv == idx
            ranks[mask] = ranks[mask].mean()
    sum_ranks_pos = np.sum(ranks[y_true == 1])
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return float("nan")
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--delta_mode", default="origin", choices=["origin", "step"])
    parser.add_argument("--split_purge", type=int, default=24)
    parser.add_argument("--split_embargo", type=int, default=24)
    parser.add_argument("--horizons", default="1,2,3,4,5,6,7,24")
    parser.add_argument("--tau_values", default="")
    parser.add_argument("--tau_min", type=float, default=0.0)
    parser.add_argument("--tau_max", type=float, default=0.45)
    parser.add_argument("--tau_step", type=float, default=0.05)
    args = parser.parse_args()

    data = load_dir_head_arrays(
        args.config,
        args.checkpoint,
        args.split,
        args.split_purge,
        args.split_embargo,
        delta_mode=args.delta_mode,
    )
    dir_prob = data["dir_prob"]
    delta = data["delta"]
    mask = data["mask"]
    H = data["H"]

    if args.delta_mode == "origin":
        ref = data["y_last"][:, None]
    else:
        ref = np.concatenate([data["y_last"][:, None], data["y_future"][:, :-1]], axis=1)
    dir_label = (delta > 0).astype(np.int32)

    if args.horizons.strip():
        h_list = [h for h in _parse_ints(args.horizons) if 1 <= h <= H]
    else:
        h_list = list(range(1, H + 1))
    h_idx = np.asarray([h - 1 for h in h_list], dtype=np.int64)

    if args.tau_values.strip():
        tau_values = _parse_floats(args.tau_values)
    else:
        tau_values = []
        tau = args.tau_min
        while tau <= args.tau_max + 1e-9:
            tau_values.append(round(tau, 6))
            tau += args.tau_step

    weights = default_horizon_weights(H)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tau",
                "coverage",
                "n_selected",
                "n_total",
                "dir_wMCC",
                "dir_wAUC",
                "dir_mcc_mean",
                "dir_auc_mean",
                "dir_acc",
            ]
        )
        total_mask = mask[:, h_idx]
        total_n = int(total_mask.sum())
        for tau in tau_values:
            conf = np.abs(dir_prob - 0.5)
            sel = (conf[:, h_idx] >= tau) & total_mask
            n_sel = int(sel.sum())
            coverage = n_sel / total_n if total_n else 0.0

            mcc_h = np.full(H, np.nan, dtype=np.float32)
            auc_h = np.full(H, np.nan, dtype=np.float32)
            acc_h = np.full(H, np.nan, dtype=np.float32)
            for pos, h in enumerate(h_idx):
                m = sel[:, pos]
                if not np.any(m):
                    continue
                y = dir_label[m, h]
                p = dir_prob[m, h]
                pred = (p >= 0.5).astype(int)
                mcc_h[h] = _mcc(y, pred)
                auc_h[h] = _roc_auc(y, p)
                acc_h[h] = float(np.mean(pred == y))

            valid = ~np.isnan(mcc_h[h_idx])
            w_mcc = float(np.sum(mcc_h[h_idx][valid] * weights[h_idx][valid]) / max(np.sum(weights[h_idx][valid]), 1e-6))
            w_auc = float(np.sum(auc_h[h_idx][valid] * weights[h_idx][valid]) / max(np.sum(weights[h_idx][valid]), 1e-6))
            mcc_mean = float(np.nanmean(mcc_h[h_idx]))
            auc_mean = float(np.nanmean(auc_h[h_idx]))
            acc_mean = float(np.nanmean(acc_h[h_idx]))

            writer.writerow([tau, coverage, n_sel, total_n, w_mcc, w_auc, mcc_mean, auc_mean, acc_mean])


if __name__ == "__main__":
    main()
