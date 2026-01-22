from __future__ import annotations

import argparse
import csv
from typing import List

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from train import compute_delta_thresholds, default_horizon_weights
from scripts.dir_head_utils import get_train_indices, load_dir_head_arrays


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
    parser.add_argument("--epsilon_qs", default="0.1,0.2,0.33,0.4,0.5")
    args = parser.parse_args()

    data = load_dir_head_arrays(
        args.config,
        args.checkpoint,
        args.split,
        args.split_purge,
        args.split_embargo,
        delta_mode=args.delta_mode,
    )
    cfg = data["cfg"]
    series_list = data["series_list"]
    split = data["split"]
    indices = data["indices"]

    delta = data["delta"]
    mask = data["mask"]
    dir_prob = data["dir_prob"]
    move_prob = data["move_prob"]
    H = data["H"]

    dir_label = (delta > 0).astype(np.int32)

    if args.horizons.strip():
        h_list = [h for h in _parse_ints(args.horizons) if 1 <= h <= H]
    else:
        h_list = list(range(1, H + 1))
    h_idx = np.asarray([h - 1 for h in h_list], dtype=np.int64)

    train_idx = get_train_indices(cfg, series_list, split, indices, args.split_purge, args.split_embargo)
    weights = default_horizon_weights(H)

    out_csv = _Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epsilon_q",
                "move_rate",
                "move_wMCC",
                "move_wAUC",
                "dir_wMCC",
                "dir_wAUC",
                "dir_acc_mean",
                "coverage",
                "n_dir",
            ]
        )
        for q in _parse_floats(args.epsilon_qs):
            tau_h = compute_delta_thresholds(series_list, train_idx, H, q, delta_mode=args.delta_mode)
            move_label = (np.abs(delta) >= tau_h[None, :]).astype(np.int32)
            dir_mask = mask & (move_label > 0)
            total_dir = int(dir_mask[:, h_idx].sum())
            coverage = total_dir / int(mask[:, h_idx].sum()) if mask[:, h_idx].sum() > 0 else 0.0

            move_mcc_h = np.full(H, np.nan, dtype=np.float32)
            move_auc_h = np.full(H, np.nan, dtype=np.float32)
            dir_mcc_h = np.full(H, np.nan, dtype=np.float32)
            dir_auc_h = np.full(H, np.nan, dtype=np.float32)
            dir_acc_h = np.full(H, np.nan, dtype=np.float32)

            for h in h_idx:
                m = mask[:, h]
                if np.any(m):
                    y = move_label[m, h]
                    p = move_prob[m, h]
                    pred = (p >= 0.5).astype(int)
                    move_mcc_h[h] = _mcc(y, pred)
                    move_auc_h[h] = _roc_auc(y, p)
                dm = dir_mask[:, h]
                if np.any(dm):
                    y = dir_label[dm, h]
                    p = dir_prob[dm, h]
                    pred = (p >= 0.5).astype(int)
                    dir_mcc_h[h] = _mcc(y, pred)
                    dir_auc_h[h] = _roc_auc(y, p)
                    dir_acc_h[h] = float(np.mean(pred == y))

            valid_move = ~np.isnan(move_mcc_h[h_idx])
            valid_dir = ~np.isnan(dir_mcc_h[h_idx])
            move_wmcc = float(np.sum(move_mcc_h[h_idx][valid_move] * weights[h_idx][valid_move]) / max(np.sum(weights[h_idx][valid_move]), 1e-6))
            move_wauc = float(np.sum(move_auc_h[h_idx][valid_move] * weights[h_idx][valid_move]) / max(np.sum(weights[h_idx][valid_move]), 1e-6))
            dir_wmcc = float(np.sum(dir_mcc_h[h_idx][valid_dir] * weights[h_idx][valid_dir]) / max(np.sum(weights[h_idx][valid_dir]), 1e-6))
            dir_wauc = float(np.sum(dir_auc_h[h_idx][valid_dir] * weights[h_idx][valid_dir]) / max(np.sum(weights[h_idx][valid_dir]), 1e-6))
            dir_acc = float(np.nanmean(dir_acc_h[h_idx]))

            move_rate = float(np.mean(move_label[:, h_idx][mask[:, h_idx]])) if mask[:, h_idx].any() else 0.0

            writer.writerow([q, move_rate, move_wmcc, move_wauc, dir_wmcc, dir_wauc, dir_acc, coverage, total_dir])


if __name__ == "__main__":
    main()
