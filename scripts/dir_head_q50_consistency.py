from __future__ import annotations

import argparse
import csv
from typing import List

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from train import default_horizon_weights
from scripts.dir_head_utils import load_dir_head_arrays


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


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
    args = parser.parse_args()

    data = load_dir_head_arrays(
        args.config,
        args.checkpoint,
        args.split,
        args.split_purge,
        args.split_embargo,
        delta_mode=args.delta_mode,
    )
    delta = data["delta"]
    mask = data["mask"]
    dir_prob = data["dir_prob"]
    q50 = data["q50"]
    y_last = data["y_last"]
    H = data["H"]

    dir_label = (delta > 0).astype(np.int32)

    if args.horizons.strip():
        h_list = [h for h in _parse_ints(args.horizons) if 1 <= h <= H]
    else:
        h_list = list(range(1, H + 1))
    h_idx = np.asarray([h - 1 for h in h_list], dtype=np.int64)
    weights = default_horizon_weights(H)

    out_csv = _Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "h",
                "n",
                "agree_rate",
                "head_acc",
                "q50_acc",
                "head_mcc",
                "q50_mcc",
                "head_better_rate",
                "q50_better_rate",
                "both_correct_rate",
            ]
        )
        head_mcc_h = np.full(H, np.nan, dtype=np.float32)
        q50_mcc_h = np.full(H, np.nan, dtype=np.float32)
        agree_h = np.full(H, np.nan, dtype=np.float32)
        head_acc_h = np.full(H, np.nan, dtype=np.float32)
        q50_acc_h = np.full(H, np.nan, dtype=np.float32)
        for h in h_idx:
            m = mask[:, h]
            if not np.any(m):
                writer.writerow([h + 1, 0, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan"])
                continue
            y = dir_label[m, h]
            head_pred = (dir_prob[m, h] >= 0.5).astype(int)
            q50_delta = q50[m, h] - y_last[m]
            q50_pred = (q50_delta >= 0).astype(int)

            agree = head_pred == q50_pred
            agree_rate = float(np.mean(agree))
            head_acc = float(np.mean(head_pred == y))
            q50_acc = float(np.mean(q50_pred == y))
            head_mcc = _mcc(y, head_pred)
            q50_mcc = _mcc(y, q50_pred)
            head_better = float(np.mean((head_pred == y) & (q50_pred != y)))
            q50_better = float(np.mean((q50_pred == y) & (head_pred != y)))
            both_correct = float(np.mean((head_pred == y) & (q50_pred == y)))

            agree_h[h] = agree_rate
            head_acc_h[h] = head_acc
            q50_acc_h[h] = q50_acc
            head_mcc_h[h] = head_mcc
            q50_mcc_h[h] = q50_mcc

            writer.writerow(
                [
                    h + 1,
                    int(np.sum(m)),
                    agree_rate,
                    head_acc,
                    q50_acc,
                    head_mcc,
                    q50_mcc,
                    head_better,
                    q50_better,
                    both_correct,
                ]
            )

    # Weighted summary printed to stdout for convenience
    valid = ~np.isnan(head_mcc_h[h_idx])
    if np.any(valid):
        w = weights[h_idx][valid]
        head_mcc_w = float(np.sum(head_mcc_h[h_idx][valid] * w) / max(np.sum(w), 1e-6))
        q50_mcc_w = float(np.sum(q50_mcc_h[h_idx][valid] * w) / max(np.sum(w), 1e-6))
        agree_w = float(np.sum(agree_h[h_idx][valid] * w) / max(np.sum(w), 1e-6))
        head_acc_w = float(np.sum(head_acc_h[h_idx][valid] * w) / max(np.sum(w), 1e-6))
        q50_acc_w = float(np.sum(q50_acc_h[h_idx][valid] * w) / max(np.sum(w), 1e-6))
        print(
            {
                "head_mcc_w": head_mcc_w,
                "q50_mcc_w": q50_mcc_w,
                "agree_w": agree_w,
                "head_acc_w": head_acc_w,
                "q50_acc_w": q50_acc_w,
            }
        )


if __name__ == "__main__":
    main()
