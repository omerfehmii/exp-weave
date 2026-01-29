from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.loader import load_panel_npz
from data.missingness import MissingnessConfig, apply_missingness_window
from data.features import compute_delta_t
from train import filter_indices_with_observed
from utils import load_config


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
    # handle ties
    unique, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            mask = inv == idx
            ranks[mask] = ranks[mask].mean()
    sum_ranks_pos = np.sum(ranks[y_true == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--missingness_pattern", default="none", choices=["none", "random", "burst"])
    parser.add_argument("--missingness_rate", type=float, default=0.0)
    parser.add_argument("--missingness_burst_prob", type=float, default=0.0)
    parser.add_argument("--missingness_burst_len", type=int, default=24)
    parser.add_argument("--missingness_seed", type=int, default=7)
    parser.add_argument("--split_purge", type=int, default=None)
    parser.add_argument("--split_embargo", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    series_list = load_panel_npz(cfg["data"]["path"])
    for s in series_list:
        s.ensure_features()
    lengths = [len(s.y) for s in series_list]
    indices = generate_panel_origins(lengths, cfg["data"]["L"], cfg["data"]["H"], step=cfg["data"].get("step", cfg["data"]["H"]))
    split = make_time_splits(min(lengths), cfg["data"].get("train_frac", 0.7), cfg["data"].get("val_frac", 0.15))
    horizon = cfg["data"]["H"]
    split_purge = int(cfg["data"].get("split_purge", 0)) if args.split_purge is None else int(args.split_purge)
    split_embargo = int(cfg["data"].get("split_embargo", 0)) if args.split_embargo is None else int(args.split_embargo)
    idx = select_indices_by_time(
        indices,
        split,
        args.split,
        horizon=horizon,
        purge=split_purge,
        embargo=split_embargo,
    )
    idx = filter_indices_with_observed(series_list, idx, cfg["data"]["L"], cfg["data"]["H"], cfg["data"].get("min_past_obs", 1), cfg["data"].get("min_future_obs", 1))

    preds = np.load(args.preds)
    q10 = preds["q10"]
    q90 = preds["q90"]
    if q10.shape[0] != len(idx):
        raise ValueError("Preds size does not match split indices. Regenerate preds with matching split.")

    w = q90 - q10
    zero = w <= args.eps
    H = w.shape[1]

    use_missingness = args.missingness_pattern != "none"
    miss_cfg = MissingnessConfig(
        pattern=args.missingness_pattern,
        rate=args.missingness_rate,
        burst_prob=args.missingness_burst_prob,
        burst_len=args.missingness_burst_len,
        seed=args.missingness_seed,
    )

    mask_ratio = np.zeros((len(idx),), dtype=np.float32)
    delta_mean = np.zeros((len(idx),), dtype=np.float32)
    delta_max = np.zeros((len(idx),), dtype=np.float32)

    for i, (s_idx, t) in enumerate(idx):
        s = series_list[s_idx]
        y = s.y
        if y.ndim == 1:
            y = y[:, None]
        mask = s.mask
        if mask is None:
            mask = np.ones_like(y)
        if mask.ndim == 1:
            mask = mask[:, None]
        delta = s.delta_t
        if delta is None:
            delta = np.zeros_like(y)
        if delta.ndim == 1:
            delta = delta[:, None]
        past = slice(t - cfg["data"]["L"] + 1, t + 1)
        mask_past = mask[past]
        if use_missingness:
            y_past = y[past]
            _, mask_past = apply_missingness_window(y_past, mask_past, miss_cfg, i)
            delta_past = compute_delta_t(mask_past)
        else:
            delta_past = delta[past]
        mask_ratio[i] = float(np.mean(mask_past))
        delta_mean[i] = float(np.mean(delta_past))
        delta_max[i] = float(np.max(delta_past))

    rows = []
    for h in range(H):
        w_h = w[:, h]
        zero_h = zero[:, h].astype(int)
        rows.append(
            [
                h + 1,
                _corr(w_h, mask_ratio),
                _corr(w_h, delta_mean),
                _corr(w_h, delta_max),
                _roc_auc(zero_h, mask_ratio),
                _roc_auc(zero_h, delta_mean),
                _roc_auc(zero_h, delta_max),
                float(np.mean(zero_h)),
            ]
        )

    import csv

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "h",
                "corr_w_mask_ratio",
                "corr_w_delta_mean",
                "corr_w_delta_max",
                "auc_zero_mask_ratio",
                "auc_zero_delta_mean",
                "auc_zero_delta_max",
                "zero_rate",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
