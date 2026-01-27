from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from width_scaling import apply_width_scaling, fit_s_global, fit_s_per_horizon


def _masked_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray) -> float:
    inside = (y >= lo) & (y <= hi) & (mask > 0)
    denom = np.sum(mask)
    return float(np.sum(inside) / max(denom, 1.0))


def _masked_width(lo: np.ndarray, hi: np.ndarray, mask: np.ndarray) -> float:
    denom = np.sum(mask)
    return float(np.sum((hi - lo) * mask) / max(denom, 1.0))


def _metrics(y: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    diff = (y - q50) * mask
    mae = float(np.sum(np.abs(diff)) / max(np.sum(mask), 1.0))
    rmse = float(np.sqrt(np.sum((diff) ** 2) / max(np.sum(mask), 1.0)))
    return {
        "mae": mae,
        "rmse": rmse,
        "coverage80": _masked_coverage(y, q10, q90, mask),
        "width80": _masked_width(q10, q90, mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_metrics", default=None)
    parser.add_argument("--mode", default="global", choices=["global", "per_horizon"])
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--window", type=int, default=720, help="Rolling window in time steps (hours).")
    parser.add_argument("--min_points", type=int, default=100)
    parser.add_argument("--s_lo", type=float, default=0.1)
    parser.add_argument("--s_hi", type=float, default=1.5)
    parser.add_argument("--iters", type=int, default=40)
    args = parser.parse_args()

    data = np.load(args.preds)
    y = data["y"]
    q10 = data["q10"]
    q50 = data["q50"]
    q90 = data["q90"]
    if "mask" in data:
        mask = data["mask"].astype(np.float32)
    else:
        mask = np.isfinite(y).astype(np.float32)
    if "origin_t" not in data:
        raise ValueError("origin_t not found in preds npz. Re-run eval with --out_npz (save_indices=true).")
    origin_t = data["origin_t"].astype(np.int64)

    order = np.argsort(origin_t)
    y_s = y[order]
    q10_s = q10[order]
    q50_s = q50[order]
    q90_s = q90[order]
    mask_s = mask[order]
    t_s = origin_t[order]

    unique_t, first_idx, counts = np.unique(t_s, return_index=True, return_counts=True)
    s_values = np.zeros((len(unique_t),), dtype=np.float32)
    if args.mode == "per_horizon":
        s_values = np.zeros((len(unique_t), q10.shape[1]), dtype=np.float32)

    start = 0
    fallback = 0
    q10_scaled = q10_s.copy()
    q90_scaled = q90_s.copy()
    for i, t in enumerate(unique_t):
        group_start = first_idx[i]
        group_end = group_start + counts[i]
        while start < group_start and t_s[start] < t - args.window:
            start += 1
        calib_slice = slice(start, group_start)
        n_calib = group_start - start
        if n_calib < args.min_points:
            fallback += 1
            s = 1.0
        else:
            if args.mode == "per_horizon":
                s = fit_s_per_horizon(
                    y_s[calib_slice],
                    q10_s[calib_slice],
                    q50_s[calib_slice],
                    q90_s[calib_slice],
                    target=args.target,
                    s_lo=args.s_lo,
                    s_hi=args.s_hi,
                    iters=args.iters,
                    mask=mask_s[calib_slice],
                )
            else:
                s = fit_s_global(
                    y_s[calib_slice],
                    q10_s[calib_slice],
                    q50_s[calib_slice],
                    q90_s[calib_slice],
                    target=args.target,
                    s_lo=args.s_lo,
                    s_hi=args.s_hi,
                    iters=args.iters,
                    mask=mask_s[calib_slice],
                )
        if args.mode == "per_horizon":
            s_values[i] = s
        else:
            s_values[i] = float(s)
        lo, _, hi = apply_width_scaling(q10_s[group_start:group_end], q50_s[group_start:group_end], q90_s[group_start:group_end], s)
        q10_scaled[group_start:group_end] = lo
        q90_scaled[group_start:group_end] = hi

    # Restore original order
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    q10_out = q10_scaled[inv]
    q90_out = q90_scaled[inv]
    q50_out = q50

    metrics = _metrics(y, q10_out, q50_out, q90_out, mask)
    metrics["fallback_groups"] = float(fallback)
    metrics["total_groups"] = float(len(unique_t))
    print("rolling_width_scaling", metrics)

    np.savez(args.out_npz, y=y, q10=q10_out, q50=q50_out, q90=q90_out, mask=mask, origin_t=origin_t)
    if args.out_metrics:
        np.savez(args.out_metrics, metrics=metrics)


if __name__ == "__main__":
    main()
