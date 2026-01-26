from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import load_panel_npz, compress_series_observed
from eval import apply_scaling
from utils import load_config


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _safe_cdf_from_quantiles(
    x: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    min_width: float,
    slope_cap: float,
    z_clip: float,
) -> np.ndarray:
    q10 = np.minimum(q10, q50)
    q90 = np.maximum(q90, q50)
    w1 = np.maximum(q50 - q10, min_width)
    w2 = np.maximum(q90 - q50, min_width)
    z10 = np.log(0.1 / 0.9)
    z50 = 0.0
    z90 = np.log(0.9 / 0.1)
    slope1 = np.clip((z50 - z10) / w1, -slope_cap, slope_cap)
    slope2 = np.clip((z90 - z50) / w2, -slope_cap, slope_cap)

    z = np.empty_like(q50, dtype=np.float32)
    left = x <= q10
    mid1 = (x > q10) & (x <= q50)
    mid2 = (x > q50) & (x <= q90)
    right = x > q90

    z[left] = z10 + slope1[left] * (x[left] - q10[left])
    z[mid1] = z10 + slope1[mid1] * (x[mid1] - q10[mid1])
    z[mid2] = z50 + slope2[mid2] * (x[mid2] - q50[mid2])
    z[right] = z90 + slope2[right] * (x[right] - q90[right])
    z = np.clip(z, -z_clip, z_clip)
    return 1.0 / (1.0 + np.exp(-z))


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--temps", default=None)
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    parser.add_argument("--h_filter", type=int, default=4)
    parser.add_argument("--min_width", type=float, default=1e-6)
    parser.add_argument("--slope_cap", type=float, default=10.0)
    parser.add_argument("--z_clip", type=float, default=20.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    preds = np.load(args.preds)
    for key in ("y", "q10", "q50", "q90", "mask", "origin_t", "series_idx"):
        if key not in preds:
            raise ValueError(f"preds npz missing required key: {key}")
    y = preds["y"]
    q10 = preds["q10"]
    q50 = preds["q50"]
    q90 = preds["q90"]
    mask = preds["mask"].astype(np.float32)
    origin_t = preds["origin_t"].astype(np.int64)
    series_idx = preds["series_idx"].astype(np.int64)

    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    lengths = [len(s.y) for s in series_list]
    split = make_time_splits(
        min(lengths),
        cfg["data"].get("train_frac", 0.7),
        cfg["data"].get("val_frac", 0.15),
    )
    pre = apply_scaling(
        series_list,
        split.train_end,
        scale_x=cfg["data"].get("scale_x", True),
        scale_y=cfg["data"].get("scale_y", True),
    )

    y_t_scaled = np.array([series_list[s_idx].y[t] for s_idx, t in zip(series_idx, origin_t)])
    if y_t_scaled.ndim == 2 and y_t_scaled.shape[1] > 1:
        y_t_scaled = y_t_scaled[:, 0]
    ref = y_t_scaled[:, None]
    ref_full = np.repeat(ref, y.shape[1], axis=1)
    cdf = _safe_cdf_from_quantiles(ref_full, q10, q50, q90, args.min_width, args.slope_cap, args.z_clip)
    p_up = 1.0 - cdf

    h_idx = args.h_filter - 1
    if h_idx < 0 or h_idx >= y.shape[1]:
        raise ValueError("h_filter out of range.")

    if args.temps:
        temps = np.load(args.temps)["temps"]
        t = float(temps[h_idx]) if h_idx < temps.shape[0] else 1.0
        logits = _logit(p_up[:, h_idx])
        p_filter = 1.0 / (1.0 + np.exp(-(logits / t)))
    else:
        p_filter = p_up[:, h_idx]

    # labels for 1H and 4H (origin delta in scaled space)
    label_1h = (y[:, 0] - y_t_scaled > 0).astype(np.int32)
    label_4h = (y[:, h_idx] - y_t_scaled > 0).astype(np.int32)
    pred_1h = (q50[:, 0] - y_t_scaled > 0).astype(np.int32)

    valid = (mask[:, 0] > 0) & (mask[:, h_idx] > 0) & np.isfinite(y_t_scaled)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    rows = []
    base_pred1_acc = _acc(label_1h[valid], pred_1h[valid])
    base_label1_rate = float(np.mean(label_1h[valid])) if np.any(valid) else float("nan")
    for t in thresholds:
        accept_up = p_filter >= t
        accept_down = p_filter <= (1.0 - t)
        accept = (accept_up | accept_down) & valid
        if not np.any(accept):
            rows.append([t, 0.0, "nan", "nan", base_pred1_acc, "nan", "nan", 0.0, base_label1_rate])
            continue
        filt_dir = np.where(accept_up, 1, 0)
        filt_dir = filt_dir[accept]
        lab1 = label_1h[accept]
        lab4 = label_4h[accept]
        pred1 = pred_1h[accept]
        cov = float(np.mean(accept))
        acc_1h = _acc(lab1, filt_dir)
        acc_4h = _acc(lab4, filt_dir)
        pred1_acc = _acc(lab1, pred1)
        agree = filt_dir == pred1
        agree_rate = float(np.mean(agree)) if agree.size else float("nan")
        pred1_acc_agree = _acc(lab1[agree], pred1[agree]) if np.any(agree) else float("nan")
        rows.append(
            [
                t,
                cov,
                acc_1h,
                acc_4h,
                base_pred1_acc,
                pred1_acc,
                pred1_acc_agree,
                agree_rate,
                base_label1_rate,
            ]
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "coverage",
                "filter_acc_1h",
                "filter_acc_4h",
                "pred1_acc_all",
                "pred1_acc_given_filter",
                "pred1_acc_when_agree",
                "filter_pred1_agree_rate",
                "label1_pos_rate",
            ]
        )
        writer.writerows(rows)
    print("pred1_acc_all", base_pred1_acc, "label1_pos_rate", base_label1_rate)


if __name__ == "__main__":
    main()
