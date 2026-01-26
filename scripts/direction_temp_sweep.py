from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

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
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


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


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return float(0.5 * (tpr + tnr))


def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    if total == 0:
        return float("nan")
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(probs[mask]))
        ece += (np.sum(mask) / total) * abs(acc - conf)
    return float(ece)


def _metrics_from_arrays(y_true: np.ndarray, p_up: np.ndarray) -> dict:
    y_pred = (p_up >= 0.5).astype(int)
    return {
        "acc": float(np.mean(y_pred == y_true)) if y_true.size else float("nan"),
        "ba": _balanced_accuracy(y_true, y_pred),
        "mcc": _mcc(y_true, y_pred),
        "auc": _roc_auc(y_true, p_up),
        "brier": float(np.mean((p_up - y_true) ** 2)) if y_true.size else float("nan"),
        "ece": _ece(y_true, p_up),
        "pos_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "n": int(y_true.size),
    }


def _fit_temperature(logits: np.ndarray, labels: np.ndarray, t_grid: np.ndarray) -> float:
    if labels.size == 0:
        return 1.0
    n_pos = int(np.sum(labels))
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    best_t = 1.0
    best_nll = float("inf")
    for t in t_grid:
        z = logits / t
        p = 1.0 / (1.0 + np.exp(-z))
        nll = -np.mean(labels * np.log(p + 1e-12) + (1 - labels) * np.log(1 - p + 1e-12))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def _prepare_arrays(cfg: dict, preds_path: str, delta_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.load(preds_path)
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

    y_orig = pre.inverse_y(y)
    q10_orig = pre.inverse_y(q10)
    q50_orig = pre.inverse_y(q50)
    q90_orig = pre.inverse_y(q90)

    y_t_scaled = np.array([series_list[s_idx].y[t] for s_idx, t in zip(series_idx, origin_t)])
    y_t_orig = pre.inverse_y(y_t_scaled)
    if y_t_orig.ndim == 2 and y_t_orig.shape[1] > 1:
        y_t_orig = y_t_orig[:, 0]

    if delta_mode == "origin":
        ref = y_t_orig[:, None]
    else:
        ref = np.concatenate([y_t_orig[:, None], y_orig[:, :-1]], axis=1)
    delta = y_orig - ref
    label = (delta > 0).astype(np.int32)
    ref_full = ref if ref.shape[1] == y_orig.shape[1] else np.repeat(ref, y_orig.shape[1], axis=1)
    return (q10_orig, q50_orig, q90_orig, ref_full, label, mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--calib_preds", required=True)
    parser.add_argument("--test_preds", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--delta_mode", default="origin", choices=["origin", "step"])
    parser.add_argument("--mode", default="per_horizon", choices=["per_horizon", "global"])
    parser.add_argument("--t_min", type=float, default=0.5)
    parser.add_argument("--t_max", type=float, default=5.0)
    parser.add_argument("--t_steps", type=int, default=30)
    parser.add_argument("--min_count", type=int, default=200)
    parser.add_argument("--min_width_factors", default="0.05,0.1,0.2")
    parser.add_argument("--slope_caps", default="2,5,10")
    parser.add_argument("--z_clips", default="6,10,20")
    args = parser.parse_args()

    cfg = load_config(args.config)
    t_grid = np.linspace(args.t_min, args.t_max, args.t_steps)

    q10_c, q50_c, q90_c, ref_c, label_c, mask_c = _prepare_arrays(cfg, args.calib_preds, args.delta_mode)
    q10_t, q50_t, q90_t, ref_t, label_t, mask_t = _prepare_arrays(cfg, args.test_preds, args.delta_mode)

    width = (q90_c - q10_c)
    width = width[np.isfinite(width)]
    width_median = float(np.median(width)) if width.size else 1.0

    min_width_factors = [float(x) for x in args.min_width_factors.split(",")]
    slope_caps = [float(x) for x in args.slope_caps.split(",")]
    z_clips = [float(x) for x in args.z_clips.split(",")]

    H = q10_c.shape[1]
    weights = np.zeros(H, dtype=np.float32)
    weights[:3] = 1.0
    weights[3:8] = 0.5
    weights[8:] = 0.2

    rows: List[list] = []
    for f in min_width_factors:
        min_width = max(width_median * f, 1e-6)
        for slope_cap in slope_caps:
            for z_clip in z_clips:
                cdf_c = _safe_cdf_from_quantiles(ref_c, q10_c, q50_c, q90_c, min_width, slope_cap, z_clip)
                p_c = 1.0 - cdf_c
                cdf_t = _safe_cdf_from_quantiles(ref_t, q10_t, q50_t, q90_t, min_width, slope_cap, z_clip)
                p_t = 1.0 - cdf_t

                temps = np.ones(H, dtype=np.float32)
                if args.mode == "global":
                    m = mask_c > 0
                    logits = _logit(p_c[m])
                    labels = label_c[m]
                    if labels.size >= args.min_count:
                        t_global = _fit_temperature(logits, labels, t_grid)
                    else:
                        t_global = 1.0
                    temps[:] = t_global
                else:
                    for h in range(H):
                        m = mask_c[:, h] > 0
                        logits = _logit(p_c[m, h])
                        labels = label_c[m, h]
                        if labels.size < args.min_count:
                            temps[h] = 1.0
                            continue
                        temps[h] = _fit_temperature(logits, labels, t_grid)

                logits_t = _logit(p_t)
                p_cal = 1.0 / (1.0 + np.exp(-(logits_t / temps.reshape(1, -1))))

                mcc = []
                auc = []
                ece = []
                brier = []
                for h in range(H):
                    m = mask_t[:, h] > 0
                    if not m.any():
                        mcc.append(np.nan)
                        auc.append(np.nan)
                        ece.append(np.nan)
                        brier.append(np.nan)
                        continue
                    mtr = _metrics_from_arrays(label_t[m, h], p_cal[m, h])
                    mcc.append(mtr["mcc"])
                    auc.append(mtr["auc"])
                    ece.append(mtr["ece"])
                    brier.append(mtr["brier"])
                mcc = np.array(mcc)
                auc = np.array(auc)
                ece = np.array(ece)
                brier = np.array(brier)
                valid = ~np.isnan(mcc)
                w_mcc = float(np.sum(mcc[valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))
                w_auc = float(np.sum(auc[valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))
                ece_mean = float(np.nanmean(ece))
                brier_mean = float(np.nanmean(brier))
                sat = float(np.mean((p_cal <= 0.01) | (p_cal >= 0.99)))
                rows.append([min_width, slope_cap, z_clip, w_mcc, w_auc, ece_mean, brier_mean, sat])

    rows.sort(key=lambda r: (-r[3], r[5]))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["min_width", "slope_cap", "z_clip", "w_mcc", "w_auc", "ece", "brier", "sat_frac"])
        writer.writerows(rows)

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_metrics, rows=np.asarray(rows, dtype=np.float32))
    print("top5", rows[:5])


if __name__ == "__main__":
    main()
