from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import load_panel_npz, compress_series_observed
from eval import apply_scaling
from utils import load_config


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


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
    z10 = _logit(0.1)
    z50 = _logit(0.5)
    z90 = _logit(0.9)
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


def _metrics_from_arrays(y_true: np.ndarray, p_up: np.ndarray) -> Dict[str, float]:
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


def _risk_coverage(y_true: np.ndarray, p_up: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cov = np.zeros_like(thresholds, dtype=np.float32)
    acc = np.zeros_like(thresholds, dtype=np.float32)
    for i, t in enumerate(thresholds):
        accept = (p_up >= t) | (p_up <= (1.0 - t))
        if not np.any(accept):
            cov[i] = 0.0
            acc[i] = np.nan
            continue
        cov[i] = float(np.mean(accept))
        y_pred = (p_up[accept] >= 0.5).astype(int)
        acc[i] = float(np.mean(y_pred == y_true[accept]))
    return cov, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--delta_mode", default="origin", choices=["origin", "step"])
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

    y_orig = pre.inverse_y(y)
    q10_orig = pre.inverse_y(q10)
    q50_orig = pre.inverse_y(q50)
    q90_orig = pre.inverse_y(q90)

    y_t_scaled = np.array([series_list[s_idx].y[t] for s_idx, t in zip(series_idx, origin_t)])
    y_t_orig = pre.inverse_y(y_t_scaled)
    if y_t_orig.ndim == 2 and y_t_orig.shape[1] > 1:
        y_t_orig = y_t_orig[:, 0]

    if args.delta_mode == "origin":
        ref = y_t_orig[:, None]
    else:
        ref = np.concatenate([y_t_orig[:, None], y_orig[:, :-1]], axis=1)
    delta = y_orig - ref
    label = (delta > 0).astype(np.int32)

    ref_full = ref if ref.shape[1] == y_orig.shape[1] else np.repeat(ref, y_orig.shape[1], axis=1)
    cdf = _safe_cdf_from_quantiles(
        ref_full,
        q10_orig,
        q50_orig,
        q90_orig,
        min_width=args.min_width,
        slope_cap=args.slope_cap,
        z_clip=args.z_clip,
    )
    p_up = 1.0 - cdf

    H = y_orig.shape[1]
    rows = []
    mcc_h = np.full(H, np.nan, dtype=np.float32)
    auc_h = np.full(H, np.nan, dtype=np.float32)
    brier_h = np.full(H, np.nan, dtype=np.float32)
    acc_h = np.full(H, np.nan, dtype=np.float32)
    ba_h = np.full(H, np.nan, dtype=np.float32)
    ece_h = np.full(H, np.nan, dtype=np.float32)
    n_h = np.zeros(H, dtype=np.int64)

    for h in range(H):
        m = mask[:, h] > 0
        if not np.any(m):
            rows.append([h + 1, "nan", "nan", "nan", "nan", "nan", "nan", "nan", 0])
            continue
        y_h = label[m, h]
        p_h = p_up[m, h]
        metrics_h = _metrics_from_arrays(y_h, p_h)
        mcc_h[h] = metrics_h["mcc"]
        auc_h[h] = metrics_h["auc"]
        brier_h[h] = metrics_h["brier"]
        acc_h[h] = metrics_h["acc"]
        ba_h[h] = metrics_h["ba"]
        ece_h[h] = metrics_h["ece"]
        n_h[h] = metrics_h["n"]
        rows.append(
            [
                h + 1,
                metrics_h["mcc"],
                metrics_h["auc"],
                metrics_h["brier"],
                metrics_h["acc"],
                metrics_h["ba"],
                metrics_h["ece"],
                metrics_h["pos_rate"],
                metrics_h["n"],
            ]
        )

    weights = np.zeros(H, dtype=np.float32)
    weights[:3] = 1.0
    weights[3:8] = 0.5
    weights[8:] = 0.2
    valid = ~np.isnan(mcc_h)
    w_mcc = float(np.sum(mcc_h[valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))
    w_auc = float(np.sum(auc_h[valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))

    mask_flat = mask > 0
    y_flat = label[mask_flat]
    p_flat = p_up[mask_flat]
    micro = _metrics_from_arrays(y_flat, p_flat)

    series_ids = np.unique(series_idx)
    macro_acc = []
    macro_ba = []
    macro_mcc = []
    macro_auc = []
    macro_brier = []
    macro_ece = []
    for s_id in series_ids:
        s_mask = series_idx == s_id
        if not np.any(s_mask):
            continue
        y_s = label[s_mask]
        p_s = p_up[s_mask]
        m_s = mask[s_mask] > 0
        y_s = y_s[m_s]
        p_s = p_s[m_s]
        if y_s.size == 0:
            continue
        m = _metrics_from_arrays(y_s, p_s)
        macro_acc.append(m["acc"])
        macro_ba.append(m["ba"])
        macro_mcc.append(m["mcc"])
        macro_auc.append(m["auc"])
        macro_brier.append(m["brier"])
        macro_ece.append(m["ece"])

    def _nanmean(x: list[float]) -> float:
        if not x:
            return float("nan")
        return float(np.nanmean(np.asarray(x, dtype=np.float32)))

    bands = {"1-6": slice(0, 6), "7-12": slice(6, 12), "13-24": slice(12, 24)}
    band_metrics = {}
    for name, sl in bands.items():
        m = mask[:, sl] > 0
        y_b = label[:, sl][m]
        p_b = p_up[:, sl][m]
        band_metrics[name] = _metrics_from_arrays(y_b, p_b)

    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], dtype=np.float32)
    cov, acc = _risk_coverage(y_flat, p_flat, thresholds)

    metrics = {
        "dirscore_wMCC": w_mcc,
        "dirscore_wAUC": w_auc,
        "micro": micro,
        "macro": {
            "acc": _nanmean(macro_acc),
            "ba": _nanmean(macro_ba),
            "mcc": _nanmean(macro_mcc),
            "auc": _nanmean(macro_auc),
            "brier": _nanmean(macro_brier),
            "ece": _nanmean(macro_ece),
        },
        "band_metrics": band_metrics,
        "risk_coverage_thresholds": thresholds,
        "risk_coverage_acc": acc,
        "risk_coverage_cov": cov,
        "delta_mode": args.delta_mode,
    }

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["h", "mcc", "auc", "brier", "acc", "ba", "ece", "pos_rate", "n"])
        writer.writerows(rows)

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_metrics,
        metrics=metrics,
        mcc=mcc_h,
        auc=auc_h,
        brier=brier_h,
        acc=acc_h,
        ba=ba_h,
        ece=ece_h,
        n=n_h,
    )
    print(metrics)


if __name__ == "__main__":
    main()
