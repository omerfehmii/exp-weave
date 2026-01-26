from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.features import compute_direction_features
from data.loader import load_panel_npz, compress_series_observed
from eval import apply_scaling
from train import compute_delta_thresholds, default_horizon_weights, filter_indices_with_observed
from utils import load_config, set_seed


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


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
    pos_rate = float(np.mean(y_true)) if y_true.size else float("nan")
    chance = float(max(pos_rate, 1.0 - pos_rate)) if y_true.size else float("nan")
    return {
        "acc": float(np.mean(y_pred == y_true)) if y_true.size else float("nan"),
        "mcc": _mcc(y_true, y_pred),
        "auc": _roc_auc(y_true, p_up),
        "brier": float(np.mean((p_up - y_true) ** 2)) if y_true.size else float("nan"),
        "ece": _ece(y_true, p_up),
        "pos_rate": pos_rate,
        "chance_acc": chance,
        "n": int(y_true.size),
    }


def _compute_sigma_samples(
    series_list: list,
    origin_t: np.ndarray,
    series_idx: np.ndarray,
    window: int,
) -> np.ndarray:
    sigmas = []
    for s in series_list:
        feats = compute_direction_features(s.y, window=window)
        if feats.ndim == 2:
            sigma = feats[:, 2]
        else:
            sigma = feats[:, 2::3].mean(axis=1)
        sigmas.append(sigma)
    return np.array([sigmas[s_idx][t] for s_idx, t in zip(series_idx, origin_t)], dtype=np.float32)


def _apply_sweep(
    label: np.ndarray,
    p_up: np.ndarray,
    mask: np.ndarray,
    delta: np.ndarray,
    eps_matrix: np.ndarray,
    h_idx: List[int] | None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    H = label.shape[1]
    if not h_idx:
        h_idx = list(range(H))
    dir_acc = np.full(H, np.nan, dtype=np.float32)
    dir_mcc = np.full(H, np.nan, dtype=np.float32)
    dir_auc = np.full(H, np.nan, dtype=np.float32)
    dir_brier = np.full(H, np.nan, dtype=np.float32)
    dir_ece = np.full(H, np.nan, dtype=np.float32)
    pos_rate = np.full(H, np.nan, dtype=np.float32)
    chance_acc = np.full(H, np.nan, dtype=np.float32)
    move_rate = np.full(H, np.nan, dtype=np.float32)
    flat_rate = np.full(H, np.nan, dtype=np.float32)

    abs_delta = np.abs(delta)
    for h in range(H):
        m = mask[:, h] > 0
        if not np.any(m):
            continue
        move_mask = m & (abs_delta[:, h] >= eps_matrix[:, h])
        move_rate[h] = float(np.mean(move_mask[m]))
        flat_rate[h] = float(1.0 - move_rate[h])
        if not np.any(move_mask):
            continue
        y_h = label[move_mask, h]
        p_h = p_up[move_mask, h]
        metrics = _metrics_from_arrays(y_h, p_h)
        dir_acc[h] = metrics["acc"]
        dir_mcc[h] = metrics["mcc"]
        dir_auc[h] = metrics["auc"]
        dir_brier[h] = metrics["brier"]
        dir_ece[h] = metrics["ece"]
        pos_rate[h] = metrics["pos_rate"]
        chance_acc[h] = metrics["chance_acc"]

    idx = np.asarray(h_idx, dtype=np.int64)
    weights = default_horizon_weights(H)[idx]
    valid = ~np.isnan(dir_mcc[idx])
    w_mcc = float(np.sum(dir_mcc[idx][valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))
    w_auc = float(np.sum(dir_auc[idx][valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))
    w_acc = float(np.sum(dir_acc[idx][valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))
    summary = {
        "dirscore_wMCC": w_mcc,
        "dirscore_wAUC": w_auc,
        "dirscore_wACC": w_acc,
        "dir_acc_mean": float(np.nanmean(dir_acc[idx])),
        "dir_mcc_mean": float(np.nanmean(dir_mcc[idx])),
        "dir_auc_mean": float(np.nanmean(dir_auc[idx])),
        "dir_brier_mean": float(np.nanmean(dir_brier[idx])),
        "dir_ece_mean": float(np.nanmean(dir_ece[idx])),
        "move_rate_mean": float(np.nanmean(move_rate[idx])),
        "flat_rate_mean": float(np.nanmean(flat_rate[idx])),
        "chance_acc_mean": float(np.nanmean(chance_acc[idx])),
    }
    arrays = {
        "dir_acc": dir_acc,
        "dir_mcc": dir_mcc,
        "dir_auc": dir_auc,
        "dir_brier": dir_brier,
        "dir_ece": dir_ece,
        "pos_rate": pos_rate,
        "chance_acc": chance_acc,
        "move_rate": move_rate,
        "flat_rate": flat_rate,
    }
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--delta_mode", default="origin", choices=["origin", "step"])
    parser.add_argument("--step_ref", default="pred", choices=["true", "pred"])
    parser.add_argument("--horizons", default="")
    parser.add_argument("--epsilon_mode", default="quantile", choices=["quantile", "fixed", "vol"])
    parser.add_argument("--epsilon_qs", default="0.0,0.1,0.2,0.33,0.5")
    parser.add_argument("--epsilon_values", default="0.0")
    parser.add_argument("--epsilon_ks", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--epsilon_window", type=int, default=24)
    parser.add_argument("--min_width", type=float, default=1e-6)
    parser.add_argument("--slope_cap", type=float, default=10.0)
    parser.add_argument("--z_clip", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split_purge", type=int, default=None)
    parser.add_argument("--split_embargo", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
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
    split = make_time_splits(min(lengths), cfg["data"].get("train_frac", 0.7), cfg["data"].get("val_frac", 0.15))
    split_purge = int(cfg["data"].get("split_purge", 0)) if args.split_purge is None else int(args.split_purge)
    split_embargo = int(cfg["data"].get("split_embargo", 0)) if args.split_embargo is None else int(args.split_embargo)
    indices = generate_panel_origins(lengths, cfg["data"]["L"], cfg["data"]["H"], cfg["data"].get("step", cfg["data"]["H"]))
    horizon = cfg["data"]["H"]
    train_idx = select_indices_by_time(
        indices,
        split,
        "train",
        horizon=horizon,
        purge=split_purge,
        embargo=split_embargo,
    )
    train_idx = filter_indices_with_observed(
        series_list,
        train_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        cfg["data"].get("min_past_obs", 1),
        cfg["data"].get("min_future_obs", 1),
    )
    pre = apply_scaling(
        series_list,
        split.train_end,
        scale_x=cfg["data"].get("scale_x", True),
        scale_y=cfg["data"].get("scale_y", True),
    )
    scale_std = 1.0
    if pre is not None and pre.y_scaler.std is not None:
        scale_std = float(np.ravel(pre.y_scaler.std)[0])

    y_orig = pre.inverse_y(y)
    q10_orig = pre.inverse_y(q10)
    q50_orig = pre.inverse_y(q50)
    q90_orig = pre.inverse_y(q90)

    y_t_scaled = np.array([series_list[s_idx].y[t] for s_idx, t in zip(series_idx, origin_t)])
    y_t_orig = pre.inverse_y(y_t_scaled)
    if y_t_orig.ndim == 2 and y_t_orig.shape[1] > 1:
        y_t_orig = y_t_orig[:, 0]

    if args.delta_mode == "origin":
        ref_label = y_t_orig[:, None]
        ref_pred = ref_label
    else:
        ref_label = np.concatenate([y_t_orig[:, None], y_orig[:, :-1]], axis=1)
        if args.step_ref == "pred":
            ref_pred = np.concatenate([y_t_orig[:, None], q50_orig[:, :-1]], axis=1)
        else:
            ref_pred = ref_label
            print("warning: step_ref=true uses future y_{t+h-1} for reference.")
    delta = y_orig - ref_label
    label = (delta > 0).astype(np.int32)
    ref_full = ref_pred if ref_pred.shape[1] == y_orig.shape[1] else np.repeat(ref_pred, y_orig.shape[1], axis=1)
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

    if args.horizons.strip():
        h_list = [h for h in _parse_ints(args.horizons) if 1 <= h <= horizon]
    else:
        h_list = list(range(1, horizon + 1))
    h_idx = [h - 1 for h in h_list]

    rows = []
    sweep_vals = []
    sweep_eps_mean = []
    sweep_arrays = []
    sweep_summaries = []

    if args.epsilon_mode == "quantile":
        qs = _parse_floats(args.epsilon_qs)
        for q in qs:
            tau_h = compute_delta_thresholds(series_list, train_idx, horizon, q, delta_mode=args.delta_mode)
            eps = tau_h * scale_std
            eps_matrix = eps.reshape(1, -1)
            arrays, summary = _apply_sweep(label, p_up, mask, delta, eps_matrix, h_idx)
            summary["epsilon_q"] = q
            summary["eps_mean"] = float(np.nanmean(eps))
            sweep_vals.append(q)
            sweep_eps_mean.append(float(np.nanmean(eps)))
            sweep_arrays.append(arrays)
            sweep_summaries.append(summary)
    elif args.epsilon_mode == "vol":
        ks = _parse_floats(args.epsilon_ks)
        sigma_sample = _compute_sigma_samples(series_list, origin_t, series_idx, args.epsilon_window)
        sqrt_h = np.sqrt(np.arange(1, horizon + 1, dtype=np.float32))
        for k in ks:
            eps_matrix = k * sigma_sample[:, None] * sqrt_h[None, :]
            eps_matrix = eps_matrix * scale_std
            arrays, summary = _apply_sweep(label, p_up, mask, delta, eps_matrix, h_idx)
            summary["epsilon_k"] = k
            summary["eps_mean"] = float(np.nanmean(eps_matrix))
            sweep_vals.append(k)
            sweep_eps_mean.append(float(np.nanmean(eps_matrix)))
            sweep_arrays.append(arrays)
            sweep_summaries.append(summary)
    else:
        vals = _parse_floats(args.epsilon_values)
        for eps in vals:
            eps_matrix = np.full_like(delta, eps, dtype=np.float32)
            arrays, summary = _apply_sweep(label, p_up, mask, delta, eps_matrix, h_idx)
            summary["epsilon"] = eps
            summary["eps_mean"] = float(eps)
            sweep_vals.append(eps)
            sweep_eps_mean.append(float(eps))
            sweep_arrays.append(arrays)
            sweep_summaries.append(summary)

    for idx, summary in enumerate(sweep_summaries):
        rows.append(
            [
                sweep_vals[idx],
                summary.get("eps_mean", float("nan")),
                summary.get("flat_rate_mean", float("nan")),
                summary.get("move_rate_mean", float("nan")),
                summary.get("dir_acc_mean", float("nan")),
                summary.get("dir_mcc_mean", float("nan")),
                summary.get("dir_auc_mean", float("nan")),
                summary.get("dir_brier_mean", float("nan")),
                summary.get("dir_ece_mean", float("nan")),
                summary.get("chance_acc_mean", float("nan")),
                summary.get("dirscore_wACC", float("nan")),
                summary.get("dirscore_wMCC", float("nan")),
                summary.get("dirscore_wAUC", float("nan")),
            ]
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sweep_value",
                "eps_mean",
                "flat_rate_mean",
                "move_rate_mean",
                "dir_acc_mean",
                "dir_mcc_mean",
                "dir_auc_mean",
                "dir_brier_mean",
                "dir_ece_mean",
                "chance_acc_mean",
                "dirscore_wACC",
                "dirscore_wMCC",
                "dirscore_wAUC",
            ]
        )
        writer.writerows(rows)

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    sweep_dict = {
        "mode": args.epsilon_mode,
        "sweep_values": np.asarray(sweep_vals, dtype=np.float32),
        "eps_mean": np.asarray(sweep_eps_mean, dtype=np.float32),
        "summaries": sweep_summaries,
        "horizons": h_list,
    }
    np.savez(
        out_metrics,
        sweep=sweep_dict,
        dir_acc=np.stack([a["dir_acc"] for a in sweep_arrays], axis=0),
        dir_mcc=np.stack([a["dir_mcc"] for a in sweep_arrays], axis=0),
        dir_auc=np.stack([a["dir_auc"] for a in sweep_arrays], axis=0),
        dir_brier=np.stack([a["dir_brier"] for a in sweep_arrays], axis=0),
        dir_ece=np.stack([a["dir_ece"] for a in sweep_arrays], axis=0),
        move_rate=np.stack([a["move_rate"] for a in sweep_arrays], axis=0),
        flat_rate=np.stack([a["flat_rate"] for a in sweep_arrays], axis=0),
        chance_acc=np.stack([a["chance_acc"] for a in sweep_arrays], axis=0),
        pos_rate=np.stack([a["pos_rate"] for a in sweep_arrays], axis=0),
    )
    print(sweep_dict)


if __name__ == "__main__":
    main()
