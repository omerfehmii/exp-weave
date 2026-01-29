from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.loader import load_panel_npz, compress_series_observed
from data.features import compute_direction_features
from eval import apply_scaling
from train import filter_indices_with_observed
from utils import load_config, set_seed


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


def _compute_eps_quantile(
    series_list: list,
    indices: List[tuple],
    H: int,
    q: float,
    delta_mode: str,
    scale_std: float | None,
) -> np.ndarray:
    buckets = [[] for _ in range(H)]
    for s_idx, t in indices:
        s = series_list[s_idx]
        y = s.y
        if y.ndim == 1:
            y = y[:, None]
        mask = s.mask
        if mask is None:
            mask = np.ones_like(y, dtype=np.float32)
        if mask.ndim == 1:
            mask = mask[:, None]
        if t + H >= y.shape[0]:
            continue
        y0 = y[t, 0]
        if mask[t, 0] <= 0:
            continue
        for h in range(H):
            idx = t + h + 1
            if mask[idx, 0] <= 0:
                continue
            if delta_mode == "step":
                prev_idx = t + h
                if mask[prev_idx, 0] <= 0:
                    continue
                prev = y[prev_idx, 0]
            else:
                prev = y0
            delta = y[idx, 0] - prev
            buckets[h].append(abs(float(delta)))
    eps = np.zeros(H, dtype=np.float32)
    for h in range(H):
        if buckets[h]:
            eps[h] = float(np.quantile(np.asarray(buckets[h], dtype=np.float32), q))
    if scale_std is not None:
        eps = eps * float(scale_std)
    return eps


def _da_move(label: np.ndarray, p_up: np.ndarray, move_mask: np.ndarray) -> float:
    if not np.any(move_mask):
        return float("nan")
    y = label[move_mask]
    p = p_up[move_mask]
    pred = (p >= 0.5).astype(int)
    return float(np.mean(pred == y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--delta_mode", default="origin", choices=["origin", "step"])
    parser.add_argument("--epsilon_mode", default="quantile", choices=["quantile", "fixed", "vol"])
    parser.add_argument("--epsilon_q", type=float, default=0.33)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--epsilon_k", type=float, default=1.0)
    parser.add_argument("--epsilon_window", type=int, default=24)
    parser.add_argument("--permute_runs", type=int, default=1)
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
    if args.epsilon_mode == "quantile":
        scale_std = None
        if pre is not None and pre.y_scaler.std is not None:
            scale_std = float(np.ravel(pre.y_scaler.std)[0])
        eps = _compute_eps_quantile(series_list, train_idx, H, args.epsilon_q, args.delta_mode, scale_std)
        eps_matrix = eps.reshape(1, -1)
    elif args.epsilon_mode == "vol":
        scale_std = 1.0
        if pre is not None and pre.y_scaler.std is not None:
            scale_std = float(np.ravel(pre.y_scaler.std)[0])
        sigmas = []
        for s in series_list:
            feats = compute_direction_features(s.y, window=args.epsilon_window)
            if feats.ndim == 2:
                sigma = feats[:, 2]
            else:
                sigma = feats[:, 2::3].mean(axis=1)
            sigmas.append(sigma)
        sigma_sample = np.array([sigmas[s_idx][t] for s_idx, t in zip(series_idx, origin_t)], dtype=np.float32)
        eps_matrix = args.epsilon_k * sigma_sample[:, None] * np.sqrt(np.arange(1, H + 1, dtype=np.float32))
        eps_matrix = eps_matrix * scale_std
        eps = np.nanmean(eps_matrix, axis=0)
    else:
        eps = np.full(H, args.epsilon, dtype=np.float32)
        eps_matrix = eps.reshape(1, -1)

    flat_rate = np.zeros(H, dtype=np.float32)
    da_move = np.zeros(H, dtype=np.float32)
    da_perm = np.zeros(H, dtype=np.float32)
    for h in range(H):
        m = mask[:, h] > 0
        if not np.any(m):
            flat_rate[h] = np.nan
            da_move[h] = np.nan
            da_perm[h] = np.nan
            continue
        delta_h = delta[m, h]
        move_mask = np.abs(delta_h) > eps_matrix[m, h]
        flat_rate[h] = float(1.0 - np.mean(move_mask))
        da_move[h] = _da_move(label[m, h], p_up[m, h], move_mask)
        if args.permute_runs > 0:
            perm_scores = []
            for _ in range(args.permute_runs):
                perm = np.random.permutation(p_up[m, h])
                perm_scores.append(_da_move(label[m, h], perm, move_mask))
            da_perm[h] = float(np.nanmean(perm_scores))
        else:
            da_perm[h] = np.nan

    bands = {"1-6": slice(0, 6), "7-12": slice(6, 12), "13-24": slice(12, 24)}
    band = {}
    for name, sl in bands.items():
        band[name] = {
            "flat_rate": float(np.nanmean(flat_rate[sl])),
            "da_move": float(np.nanmean(da_move[sl])),
            "da_perm": float(np.nanmean(da_perm[sl])),
        }

    metrics = {
        "delta_mode": args.delta_mode,
        "epsilon_mode": args.epsilon_mode,
        "epsilon_q": args.epsilon_q if args.epsilon_mode == "quantile" else None,
        "epsilon": float(args.epsilon) if args.epsilon_mode == "fixed" else None,
        "flat_rate": flat_rate,
        "da_move": da_move,
        "da_perm": da_perm,
        "band": band,
    }

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_metrics, metrics=metrics, flat_rate=flat_rate, da_move=da_move, da_perm=da_perm, eps=eps)
    print(metrics)


if __name__ == "__main__":
    main()
