from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from calibration.aci import ACIGuardrail, ResidualBuffer, apply_scale, normalized_residual, update_alpha
from calibration.aci_state import load_state, save_state
from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.loader import load_panel_npz, compress_series_observed
from train import filter_indices_with_observed
from utils import load_config


def _hour_of_day(timestamps: np.ndarray, t: int) -> int:
    try:
        import pandas as pd
    except ImportError:
        return int(t % 24)
    ts = pd.to_datetime(timestamps[t])
    return int(ts.hour)


def _volatility(y: np.ndarray, mask: np.ndarray, t: int, L: int) -> float:
    past = y[t - L + 1 : t + 1]
    if mask is not None:
        past_mask = mask[t - L + 1 : t + 1]
        past = np.where(past_mask > 0, past, np.nan)
    past = past[np.isfinite(past)]
    if past.size == 0:
        return 0.0
    return float(np.std(past))


def _compute_vol_thresholds(cfg: dict, series_list: list, vol_bins: int) -> List[float]:
    lengths = [len(s.y) for s in series_list]
    split = make_time_splits(min(lengths), cfg["data"].get("train_frac", 0.7), cfg["data"].get("val_frac", 0.15))
    split_purge = int(cfg["data"].get("split_purge", 0))
    split_embargo = int(cfg["data"].get("split_embargo", 0))
    indices = generate_panel_origins(lengths, cfg["data"]["L"], cfg["data"]["H"], cfg["data"].get("step", cfg["data"]["H"]))
    horizon = cfg["data"]["H"]
    train_idx = select_indices_by_time(indices, split, "train", horizon=horizon, purge=split_purge, embargo=split_embargo)
    train_idx = filter_indices_with_observed(
        series_list,
        train_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        cfg["data"].get("min_past_obs", 1),
        cfg["data"].get("min_future_obs", 1),
    )
    vols = []
    for s_idx, t in train_idx:
        s = series_list[s_idx]
        y = s.y
        if y.ndim > 1:
            y = y[:, 0]
        mask = s.mask
        if mask is not None and mask.ndim > 1:
            mask = mask[:, 0]
        vols.append(_volatility(y, mask, t, cfg["data"]["L"]))
    if not vols:
        return []
    qs = [i / vol_bins for i in range(1, vol_bins)]
    thresholds = np.quantile(np.asarray(vols, dtype=np.float32), qs).tolist()
    return [float(x) for x in thresholds]


def _vol_regime(vol: float, thresholds: List[float]) -> int:
    for idx, thr in enumerate(thresholds):
        if vol <= thr:
            return idx
    return len(thresholds)


def _bucket(hod: int, hod_bins: int, vol_reg: int) -> Tuple[int | None, int | None]:
    bin_size = max(24 // hod_bins, 1)
    hod_bin = min(hod // bin_size, hod_bins - 1)
    return int(hod_bin), int(vol_reg)


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_metrics", default=None)
    parser.add_argument("--alpha_target", type=float, default=0.20)
    parser.add_argument("--coverage_mode", default="per_horizon", choices=["per_horizon", "trajectory"])
    parser.add_argument("--coverage_target", type=float, default=0.80)
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--gamma_base", type=float, default=0.02)
    parser.add_argument("--alpha_clip_min", type=float, default=1e-3)
    parser.add_argument("--alpha_clip_max", type=float, default=0.5)
    parser.add_argument("--eps_width", type=float, default=1e-6)
    parser.add_argument("--s_clip_min", type=float, default=0.1)
    parser.add_argument("--s_clip_max", type=float, default=5.0)
    parser.add_argument("--hod_bins", type=int, default=6)
    parser.add_argument("--vol_bins", type=int, default=3)
    parser.add_argument("--min_count", type=int, default=100)
    parser.add_argument("--shrinkage_tau", type=float, default=1000.0)
    parser.add_argument("--state_in", default=None)
    parser.add_argument("--state_out", default=None)
    parser.add_argument("--recent_window", type=int, default=0, help="If set, only process last N hours.")
    parser.add_argument("--update_mode", default="strict", choices=["strict", "leaky", "frozen"])
    parser.add_argument("--retro_refresh", default="false")
    parser.add_argument("--retro_every", type=int, default=1)
    parser.add_argument("--switch_prob", default="false")
    parser.add_argument("--ema_fast", type=float, default=0.2)
    parser.add_argument("--ema_slow", type=float, default=0.02)
    parser.add_argument("--switch_gamma", type=float, default=1.0)
    parser.add_argument("--lambda_min", type=float, default=0.15)
    parser.add_argument("--lambda_max", type=float, default=0.5)
    parser.add_argument("--switch_eps", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    preds = np.load(args.preds)
    y = preds["y"]
    q10 = preds["q10"]
    q50 = preds["q50"]
    q90 = preds["q90"]
    mask = preds["mask"] if "mask" in preds else np.isfinite(y).astype(np.float32)
    if "origin_t" not in preds or "series_idx" not in preds:
        raise ValueError("preds npz must include origin_t and series_idx (re-run eval with --out_npz).")
    origin_t = preds["origin_t"].astype(np.int64)
    series_idx = preds["series_idx"].astype(np.int64)

    H = q50.shape[1]
    alpha_target = args.alpha_target
    if args.coverage_mode == "trajectory":
        alpha_target = min(alpha_target, (1.0 - args.coverage_target) / max(H, 1))
    alpha = np.full(H, alpha_target, dtype=np.float32)
    gamma = np.array([args.gamma_base / np.sqrt(h + 1) for h in range(H)], dtype=np.float32)
    alpha_clip = (args.alpha_clip_min, args.alpha_clip_max)
    guard = ACIGuardrail(eps_width=args.eps_width, s_clip=(args.s_clip_min, args.s_clip_max))
    buffer = ResidualBuffer(args.window, args.min_count, args.shrinkage_tau, guard)

    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    vol_thresholds = _compute_vol_thresholds(cfg, series_list, args.vol_bins)
    if args.state_in:
        alpha_in, buffer_in, vol_thr_in, last_t, cfg_in = load_state(args.state_in)
        if alpha_in:
            alpha = np.array(alpha_in, dtype=np.float32)
        buffer = buffer_in
        if vol_thr_in:
            vol_thresholds = vol_thr_in
        if cfg_in.get("hod_bins") is not None:
            args.hod_bins = int(cfg_in.get("hod_bins", args.hod_bins))
        if cfg_in.get("vol_bins") is not None:
            args.vol_bins = int(cfg_in.get("vol_bins", args.vol_bins))
        if cfg_in.get("window") is not None:
            args.window = int(cfg_in.get("window", args.window))
        if cfg_in.get("min_count") is not None:
            args.min_count = int(cfg_in.get("min_count", args.min_count))
        if cfg_in.get("shrinkage_tau") is not None:
            args.shrinkage_tau = float(cfg_in.get("shrinkage_tau", args.shrinkage_tau))
        if last_t is not None:
            keep = origin_t > int(last_t)
            y = y[keep]
            q10 = q10[keep]
            q50 = q50[keep]
            q90 = q90[keep]
            mask = mask[keep]
            origin_t = origin_t[keep]
            series_idx = series_idx[keep]

    if args.recent_window and origin_t.size:
        max_t = int(np.max(origin_t))
        keep = origin_t >= (max_t - args.recent_window)
        y = y[keep]
        q10 = q10[keep]
        q50 = q50[keep]
        q90 = q90[keep]
        mask = mask[keep]
        origin_t = origin_t[keep]
        series_idx = series_idx[keep]

    # Precompute buckets per sample
    buckets: List[Tuple[int | None, int | None]] = []
    for idx, (s_idx, t) in enumerate(zip(series_idx, origin_t)):
        s = series_list[s_idx]
        if s.timestamps is not None:
            hod = _hour_of_day(s.timestamps, t)
        else:
            hod = int(t % 24)
        y_series = s.y
        if y_series.ndim > 1:
            y_series = y_series[:, 0]
        mask_series = s.mask
        if mask_series is not None and mask_series.ndim > 1:
            mask_series = mask_series[:, 0]
        vol = _volatility(y_series, mask_series, t, cfg["data"]["L"])
        vol_reg = _vol_regime(vol, vol_thresholds)
        buckets.append(_bucket(hod, args.hod_bins, vol_reg))

    # Group indices by origin time
    indices_by_origin: Dict[int, List[int]] = defaultdict(list)
    for idx, t in enumerate(origin_t):
        indices_by_origin[int(t)].append(idx)
    origins = sorted(indices_by_origin.keys())

    updates_by_time: Dict[int, List[Tuple[int, Tuple[int | None, int | None], float, float]]] = defaultdict(list)
    residual_by_time: Dict[int, List[Tuple[int, Tuple[int | None, int | None], float]]] = defaultdict(list)
    q10_cal = q10.copy()
    q90_cal = q90.copy()
    retro_refresh = _parse_bool(args.retro_refresh)
    switch_prob = _parse_bool(args.switch_prob)
    ema_fast = np.zeros(H, dtype=np.float32)
    ema_slow = np.zeros(H, dtype=np.float32)
    lambda_schedule = np.linspace(args.lambda_min, args.lambda_max, H).astype(np.float32)

    for t in origins:
        # Apply matured updates (embargo) unless frozen/leaky
        if args.update_mode == "strict":
            for mt in sorted([k for k in updates_by_time.keys() if k <= t]):
                for h, bucket, r, s_used in updates_by_time[mt]:
                    miss = 1.0 if r > (s_used - 1.0) else 0.0
                    alpha[h] = update_alpha(alpha[h], miss, alpha_target, gamma[h], alpha_clip)
                    if switch_prob:
                        r_pos = max(r, 0.0)
                        ema_fast[h] = (1.0 - args.ema_fast) * ema_fast[h] + args.ema_fast * r_pos
                        ema_slow[h] = (1.0 - args.ema_slow) * ema_slow[h] + args.ema_slow * r_pos
                    if not retro_refresh:
                        buffer.add(h, bucket, mt, r)
                del updates_by_time[mt]
            if retro_refresh and args.retro_every > 0 and (t % args.retro_every == 0):
                buffer.clear()
                window_start = t - args.window
                if switch_prob:
                    ema_fast = np.zeros(H, dtype=np.float32)
                    ema_slow = np.zeros(H, dtype=np.float32)
                for mt in range(window_start, t + 1):
                    for h, bucket, r in residual_by_time.get(mt, []):
                        buffer.add(h, bucket, mt, r)
                        if switch_prob:
                            r_pos = max(r, 0.0)
                            ema_fast[h] = (1.0 - args.ema_fast) * ema_fast[h] + args.ema_fast * r_pos
                            ema_slow[h] = (1.0 - args.ema_slow) * ema_slow[h] + args.ema_slow * r_pos
            else:
                buffer.trim(t)

        for idx in indices_by_origin[t]:
            bucket = buckets[idx]
            fallback_bucket = (None, bucket[1])
            global_bucket = (None, None)
            r, wL, wU = normalized_residual(
                y[idx],
                q10[idx],
                q50[idx],
                q90[idx],
                args.eps_width,
            )
            for h in range(H):
                if mask[idx, h] <= 0:
                    continue
                s = buffer.get_scale(h, bucket, float(alpha[h]), fallback_bucket, global_bucket)
                if switch_prob:
                    denom = abs(float(ema_slow[h])) + args.switch_eps
                    z = args.switch_gamma * (float(ema_fast[h]) - float(ema_slow[h])) / denom
                    raw = 1.0 / (1.0 + np.exp(-z))
                    sp = max(0.0, (raw - 0.5) * 2.0)
                    s = s * (1.0 + float(lambda_schedule[h]) * sp)
                    s = float(np.clip(s, args.s_clip_min, args.s_clip_max))
                lo, hi = apply_scale(q50[idx, h], wL[h], wU[h], s)
                q10_cal[idx, h] = lo
                q90_cal[idx, h] = hi
                if args.update_mode == "leaky":
                    miss = 1.0 if float(r[h]) > (s - 1.0) else 0.0
                    alpha[h] = update_alpha(alpha[h], miss, alpha_target, gamma[h], alpha_clip)
                    if switch_prob:
                        r_pos = max(float(r[h]), 0.0)
                        ema_fast[h] = (1.0 - args.ema_fast) * ema_fast[h] + args.ema_fast * r_pos
                        ema_slow[h] = (1.0 - args.ema_slow) * ema_slow[h] + args.ema_slow * r_pos
                    buffer.add(h, bucket, t, float(r[h]))
                elif args.update_mode == "strict":
                    maturity = t + h + 1
                    updates_by_time[maturity].append((h, bucket, float(r[h]), s))
                    if retro_refresh:
                        residual_by_time[maturity].append((h, bucket, float(r[h])))

    # Metrics (overall)
    mask_f = mask.astype(np.float32)
    def _masked_mean(values: np.ndarray, m: np.ndarray) -> float:
        denom = np.sum(m)
        return float(np.sum(values * m) / max(denom, 1.0))

    mae = _masked_mean(np.abs(y - q50), mask_f)
    rmse = float(np.sqrt(_masked_mean((y - q50) ** 2, mask_f)))
    inside = (y >= q10_cal) & (y <= q90_cal)
    cov = _masked_mean(inside.astype(np.float32), mask_f)
    width = _masked_mean(q90_cal - q10_cal, mask_f)
    mask_full = np.all(mask_f > 0, axis=1)
    if np.any(mask_full):
        cov_traj = float(np.mean(np.all(inside[mask_full], axis=1)))
    else:
        cov_traj = float("nan")
    cov_min = float(np.nanmin(np.where(np.sum(mask_f, axis=0) > 0, np.mean(inside * mask_f, axis=0) / np.maximum(np.mean(mask_f, axis=0), 1e-6), np.nan)))
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "coverage90": cov,
        "width90": width,
        "coverage_traj": cov_traj,
        "coverage90_min": cov_min,
        "alpha_target": alpha_target,
    }
    print("aci_metrics", metrics)

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, y=y, q10=q10_cal, q50=q50, q90=q90_cal, mask=mask, origin_t=origin_t, series_idx=series_idx)
    if args.out_metrics:
        np.savez(args.out_metrics, metrics=metrics)
    if args.state_out:
        config_state = {
            "window": args.window,
            "min_count": args.min_count,
            "shrinkage_tau": args.shrinkage_tau,
            "eps_width": args.eps_width,
            "s_clip_min": args.s_clip_min,
            "s_clip_max": args.s_clip_max,
            "hod_bins": args.hod_bins,
            "vol_bins": args.vol_bins,
            "coverage_mode": args.coverage_mode,
            "coverage_target": args.coverage_target,
        }
        last_t = int(np.max(origin_t)) if origin_t.size else None
        save_state(args.state_out, alpha.tolist(), buffer, vol_thresholds, last_t, config_state)


if __name__ == "__main__":
    main()
