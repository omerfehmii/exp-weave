from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import compress_series_observed, load_panel_npz, filter_series_by_active_ratio, filter_series_by_future_ratio
from eval import apply_scaling
from utils import load_config


def _load_series(cfg: Dict) -> list:
    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    min_ratio = float(cfg.get("data", {}).get("universe_min_active_ratio", 0.0))
    min_points = int(cfg.get("data", {}).get("universe_min_active_points", 0))
    active_end = cfg.get("data", {}).get("universe_active_end")
    min_future_ratio = float(cfg.get("data", {}).get("universe_min_future_ratio", 0.0))
    future_horizon = int(cfg.get("data", {}).get("H", cfg.get("data", {}).get("future_horizon", 0)))
    if min_ratio or min_points:
        series_list = filter_series_by_active_ratio(series_list, min_ratio, min_points, active_end)
    if min_future_ratio and future_horizon > 0:
        series_list = filter_series_by_future_ratio(series_list, min_future_ratio, future_horizon, active_end)
    for s in series_list:
        s.ensure_features()
    return series_list


def _summarize(name: str, pnl: np.ndarray) -> None:
    if pnl.size == 0:
        print(f"{name}: empty")
        return
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    sharpe = mean / (std + 1e-12)
    hit = float(np.mean(pnl > 0))
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum)) if cum.size else float("nan")
    print(
        f"{name}: mean={mean:.6f} std={std:.6f} sharpe={sharpe:.3f} "
        f"hit={hit:.3f} max_dd={max_dd:.6f} n={pnl.size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--value_scale", default="orig", choices=["orig", "scaled"])
    parser.add_argument("--weight_mode", default="signed", choices=["signed", "abs"])
    parser.add_argument("--group_by", default="origin", choices=["origin", "timestamp"])
    parser.add_argument("--use_cs", action="store_true")
    parser.add_argument("--ret_cs", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--score_transform", default="none", choices=["none", "cs_zscore"])
    parser.add_argument("--score_clip", type=float, default=0.0)
    parser.add_argument("--pos_cap", type=float, default=0.0)
    parser.add_argument("--gross_target", type=float, default=1.0)
    parser.add_argument("--score_map", default=None)
    parser.add_argument("--disp_metric", default="std", choices=["std", "p90p10"])
    parser.add_argument("--disp_lo", type=float, default=0.0)
    parser.add_argument("--disp_hi", type=float, default=0.0)
    parser.add_argument("--disp_min_scale", type=float, default=0.0)
    parser.add_argument("--disp_hist_window", type=int, default=0)
    parser.add_argument("--disp_flat_q_low", type=float, default=0.0)
    parser.add_argument("--disp_flat_q_high", type=float, default=0.0)
    parser.add_argument("--disp_scale_q_low", type=float, default=0.0)
    parser.add_argument("--disp_scale_q_high", type=float, default=0.0)
    parser.add_argument("--disp_scale_floor", type=float, default=0.0)
    parser.add_argument("--disp_scale_power", type=float, default=1.0)
    parser.add_argument("--consistency_min", type=float, default=None)
    parser.add_argument("--consistency_scale", type=float, default=0.0)
    parser.add_argument("--disagree_q_low", type=float, default=0.0)
    parser.add_argument("--disagree_q_high", type=float, default=0.0)
    parser.add_argument("--disagree_hist_window", type=int, default=0)
    parser.add_argument("--disagree_scale", type=float, default=0.0)
    parser.add_argument("--shock_hist_window", type=int, default=0)
    parser.add_argument("--shock_q", type=float, default=0.0)
    parser.add_argument("--shock_k", type=float, default=0.0)
    parser.add_argument("--shock_floor", type=float, default=0.0)
    parser.add_argument("--shock_metric", default="median", choices=["median", "p90", "p95"])
    parser.add_argument("--pca_neutral", action="store_true")
    parser.add_argument("--pca_k", type=int, default=3)
    parser.add_argument("--pca_lookback", type=int, default=90)
    parser.add_argument("--pca_min_obs", type=int, default=30)
    parser.add_argument("--topn_cap", type=float, default=0.0)
    parser.add_argument("--topn_n", type=int, default=10)
    parser.add_argument("--topn_cap_low", type=float, default=0.0)
    parser.add_argument("--topn_dyn_q_hi", type=float, default=0.0)
    parser.add_argument("--topn_dyn_q_lo", type=float, default=0.0)
    parser.add_argument("--gate_combine", default="mul", choices=["mul", "min", "avg"])
    parser.add_argument("--gate_avg_weights", default="1,1,1")
    parser.add_argument("--ema_halflife", type=float, default=0.0)
    parser.add_argument("--ema_halflife_min", type=float, default=0.0)
    parser.add_argument("--ema_halflife_max", type=float, default=0.0)
    parser.add_argument("--ema_disp_lo", type=float, default=0.0)
    parser.add_argument("--ema_disp_hi", type=float, default=0.0)
    parser.add_argument("--min_hold", type=int, default=0)
    parser.add_argument("--turnover_cap", type=float, default=0.0)
    parser.add_argument("--turnover_budget", type=float, default=0.0)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--opt_lambda", type=float, default=0.0)
    parser.add_argument("--opt_kappa", type=float, default=0.0)
    parser.add_argument("--opt_steps", type=int, default=20)
    parser.add_argument("--opt_lr", type=float, default=0.0)
    parser.add_argument("--opt_risk_window", type=int, default=0)
    parser.add_argument("--opt_risk_eps", type=float, default=1.0e-6)
    parser.add_argument("--opt_dollar_neutral", action="store_true")
    parser.add_argument("--walk_folds", type=int, default=0)
    parser.add_argument("--min_ic_count", type=int, default=0)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_metrics", default=None)
    parser.add_argument("--oos_last_steps", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("data", {}).get("target_mode", "level")
    use_return_target = target_mode != "level"

    preds = np.load(args.preds)
    y = preds["y"]
    q50 = preds["q50"]
    q50_std = preds["q50_std"] if "q50_std" in preds else None
    mask = preds["mask"] if "mask" in preds else np.isfinite(y).astype(np.float32)
    if mask is None:
        mask = np.isfinite(y).astype(np.float32)
    elif np.sum(mask) == 0:
        print("warning: mask is all zeros; treating all entries as valid")
        mask = np.ones_like(y, dtype=np.float32)
    origin_t = preds["origin_t"].astype(np.int64) if "origin_t" in preds else None
    series_idx = preds["series_idx"].astype(np.int64) if "series_idx" in preds else None

    series_list = _load_series(cfg)
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
    risk_cache = None
    if args.optimize and args.opt_risk_window > 1:
        risk_cache = []
        w = int(args.opt_risk_window)
        for series in series_list:
            y_hist = series.y.astype(np.float64)
            if y_hist.ndim > 1:
                y_hist = y_hist[:, 0]
            y_hist = np.nan_to_num(y_hist, nan=0.0)
            r = np.diff(y_hist)
            if r.size == 0:
                risk_cache.append(np.full_like(y_hist, 1.0, dtype=np.float64))
                continue
            csum = np.cumsum(r)
            csum2 = np.cumsum(r * r)
            sigma = np.zeros_like(y_hist, dtype=np.float64)
            for t in range(y_hist.size):
                start = max(0, t - w)
                end = min(r.size, t)
                if end <= start:
                    sigma[t] = np.std(r[:end]) if end > 0 else 0.0
                    continue
                n = end - start
                s1 = csum[end - 1] - (csum[start - 1] if start > 0 else 0.0)
                s2 = csum2[end - 1] - (csum2[start - 1] if start > 0 else 0.0)
                mean = s1 / max(n, 1)
                var = max(s2 / max(n, 1) - mean * mean, 0.0)
                sigma[t] = np.sqrt(var)
            risk_cache.append(sigma)
    shock_cache = None
    if (args.shock_q and args.shock_q > 0) or (args.topn_dyn_q_hi and args.topn_dyn_q_hi > 0):
        shock_cache = []
        for series in series_list:
            y_hist = series.y.astype(np.float64)
            if y_hist.ndim > 1:
                y_hist = y_hist[:, 0]
            y_hist = np.nan_to_num(y_hist, nan=0.0)
            r = np.diff(y_hist, prepend=np.nan)
            shock_cache.append(r)
    return_cache = None
    if args.pca_neutral:
        return_cache = []
        for series in series_list:
            y_hist = series.y.astype(np.float64)
            if y_hist.ndim > 1:
                y_hist = y_hist[:, 0]
            y_hist = np.nan_to_num(y_hist, nan=0.0)
            r = np.diff(y_hist, prepend=np.nan)
            return_cache.append(r)
    if args.value_scale == "orig":
        if use_return_target:
            y = pre.inverse_return(y)
            q50 = pre.inverse_return(q50)
        else:
            y = pre.inverse_y(y)
            q50 = pre.inverse_y(q50)

    h_idx = args.h - 1
    valid = mask[:, h_idx] > 0
    mu = q50[:, h_idx]
    mu_std = q50_std[:, h_idx] if q50_std is not None else None
    ret = y[:, h_idx]
    if args.oos_last_steps and args.oos_last_steps > 0:
        valid_times = np.unique(origin_t[valid])
        if valid_times.size == 0:
            print("oos_last_steps: no valid times available")
        else:
            n = min(int(args.oos_last_steps), valid_times.size)
            sel_times = valid_times[-n:]
            valid = valid & np.isin(origin_t, sel_times)
            print(f"oos_last_steps={n} time_min={int(sel_times[0])} time_max={int(sel_times[-1])}")

    if origin_t is None or series_idx is None:
        raise ValueError("preds must include origin_t and series_idx for portfolio backtest.")

    # compute time_key
    if args.group_by == "timestamp":
        time_key = np.zeros_like(origin_t, dtype=np.int64)
        for i, (s_idx, t) in enumerate(zip(series_idx, origin_t)):
            s = series_list[int(s_idx)]
            if s.timestamps is None:
                time_key[i] = int(t)
            else:
                time_key[i] = int(np.asarray(s.timestamps[int(t)]).astype("datetime64[ns]").astype(np.int64))
    else:
        time_key = origin_t

    n_series = int(np.max(series_idx)) + 1
    prev_w = np.zeros(n_series, dtype=np.float64)
    hold = np.zeros(n_series, dtype=np.int64)
    ema_state = np.zeros(n_series, dtype=np.float64)
    prev_scores = np.full(n_series, np.nan, dtype=np.float64)
    disp_hist = []
    disagree_hist = []
    gate_on = True
    disagree_on = True
    gated_flat = 0
    gated_consistency = 0
    gated_disagree = 0
    gate_scale_sum = 0.0
    gate_scale_count = 0
    wD, wU, wS = _parse_gate_weights(args.gate_avg_weights)
    use_dynamic_ema = (
        args.ema_halflife_min > 0
        and args.ema_halflife_max > 0
        and args.ema_disp_hi > args.ema_disp_lo
    )
    if args.ema_halflife and args.ema_halflife > 0:
        alpha = 1.0 - 0.5 ** (1.0 / args.ema_halflife)
    else:
        alpha = None
    score_edges = None
    score_vals = None
    if args.score_map:
        score_edges, score_vals = _load_score_map(Path(args.score_map))
    pnl_time = []
    hhi_time = []
    top10_time = []
    turnover_time = []
    gross_time = []
    vol_mkt_time = []
    shock_hist = []
    shock_scale_sum = 0.0
    shock_scale_count = 0
    topn_cap_used = []
    topn_riskoff = False
    pca_exposure_abs = []
    total_times = 0
    used_times = 0
    for t in np.unique(time_key):
        idx = np.where((time_key == t) & valid)[0]
        if idx.size == 0:
            continue
        total_times += 1
        if args.min_ic_count and idx.size < args.min_ic_count:
            continue
        used_times += 1
        mu_t = mu[idx].astype(np.float64, copy=True)
        ret_t = ret[idx].astype(np.float64, copy=True)
        ret_t_raw = ret_t.copy()
        if args.use_cs:
            mu_t = mu_t - np.mean(mu_t)
        if args.ret_cs:
            ret_t = ret_t - np.mean(ret_t)
        assets = series_idx[idx]
        if args.score_transform == "cs_zscore":
            std = np.std(mu_t)
            if std > 1e-12:
                mu_t = (mu_t - np.mean(mu_t)) / std
        if args.score_clip and args.score_clip > 0:
            mu_t = np.clip(mu_t, -args.score_clip, args.score_clip)
        if score_edges is not None and score_vals is not None:
            mu_t = _apply_score_map(mu_t, score_edges, score_vals)
        # dispersion (before EMA)
        if args.disp_metric == "p90p10":
            disp = float(np.percentile(mu_t, 90) - np.percentile(mu_t, 10))
        else:
            disp = float(np.std(mu_t))
        if args.disp_hist_window > 0:
            hist = disp_hist[-args.disp_hist_window :]
        else:
            hist = disp_hist
        if args.disp_flat_q_low > 0:
            if hist:
                q_low = float(np.quantile(hist, args.disp_flat_q_low))
                q_high = float(np.quantile(hist, args.disp_flat_q_high)) if args.disp_flat_q_high > 0 else q_low
                if gate_on and disp <= q_low:
                    gate_on = False
                elif (not gate_on) and disp >= q_high:
                    gate_on = True
            disp_hist.append(disp)
        else:
            disp_hist.append(disp)
        # tail-shock scaling (uses last observed return)
        m_shock = 1.0
        if shock_cache is not None and ((args.shock_q and args.shock_q > 0) or (args.topn_dyn_q_hi and args.topn_dyn_q_hi > 0)):
            shock_vals = []
            for a, t0 in zip(assets, origin_t[idx]):
                if a < len(shock_cache):
                    r = shock_cache[a]
                    if 0 <= t0 < r.size:
                        v = r[int(t0)]
                        if np.isfinite(v):
                            shock_vals.append(abs(v))
            if shock_vals:
                if args.shock_metric == "p90":
                    shock_t = float(np.percentile(shock_vals, 90))
                elif args.shock_metric == "p95":
                    shock_t = float(np.percentile(shock_vals, 95))
                else:
                    shock_t = float(np.median(shock_vals))
            else:
                shock_t = 0.0
            hist_s = shock_hist[-args.shock_hist_window :] if args.shock_hist_window > 0 else shock_hist
            if hist_s:
                q = float(np.quantile(hist_s, args.shock_q))
                if q > 0 and shock_t > q:
                    frac = (shock_t - q) / (q + 1e-12)
                    m_shock = max(args.shock_floor, 1.0 - args.shock_k * frac)
            shock_hist.append(shock_t)
        # PCA loadings for factor neutralization
        pca_loadings = None
        if args.pca_neutral and return_cache is not None:
            pca_loadings = _compute_pca_loadings(
                assets,
                int(t),
                return_cache,
                args.pca_lookback,
                args.pca_k,
                args.pca_min_obs,
            )
        # disagreement gate (requires q50_std in preds)
        if mu_std is not None and (args.disagree_q_low > 0 or args.disagree_q_high > 0):
            u_t = float(np.mean(mu_std[idx]))
            if args.disagree_hist_window > 0:
                dhist = disagree_hist[-args.disagree_hist_window :]
            else:
                dhist = disagree_hist
            if dhist:
                q_hi = float(np.quantile(dhist, args.disagree_q_high)) if args.disagree_q_high > 0 else None
                q_lo = float(np.quantile(dhist, args.disagree_q_low)) if args.disagree_q_low > 0 else None
                if disagree_on and q_hi is not None and u_t >= q_hi:
                    disagree_on = False
                elif (not disagree_on) and q_lo is not None and u_t <= q_lo:
                    disagree_on = True
            disagree_hist.append(u_t)
        # self-consistency gate
        cons_scale = 1.0
        if args.consistency_min is not None:
            prev = prev_scores[assets]
            valid_prev = np.isfinite(prev)
            if np.sum(valid_prev) >= 2:
                a = mu_t[valid_prev]
                b = prev[valid_prev]
                a = a - np.mean(a)
                b = b - np.mean(b)
                denom = (np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12)
                corr = float(np.sum(a * b) / denom)
                if corr < args.consistency_min:
                    cons_scale = args.consistency_scale
                    if cons_scale <= 0:
                        gated_consistency += 1
        prev_scores[assets] = mu_t
        alpha_t = alpha
        if use_dynamic_ema:
            if disp <= args.ema_disp_lo:
                hl = args.ema_halflife_max
            elif disp >= args.ema_disp_hi:
                hl = args.ema_halflife_min
            else:
                frac = (disp - args.ema_disp_lo) / (args.ema_disp_hi - args.ema_disp_lo)
                hl = args.ema_halflife_max - frac * (args.ema_halflife_max - args.ema_halflife_min)
            alpha_t = 1.0 - 0.5 ** (1.0 / max(hl, 1e-6))
        if alpha_t is not None and alpha_t > 0:
            for j, a in enumerate(assets):
                ema_state[a] = (1.0 - alpha_t) * ema_state[a] + alpha_t * mu_t[j]
                mu_t[j] = ema_state[a]
        # dispersion soft scaling
        d_scale = 1.0
        if args.disp_scale_q_low > 0 and args.disp_scale_q_high > args.disp_scale_q_low:
            hist = disp_hist[-args.disp_hist_window :] if args.disp_hist_window > 0 else disp_hist
            if hist:
                q_low = float(np.quantile(hist, args.disp_scale_q_low))
                q_high = float(np.quantile(hist, args.disp_scale_q_high))
                if q_high > q_low:
                    m = (disp - q_low) / (q_high - q_low)
                    m = float(np.clip(m, 0.0, 1.0))
                    m = m ** max(args.disp_scale_power, 1e-6)
                    d_scale = float(args.disp_scale_floor + (1.0 - args.disp_scale_floor) * m)
        elif args.disp_hi > args.disp_lo:
            if disp <= args.disp_lo:
                d_scale = args.disp_min_scale
            elif disp >= args.disp_hi:
                d_scale = 1.0
            else:
                d_scale = args.disp_min_scale + (disp - args.disp_lo) * (1.0 - args.disp_min_scale) / (args.disp_hi - args.disp_lo)

        # disagreement scaling
        u_scale = 1.0
        if mu_std is not None and (args.disagree_q_low > 0 or args.disagree_q_high > 0):
            u_t = float(np.mean(mu_std[idx]))
            dhist = disagree_hist[-args.disagree_hist_window :] if args.disagree_hist_window > 0 else disagree_hist
            if dhist:
                q_hi = float(np.quantile(dhist, args.disagree_q_high)) if args.disagree_q_high > 0 else None
                q_lo = float(np.quantile(dhist, args.disagree_q_low)) if args.disagree_q_low > 0 else None
                if q_hi is not None and q_lo is not None and q_hi > q_lo:
                    if u_t <= q_lo:
                        u_scale = 1.0
                    elif u_t >= q_hi:
                        u_scale = args.disagree_scale
                    else:
                        frac = (u_t - q_lo) / (q_hi - q_lo)
                        u_scale = 1.0 - frac * (1.0 - args.disagree_scale)
        # self-consistency scaling
        s_scale = cons_scale

        # combine scales
        if args.gate_combine == "min":
            gate_scale = min(d_scale, u_scale, s_scale)
        elif args.gate_combine == "avg":
            gate_scale = wD * d_scale + wU * u_scale + wS * s_scale
        else:
            gate_scale = d_scale * u_scale * s_scale

        if not gate_on:
            gate_scale = 0.0
            gated_flat += 1
        if not disagree_on:
            gated_disagree += 1
        if gate_scale <= 0.0:
            w_full = np.zeros_like(prev_w)
            pnl_time.append(0.0)
            turnover_time.append(float(np.sum(np.abs(w_full - prev_w))))
            prev_w = w_full
            continue

        mu_t = mu_t * gate_scale
        gate_scale_sum += gate_scale
        gate_scale_count += 1
        if args.optimize:
            w = _optimize_weights(
                mu_t,
                prev_w[assets],
                assets,
                int(t),
                risk_cache,
                pca_loadings,
                args,
            )
            w_full = prev_w.copy()
            w_full[assets] = w
        else:
            if args.topk and args.topk > 0:
                k = int(args.topk)
                if idx.size < 2 * k:
                    continue
                order = np.argsort(mu_t)
                bottom = order[:k]
                top = order[-k:]
                w = np.zeros_like(mu_t, dtype=np.float64)
                if args.weight_mode == "abs":
                    w[top] = np.abs(mu_t[top])
                    w[bottom] = -np.abs(mu_t[bottom])
                else:
                    w[top] = mu_t[top]
                    w[bottom] = mu_t[bottom]
            else:
                if args.weight_mode == "abs":
                    w = np.abs(mu_t)
                else:
                    w = mu_t
            denom = np.sum(np.abs(w))
            if denom <= 1e-12:
                continue
            w = w / denom
            w_full = prev_w.copy()
            w_full[assets] = w
        if args.pca_neutral and pca_loadings is not None:
            w_full[assets] = _neutralize_factors(w_full[assets], pca_loadings)
        if args.min_hold and args.min_hold > 0:
            for a in assets:
                if hold[a] > 0:
                    w_full[a] = prev_w[a]
                    hold[a] -= 1
                else:
                    if prev_w[a] == 0.0 and w_full[a] != 0.0:
                        hold[a] = args.min_hold - 1
                    elif prev_w[a] != 0.0 and np.sign(w_full[a]) != np.sign(prev_w[a]):
                        hold[a] = args.min_hold - 1
        if args.turnover_cap and args.turnover_cap > 0:
            delta = w_full - prev_w
            delta = np.clip(delta, -args.turnover_cap, args.turnover_cap)
            w_full = prev_w + delta
        if args.turnover_budget and args.turnover_budget > 0:
            delta = w_full - prev_w
            total = np.sum(np.abs(delta))
            if total > args.turnover_budget:
                scale = args.turnover_budget / (total + 1e-12)
                w_full = prev_w + delta * scale
        if args.pos_cap and args.pos_cap > 0:
            w_full = np.clip(w_full, -args.pos_cap, args.pos_cap)
        denom_full = np.sum(np.abs(w_full[assets]))
        if denom_full > 1e-12:
            w_full = w_full / denom_full
        if args.gross_target and args.gross_target > 0:
            gross = np.sum(np.abs(w_full[assets]))
            target_gross = args.gross_target * gate_scale * m_shock
            if target_gross <= 0.0:
                w_full[assets] = 0.0
            elif gross > 1e-12:
                w_full[assets] = w_full[assets] * (target_gross / gross)
        cap_used = args.topn_cap
        if (
            args.topn_cap
            and args.topn_cap > 0
            and args.topn_cap_low
            and args.topn_cap_low > 0
            and args.topn_dyn_q_hi
            and args.topn_dyn_q_hi > 0
            and shock_hist
        ):
            hist_s = shock_hist[-args.shock_hist_window :] if args.shock_hist_window > 0 else shock_hist
            if hist_s:
                q_hi = float(np.quantile(hist_s, args.topn_dyn_q_hi))
                q_lo = float(np.quantile(hist_s, args.topn_dyn_q_lo)) if args.topn_dyn_q_lo > 0 else q_hi
                if (not topn_riskoff) and shock_t >= q_hi:
                    topn_riskoff = True
                elif topn_riskoff and shock_t <= q_lo:
                    topn_riskoff = False
            cap_used = args.topn_cap_low if topn_riskoff else args.topn_cap
        if args.topn_cap and args.topn_cap > 0:
            w_full[assets] = _apply_topn_cap(
                w_full[assets],
                cap_used,
                args.topn_n,
                args.gross_target * gate_scale * m_shock,
            )
        w_now = w_full[assets]
        pnl_time.append(float(np.sum(w_now * ret_t)))
        gross_time.append(float(np.sum(np.abs(w_now))))
        if ret_t_raw.size:
            vol_mkt_time.append(float(np.median(np.abs(ret_t_raw))))
        if w_now.size:
            hhi_time.append(float(np.sum(w_now * w_now)))
            topk = 10 if w_now.size >= 10 else w_now.size
            if topk > 0:
                top10_time.append(float(np.sum(np.sort(np.abs(w_now))[-topk:])))
            if pca_loadings is not None:
                expo = pca_loadings.T @ w_now
                pca_exposure_abs.append(float(np.mean(np.abs(expo))))
            if args.topn_cap and args.topn_cap > 0:
                topn_cap_used.append(cap_used)
        turnover_time.append(float(np.sum(np.abs(w_full - prev_w))))
        prev_w = w_full
        shock_scale_sum += m_shock
        shock_scale_count += 1

    pnl_time = np.asarray(pnl_time, dtype=np.float64)
    print(
        f"mu_value_weighted h={args.h} mode={args.weight_mode} "
        f"cs={args.use_cs} ret_cs={args.ret_cs} topk={args.topk} n_times={pnl_time.size}"
    )
    _summarize("portfolio", pnl_time)
    if gross_time:
        gross_arr = np.asarray(gross_time, dtype=np.float64)
        pnl_arr = np.asarray(pnl_time, dtype=np.float64)
        mask_g = (gross_arr > 1e-12) & np.isfinite(gross_arr) & np.isfinite(pnl_arr)
        if np.any(mask_g):
            _summarize("gross_norm", pnl_arr[mask_g] / gross_arr[mask_g])
    if vol_mkt_time:
        vol_arr = np.asarray(vol_mkt_time, dtype=np.float64)
        pnl_arr = np.asarray(pnl_time, dtype=np.float64)
        mask_v = (vol_arr > 1e-12) & np.isfinite(vol_arr) & np.isfinite(pnl_arr)
        if np.any(mask_v):
            _summarize("vol_norm", pnl_arr[mask_v] / vol_arr[mask_v])
    if pnl_time.size > 0:
        avg_scale = gate_scale_sum / max(gate_scale_count, 1)
        print(
            f"gate_flat_frac={gated_flat / pnl_time.size:.3f} gate_disagree_frac={gated_disagree / pnl_time.size:.3f} "
            f"gate_consistency_flat_frac={gated_consistency / pnl_time.size:.3f} gate_scale_mean={avg_scale:.3f}"
        )
        if shock_scale_count > 0:
            print(f"shock_scale_mean={shock_scale_sum / max(shock_scale_count, 1):.3f}")
        if pca_exposure_abs:
            exp_arr = np.asarray(pca_exposure_abs, dtype=np.float64)
            print(
                f"pca_exposure_abs_mean={float(np.mean(exp_arr)):.6f} "
                f"pca_exposure_abs_p90={float(np.percentile(exp_arr, 90)):.6f}"
            )
    if topn_cap_used:
        cap_arr = np.asarray(topn_cap_used, dtype=np.float64)
        print(
            f"topn_cap_used_mean={float(np.mean(cap_arr)):.3f} "
            f"topn_riskoff_frac={float(np.mean(cap_arr <= args.topn_cap_low)):.3f}"
        )
    if total_times:
        dropped = total_times - used_times
        print(f"min_ic_count_drop={dropped} used_times={used_times} total_times={total_times}")
    if turnover_time:
        avg_turnover = float(np.mean(turnover_time))
        print(f"turnover_mean={avg_turnover:.6f}")
    if hhi_time:
        hhi_arr = np.asarray(hhi_time, dtype=np.float64)
        top10_arr = np.asarray(top10_time, dtype=np.float64)
        print(
            f"hhi_mean={float(np.mean(hhi_arr)):.6f} hhi_p90={float(np.percentile(hhi_arr, 90)):.6f} "
            f"top10_mean={float(np.mean(top10_arr)):.6f} top10_p90={float(np.percentile(top10_arr, 90)):.6f}"
        )
    if pnl_time.size > 1:
        rho1 = float(np.corrcoef(pnl_time[:-1], pnl_time[1:])[0, 1])
        neff = pnl_time.size * (1.0 - rho1) / (1.0 + rho1 + 1e-12)
        print(f"pnl_rho1={rho1:.4f} pnl_n_eff={neff:.1f}")

    if args.walk_folds and args.walk_folds > 1 and pnl_time.size > 0:
        folds = int(args.walk_folds)
        fold_size = max(1, pnl_time.size // folds)
        print(f"walk_folds={folds} fold_size={fold_size}")
        fold_stats = []
        for i in range(folds):
            start = i * fold_size
            end = pnl_time.size if i == folds - 1 else (i + 1) * fold_size
            chunk = pnl_time[start:end]
            if chunk.size == 0:
                continue
            mean = float(np.mean(chunk))
            std = float(np.std(chunk))
            sharpe = mean / (std + 1e-12)
            hit = float(np.mean(chunk > 0))
            cum = np.cumsum(chunk)
            peak = np.maximum.accumulate(cum)
            max_dd = float(np.max(peak - cum)) if cum.size else float("nan")
            fold_stats.append((mean, sharpe))
            print(f"fold{i}: mean={mean:.6f} sharpe={sharpe:.3f} hit={hit:.3f} max_dd={max_dd:.6f} n={chunk.size}")
        if fold_stats:
            means = [m for m, _ in fold_stats]
            sharpes = [s for _, s in fold_stats]
            print(f"fold_mean_std={np.std(means):.6f} fold_sharpe_std={np.std(sharpes):.6f}")
            total_pnl = float(np.sum(pnl_time))
            best_fold = float(np.max([np.sum(pnl_time[i*fold_size:(pnl_time.size if i == folds - 1 else (i+1)*fold_size)]) for i in range(folds)]))
            if total_pnl != 0:
                print(f"pnl_dominance={best_fold/total_pnl:.3f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("pnl\n")
            for v in pnl_time:
                f.write(f"{v:.8f}\n")
    if args.out_metrics:
        out_path = Path(args.out_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("pnl,hhi,top10,gross,vol_mkt\n")
            for i, v in enumerate(pnl_time):
                h = hhi_time[i] if i < len(hhi_time) else float("nan")
                t = top10_time[i] if i < len(top10_time) else float("nan")
                g = gross_time[i] if i < len(gross_time) else float("nan")
                vm = vol_mkt_time[i] if i < len(vol_mkt_time) else float("nan")
                f.write(f"{v:.8f},{h:.8f},{t:.8f},{g:.8f},{vm:.8f}\n")


def _load_score_map(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    edges = data[:, 0:2]
    values = data[:, 2]
    return edges, values


def _apply_score_map(scores: np.ndarray, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float64)
    for i, s in enumerate(scores):
        idx = np.where((s >= edges[:, 0]) & (s < edges[:, 1]))[0]
        if idx.size == 0:
            idx = np.array([np.argmax(edges[:, 1])])
        out[i] = values[int(idx[0])]
    return out


def _parse_gate_weights(raw: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        return 1.0, 1.0, 1.0
    vals = [float(p) for p in parts]
    total = sum(vals)
    if total <= 0:
        return 1.0, 1.0, 1.0
    return vals[0] / total, vals[1] / total, vals[2] / total


def _soft_threshold(x: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def _project_weights(
    w: np.ndarray,
    cap: float,
    gross_target: float,
    dollar_neutral: bool,
) -> np.ndarray:
    out = w.copy()
    if dollar_neutral:
        out = out - np.mean(out)
    if cap and cap > 0:
        out = np.clip(out, -cap, cap)
    gross = np.sum(np.abs(out))
    if gross_target and gross > 1e-12:
        out = out * (gross_target / gross)
    if dollar_neutral:
        out = out - np.mean(out)
    if cap and cap > 0:
        out = np.clip(out, -cap, cap)
    gross = np.sum(np.abs(out))
    if gross_target and gross > 1e-12:
        out = out * (gross_target / gross)
    return out


def _optimize_weights(
    alpha: np.ndarray,
    w_prev: np.ndarray,
    assets: np.ndarray,
    t_idx: int,
    risk_cache: list | None,
    pca_loadings: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    n = alpha.size
    if n == 0:
        return alpha
    sigma2 = np.ones(n, dtype=np.float64)
    if risk_cache is not None:
        sig = np.zeros(n, dtype=np.float64)
        for j, a in enumerate(assets):
            if a < len(risk_cache) and t_idx < risk_cache[a].size:
                sig[j] = risk_cache[a][t_idx]
        if np.all(sig == 0):
            sig = np.ones_like(sig)
        fallback = np.nanmedian(sig) if np.isfinite(sig).any() else 1.0
        sig = np.nan_to_num(sig, nan=fallback, posinf=fallback, neginf=fallback)
        sigma2 = np.maximum(sig * sig, args.opt_risk_eps)
    lam = max(args.opt_lambda, 0.0)
    kappa = max(args.opt_kappa, 0.0)
    if args.opt_lr and args.opt_lr > 0:
        lr = args.opt_lr
    else:
        denom = 2.0 * lam * float(np.max(sigma2)) + 1.0
        lr = 1.0 / denom
    w = w_prev.copy()
    for _ in range(max(args.opt_steps, 1)):
        grad = alpha - 2.0 * lam * sigma2 * w
        w = w + lr * grad
        if kappa > 0:
            w = w_prev + _soft_threshold(w - w_prev, kappa * lr)
        if pca_loadings is not None:
            w = _neutralize_factors(w, pca_loadings)
        w = _project_weights(w, args.pos_cap, args.gross_target, args.opt_dollar_neutral)
    return w


def _neutralize_factors(w: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    if loadings is None or loadings.size == 0:
        return w
    try:
        coeff, *_ = np.linalg.lstsq(loadings, w, rcond=None)
        return w - loadings @ coeff
    except np.linalg.LinAlgError:
        return w


def _compute_pca_loadings(
    assets: np.ndarray,
    t_idx: int,
    return_cache: list,
    lookback: int,
    k: int,
    min_obs: int,
) -> np.ndarray | None:
    if k <= 0 or lookback <= 1:
        return None
    n = assets.size
    if n <= k:
        return None
    start = max(1, t_idx - lookback + 1)
    end = t_idx + 1
    T = end - start
    if T < min_obs:
        return None
    X = np.zeros((T, n), dtype=np.float64)
    valid_cols = 0
    for j, a in enumerate(assets):
        if a >= len(return_cache):
            continue
        r = return_cache[a]
        if end > r.size:
            continue
        col = r[start:end].astype(np.float64, copy=True)
        mask = np.isfinite(col)
        if np.sum(mask) < max(5, min_obs // 3):
            continue
        mean = np.nanmean(col)
        std = np.nanstd(col)
        if std <= 1e-12:
            continue
        col = (col - mean) / std
        col = np.nan_to_num(col, nan=0.0)
        X[:, j] = col
        valid_cols += 1
    if valid_cols <= k:
        return None
    X = X - np.mean(X, axis=0, keepdims=True)
    if not np.isfinite(X).all():
        return None
    if np.all(np.abs(X) < 1e-12):
        return None
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    loadings = vt[:k].T
    return loadings


def _apply_topn_cap(
    w: np.ndarray,
    cap: float,
    topn: int,
    target_gross: float,
) -> np.ndarray:
    if cap <= 0 or w.size == 0:
        return w
    n = min(int(topn), w.size)
    if n <= 0:
        return w
    absw = np.abs(w)
    total = float(np.sum(absw))
    if total <= 1e-12:
        return w
    idx = np.argpartition(absw, -n)[-n:]
    top_sum = float(np.sum(absw[idx]))
    if top_sum <= cap:
        return w
    scale_top = cap / max(top_sum, 1e-12)
    out = w.copy()
    out[idx] = out[idx] * scale_top
    rest_mask = np.ones_like(w, dtype=bool)
    rest_mask[idx] = False
    rest_sum = float(np.sum(np.abs(out[rest_mask])))
    if target_gross > 0 and rest_sum > 1e-12:
        remaining = max(target_gross - cap, 0.0)
        scale_rest = remaining / rest_sum
        out[rest_mask] = out[rest_mask] * scale_rest
    return out


if __name__ == "__main__":
    main()
