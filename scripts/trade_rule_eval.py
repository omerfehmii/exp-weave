from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import compress_series_observed, load_panel_npz
from eval import apply_scaling
from utils import load_config


def _parse_horizons(value: str, H: int) -> List[int]:
    if not value:
        return list(range(1, H + 1))
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return [h for h in out if 1 <= h <= H]


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    try:
        erf = np.erf
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    except AttributeError:
        # Fallback for older numpy.
        from math import erf as _erf

        return 0.5 * (1.0 + np.vectorize(_erf)(x / np.sqrt(2.0)))


def _load_series(cfg: Dict) -> list:
    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    return series_list


def _origin_values(series_list: list, series_idx: np.ndarray, origin_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.zeros(series_idx.shape[0], dtype=np.float32)
    origin_mask = np.ones_like(y0, dtype=np.float32)
    for i, (s_idx, t) in enumerate(zip(series_idx, origin_t)):
        s = series_list[int(s_idx)]
        y = s.y
        if y.ndim > 1:
            y = y[:, 0]
        y0[i] = float(y[int(t)])
        if s.mask is not None:
            m = s.mask
            if m.ndim > 1:
                m = m[:, 0]
            origin_mask[i] = float(m[int(t)])
    return y0, origin_mask


def _select_scores(
    score_mode: str,
    mu_r: np.ndarray,
    p_plus: np.ndarray | None,
    p_minus: np.ndarray | None,
) -> np.ndarray:
    if score_mode in {"mean", "q50"}:
        return mu_r
    if score_mode in {"prob_edge", "edge_prob"}:
        if p_plus is None or p_minus is None:
            raise ValueError("prob_edge requires p_plus/p_minus.")
        return p_plus - p_minus
    raise ValueError(f"Unknown score_mode: {score_mode}")


def _assign_rank_sides(
    scores: np.ndarray,
    times: np.ndarray,
    valid_mask: np.ndarray,
    topk: int,
    rank_mode: str,
) -> np.ndarray:
    sides = np.zeros_like(scores, dtype=np.int8)
    if topk <= 0:
        return sides
    uniq = np.unique(times)
    for t in uniq:
        idx = np.where((times == t) & valid_mask)[0]
        if rank_mode == "long_short":
            if idx.size < 2 * topk:
                continue
        else:
            if idx.size < topk:
                continue
        order = np.argsort(scores[idx])
        if rank_mode in {"long_short", "short_only"}:
            bottom = idx[order[:topk]]
            sides[bottom] = -1
        if rank_mode in {"long_short", "long_only"}:
            top = idx[order[-topk:]]
            sides[top] = 1
    return sides


def _assign_rank_buckets(
    scores: np.ndarray,
    times: np.ndarray,
    valid_mask: np.ndarray,
    topk: int,
    rank_mode: str,
) -> np.ndarray:
    buckets = np.zeros_like(scores, dtype=np.int8)
    if topk <= 0:
        return buckets
    uniq = np.unique(times)
    for t in uniq:
        idx = np.where((times == t) & valid_mask)[0]
        if rank_mode == "long_short":
            if idx.size < 2 * topk:
                continue
        else:
            if idx.size < topk:
                continue
        order = np.argsort(scores[idx])
        if rank_mode in {"long_short", "short_only"}:
            bottom = idx[order[:topk]]
            if bottom.size:
                buckets[bottom[0]] = 1
                if bottom.size > 1:
                    buckets[bottom[1:]] = 2
        if rank_mode in {"long_short", "long_only"}:
            top = idx[order[-topk:][::-1]]
            if top.size:
                buckets[top[0]] = 1
                if top.size > 1:
                    buckets[top[1:]] = 2
    return buckets


def _apply_score_transform(
    scores: np.ndarray,
    valid: np.ndarray,
    series_idx: np.ndarray,
    time_key: np.ndarray,
    mode: str,
    ts_lookback: int,
) -> np.ndarray:
    if mode == "none":
        return scores
    out = scores.copy()
    if mode in {"cs_demean", "cs_zscore"}:
        uniq = np.unique(time_key)
        for t in uniq:
            idx = np.where((time_key == t) & valid)[0]
            if idx.size == 0:
                continue
            mean = float(np.mean(scores[idx]))
            if mode == "cs_demean":
                out[idx] = scores[idx] - mean
            else:
                std = float(np.std(scores[idx]))
                if std < 1e-8:
                    std = 1.0
                out[idx] = (scores[idx] - mean) / std
        return out
    if mode == "ts_zscore":
        order = np.lexsort((time_key, series_idx))
        unique_assets = np.unique(series_idx)
        for asset in unique_assets:
            idx = order[series_idx[order] == asset]
            if idx.size == 0:
                continue
            for j, gidx in enumerate(idx):
                if ts_lookback > 0:
                    start = max(0, j - ts_lookback + 1)
                else:
                    start = 0
                window = idx[start : j + 1]
                window = window[valid[window]]
                if window.size == 0:
                    out[gidx] = 0.0
                    continue
                wvals = scores[window]
                mean = float(np.mean(wvals))
                std = float(np.std(wvals))
                if std < 1e-8:
                    std = 1.0
                out[gidx] = (scores[gidx] - mean) / std
        return out
    raise ValueError(f"Unknown score_transform: {mode}")


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _center_scale(x: np.ndarray, stat: str, scale: str) -> Tuple[float, float]:
    if stat == "median":
        center = float(np.median(x))
    else:
        center = float(np.mean(x))
    if scale == "mad":
        sc = _mad(x)
    else:
        sc = float(np.std(x))
    if sc < 1e-8:
        sc = 0.0
    return center, sc


def _realized_vol(
    series: np.ndarray,
    mask: np.ndarray | None,
    t: int,
    lookback: int,
    return_mode: str,
    scale_mode: str,
    floor: float,
) -> float:
    if t <= 0:
        return floor
    if series.ndim > 1:
        series = series[:, 0]
    if mask is not None and mask.ndim > 1:
        mask = mask[:, 0]
    start = max(1, t - lookback + 1)
    seg = series[start - 1 : t + 1]
    if seg.size < 2:
        return floor
    if return_mode == "pct":
        prev = seg[:-1]
        denom = np.maximum(np.abs(prev), 1e-8)
        ret = (seg[1:] - prev) / denom
    elif return_mode == "log":
        prev = seg[:-1]
        curr = seg[1:]
        valid = (prev > 0) & (curr > 0)
        ret = np.zeros_like(curr, dtype=np.float32)
        ret[valid] = np.log(curr[valid]) - np.log(prev[valid])
        ret = ret[valid]
    else:
        ret = np.diff(seg)
    if mask is not None:
        m_prev = mask[start - 1 : t]
        m_curr = mask[start : t + 1]
        valid = (m_prev > 0) & (m_curr > 0)
        ret = ret[valid]
    if ret.size == 0:
        return floor
    if scale_mode == "mad":
        sc = _mad(ret)
    else:
        sc = float(np.std(ret))
    if sc < floor:
        return floor
    return float(sc)


def _compute_time_metrics(
    origin_t: np.ndarray, pnl: np.ndarray, gross_exposure: np.ndarray | None = None
) -> Tuple[float, float, float, float, float, float, float]:
    if origin_t.size == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    uniq, inv = np.unique(origin_t, return_inverse=True)
    pnl_time = np.zeros(uniq.shape[0], dtype=np.float64)
    np.add.at(pnl_time, inv, pnl.astype(np.float64))
    mean_t = float(np.mean(pnl_time))
    std_t = float(np.std(pnl_time))
    sharpe_t = mean_t / (std_t + 1e-12)
    cum = np.cumsum(pnl_time)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum)) if cum.size else float("nan")
    mean_t_norm = float("nan")
    std_t_norm = float("nan")
    sharpe_t_norm = float("nan")
    if gross_exposure is not None:
        gross_time = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(gross_time, inv, gross_exposure.astype(np.float64))
        pnl_time_norm = np.zeros_like(pnl_time)
        active = gross_time > 1e-12
        pnl_time_norm[active] = pnl_time[active] / gross_time[active]
        mean_t_norm = float(np.mean(pnl_time_norm))
        std_t_norm = float(np.std(pnl_time_norm))
        sharpe_t_norm = mean_t_norm / (std_t_norm + 1e-12)
    return mean_t, std_t, sharpe_t, max_dd, mean_t_norm, std_t_norm, sharpe_t_norm


def _apply_hold_and_flip(
    side: np.ndarray,
    valid: np.ndarray,
    series_idx: np.ndarray,
    time_key_hours: np.ndarray,
    hold_min_hours: int,
    flip_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if hold_min_hours <= 0 and flip_penalty <= 0.0:
        return side, np.zeros_like(side, dtype=np.float32)
    extra_cost = np.zeros_like(side, dtype=np.float32)
    last_time: Dict[int, int] = {}
    last_side: Dict[int, int] = {}
    order = np.lexsort((time_key_hours, series_idx))
    for idx in order:
        if not valid[idx]:
            continue
        s_idx = int(series_idx[idx])
        t = int(time_key_hours[idx])
        if hold_min_hours > 0:
            lt = last_time.get(s_idx)
            if lt is not None and (t - lt) < hold_min_hours:
                side[idx] = 0
                continue
        if side[idx] != 0:
            ls = last_side.get(s_idx, 0)
            if flip_penalty > 0.0 and ls != 0 and side[idx] != ls:
                extra_cost[idx] = flip_penalty
            last_time[s_idx] = t
            last_side[s_idx] = int(side[idx])
    return side, extra_cost


def _normalize_rank_exposure(
    size: np.ndarray,
    side: np.ndarray,
    valid: np.ndarray,
    time_key: np.ndarray,
    target_gross: float,
) -> np.ndarray:
    trade = (valid > 0) & (side != 0)
    if not np.any(trade):
        return size
    out = size.copy()
    times = time_key[trade]
    uniq, inv = np.unique(times, return_inverse=True)
    gross = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(gross, inv, np.abs(out[trade] * side[trade]))
    scale = np.ones_like(gross)
    scale[gross > 0] = target_gross / gross[gross > 0]
    out[trade] = out[trade] * scale[inv]
    return out


def _time_delta_stats(time_key_hours: np.ndarray, valid: np.ndarray) -> Tuple[float, float, float]:
    if not np.any(valid):
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(time_key_hours[valid])
    if uniq.size < 2:
        return float("nan"), float("nan"), float("nan")
    diffs = np.diff(np.sort(uniq)).astype(np.float64)
    return float(np.min(diffs)), float(np.median(diffs)), float(np.max(diffs))


def _weight_matrix(
    weights: np.ndarray,
    series_idx: np.ndarray,
    time_key_hours: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = valid > 0
    if not np.any(mask):
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64)
    times = time_key_hours[mask]
    uniq_times, inv = np.unique(times, return_inverse=True)
    n_times = uniq_times.shape[0]
    n_series = int(np.max(series_idx)) + 1 if series_idx.size else 0
    mat = np.zeros((n_times, n_series), dtype=np.float32)
    w = weights[mask]
    s_idx = series_idx[mask].astype(np.int64)
    np.add.at(mat, (inv, s_idx), w)
    return mat, uniq_times


def _compute_weight_metrics(
    weights: np.ndarray,
    series_idx: np.ndarray,
    time_key_hours: np.ndarray,
    valid: np.ndarray,
) -> Tuple[float, float, float, float]:
    mat, times = _weight_matrix(weights, series_idx, time_key_hours, valid)
    if mat.shape[0] < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = np.diff(mat, axis=0)
    turnover_sum = float(np.sum(np.abs(diff)))
    sign = np.sign(mat)
    flip_rates = []
    jaccards = []
    for i in range(1, sign.shape[0]):
        prev = sign[i - 1]
        curr = sign[i]
        active = (prev != 0) | (curr != 0)
        denom = int(np.sum(active))
        if denom:
            flips = int(np.sum((prev * curr) < 0))
            flip_rates.append(flips / denom)
        active_prev = prev != 0
        active_curr = curr != 0
        union = int(np.sum(active_prev | active_curr))
        if union:
            inter = int(np.sum(active_prev & active_curr))
            jaccards.append(inter / union)
    flip_rate = float(np.mean(flip_rates)) if flip_rates else float("nan")
    jaccard = float(np.mean(jaccards)) if jaccards else float("nan")

    hold_durations = []
    for s in range(sign.shape[1]):
        s_sign = sign[:, s]
        if not np.any(s_sign != 0):
            continue
        start_idx = 0
        curr = s_sign[0]
        for i in range(1, s_sign.shape[0]):
            if s_sign[i] == curr:
                continue
            if curr != 0:
                hold_durations.append(float(times[i] - times[start_idx]))
            curr = s_sign[i]
            start_idx = i
        if curr != 0:
            hold_durations.append(float(times[-1] - times[start_idx]))
    avg_holding = float(np.mean(hold_durations)) if hold_durations else float("nan")

    return turnover_sum, flip_rate, avg_holding, jaccard


def _summarize(
    scope: str,
    valid_mask: np.ndarray,
    side: np.ndarray,
    size: np.ndarray,
    pnl: np.ndarray,
    origin_t: np.ndarray,
    gross_pnl: np.ndarray | None = None,
    cost_pnl: np.ndarray | None = None,
    edge_prob: np.ndarray | None = None,
    time_key_hours: np.ndarray | None = None,
    rank_bucket: np.ndarray | None = None,
    series_idx: np.ndarray | None = None,
    hold_hours: np.ndarray | None = None,
) -> Dict[str, float | str]:
    valid = valid_mask > 0
    trade = valid & (side != 0)
    n_valid = int(np.sum(valid))
    n_trades = int(np.sum(trade))
    trade_rate = n_trades / max(n_valid, 1)
    hit_rate = float(np.mean(pnl[trade] > 0)) if n_trades else float("nan")
    mean_pnl = float(np.mean(pnl[trade])) if n_trades else float("nan")
    std_pnl = float(np.std(pnl[trade])) if n_trades else float("nan")
    sharpe_trade = mean_pnl / (std_pnl + 1e-12) if n_trades else float("nan")
    avg_size = float(np.mean(size[trade])) if n_trades else float("nan")
    gross_exposure = np.abs(size * side)
    (
        mean_t,
        std_t,
        sharpe_t,
        max_dd,
        mean_t_norm,
        std_t_norm,
        sharpe_t_norm,
    ) = _compute_time_metrics(origin_t[valid], pnl[valid], gross_exposure[valid])
    avg_hold = float(np.mean(hold_hours[trade])) if hold_hours is not None and n_trades else float("nan")
    trades_per_time = float("nan")
    turnover = float("nan")
    turnover_sum = float("nan")
    flip_rate = float("nan")
    avg_holding_time = float("nan")
    jaccard = float("nan")
    net_pnl_sum = float(np.sum(pnl[trade])) if n_trades else 0.0
    gross_pnl_sum = float("nan")
    cost_pnl_sum = float("nan")
    turnover_tax = float("nan")
    if gross_pnl is not None:
        gross_pnl_sum = float(np.sum(gross_pnl[trade])) if n_trades else 0.0
    if cost_pnl is not None:
        cost_pnl_sum = float(np.sum(cost_pnl[trade])) if n_trades else 0.0
    if not np.isnan(gross_pnl_sum):
        turnover_tax = cost_pnl_sum / max(abs(gross_pnl_sum), 1e-12)
    edge_p10 = float("nan")
    edge_p50 = float("nan")
    edge_p90 = float("nan")
    if edge_prob is not None and n_trades:
        edge_vals = edge_prob[trade]
        edge_p10 = float(np.nanquantile(edge_vals, 0.10))
        edge_p50 = float(np.nanquantile(edge_vals, 0.50))
        edge_p90 = float(np.nanquantile(edge_vals, 0.90))
    gross_p10 = float("nan")
    gross_p50 = float("nan")
    gross_p90 = float("nan")
    gross_mean = float("nan")
    if n_trades:
        if hold_hours is not None:
            group_key = np.stack((origin_t[trade], hold_hours[trade]), axis=1)
            _, inv = np.unique(group_key, axis=0, return_inverse=True)
        else:
            times_trade = origin_t[trade]
            _, inv = np.unique(times_trade, return_inverse=True)
        gross_per_time = np.zeros(int(np.max(inv)) + 1, dtype=np.float64)
        np.add.at(gross_per_time, inv, np.abs(size[trade] * side[trade]))
        gross_p10 = float(np.quantile(gross_per_time, 0.10))
        gross_p50 = float(np.quantile(gross_per_time, 0.50))
        gross_p90 = float(np.quantile(gross_per_time, 0.90))
        gross_mean = float(np.mean(gross_per_time))
    time_delta_min = float("nan")
    time_delta_med = float("nan")
    time_delta_max = float("nan")
    if time_key_hours is not None:
        time_delta_min, time_delta_med, time_delta_max = _time_delta_stats(time_key_hours, valid)
    if origin_t is not None and valid.any():
        times = origin_t[valid]
        uniq_times = np.unique(times)
        n_times = max(len(uniq_times), 1)
        trades_per_time = n_trades / n_times
        if n_trades:
            inv = np.searchsorted(uniq_times, origin_t[trade])
            abs_size = np.zeros(len(uniq_times), dtype=np.float64)
            np.add.at(abs_size, inv, np.abs(size[trade]))
            turnover = float(np.mean(abs_size))
    if time_key_hours is not None and series_idx is not None:
        weights = side.astype(np.float32) * size
        turnover_sum, flip_rate, avg_holding_time, jaccard = _compute_weight_metrics(
            weights,
            series_idx,
            time_key_hours,
            valid,
        )
    macro_pnl_trade = float("nan")
    if series_idx is not None and n_trades:
        series = series_idx[trade]
        uniq_series, inv = np.unique(series, return_inverse=True)
        pnl_sum = np.zeros(len(uniq_series), dtype=np.float64)
        counts = np.zeros(len(uniq_series), dtype=np.float64)
        np.add.at(pnl_sum, inv, pnl[trade])
        np.add.at(counts, inv, 1.0)
        per_series = pnl_sum / np.maximum(counts, 1.0)
        macro_pnl_trade = float(np.mean(per_series))
    rank1_gross = float("nan")
    rank1_cost = float("nan")
    rank1_net = float("nan")
    rank2_gross = float("nan")
    rank2_cost = float("nan")
    rank2_net = float("nan")
    rank1_trades = 0
    rank2_trades = 0
    if rank_bucket is not None and n_trades:
        rank1_mask = trade & (rank_bucket == 1)
        rank2_mask = trade & (rank_bucket == 2)
        rank1_trades = int(np.sum(rank1_mask))
        rank2_trades = int(np.sum(rank2_mask))
        if gross_pnl is not None and cost_pnl is not None:
            rank1_gross = float(np.sum(gross_pnl[rank1_mask])) if rank1_trades else 0.0
            rank1_cost = float(np.sum(cost_pnl[rank1_mask])) if rank1_trades else 0.0
            rank1_net = float(np.sum(pnl[rank1_mask])) if rank1_trades else 0.0
            rank2_gross = float(np.sum(gross_pnl[rank2_mask])) if rank2_trades else 0.0
            rank2_cost = float(np.sum(cost_pnl[rank2_mask])) if rank2_trades else 0.0
            rank2_net = float(np.sum(pnl[rank2_mask])) if rank2_trades else 0.0
    return {
        "scope": scope,
        "n_valid": n_valid,
        "n_trades": n_trades,
        "trade_rate": trade_rate,
        "hit_rate": hit_rate,
        "mean_pnl_trade": mean_pnl,
        "std_pnl_trade": std_pnl,
        "sharpe_trade": sharpe_trade,
        "mean_pnl_time": mean_t,
        "std_pnl_time": std_t,
        "sharpe_time": sharpe_t,
        "mean_pnl_time_norm": mean_t_norm,
        "std_pnl_time_norm": std_t_norm,
        "sharpe_time_norm": sharpe_t_norm,
        "max_dd": max_dd,
        "gross_pnl_sum": gross_pnl_sum,
        "cost_pnl_sum": cost_pnl_sum,
        "net_pnl_sum": net_pnl_sum,
        "turnover_tax": turnover_tax,
        "gross_exposure_mean": gross_mean,
        "gross_exposure_p10": gross_p10,
        "gross_exposure_p50": gross_p50,
        "gross_exposure_p90": gross_p90,
        "turnover_sum": turnover_sum,
        "flip_rate": flip_rate,
        "avg_holding_time": avg_holding_time,
        "jaccard": jaccard,
        "edge_prob_p10": edge_p10,
        "edge_prob_p50": edge_p50,
        "edge_prob_p90": edge_p90,
        "time_delta_min_hours": time_delta_min,
        "time_delta_median_hours": time_delta_med,
        "time_delta_max_hours": time_delta_max,
        "rank1_trades": rank1_trades,
        "rank1_gross_pnl_sum": rank1_gross,
        "rank1_cost_pnl_sum": rank1_cost,
        "rank1_net_pnl_sum": rank1_net,
        "rank2plus_trades": rank2_trades,
        "rank2plus_gross_pnl_sum": rank2_gross,
        "rank2plus_cost_pnl_sum": rank2_cost,
        "rank2plus_net_pnl_sum": rank2_net,
        "avg_size": avg_size,
        "avg_hold_time": avg_hold,
        "trades_per_time": trades_per_time,
        "turnover": turnover,
        "macro_pnl_trade": macro_pnl_trade,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--out_trades", default=None)
    parser.add_argument("--rule", default="prob_cost", choices=["prob_cost", "conformal_bounds", "rank", "ts"])
    parser.add_argument("--score_mode", default="mean", choices=["mean", "prob_edge", "q50", "edge_prob"])
    parser.add_argument("--horizons", default="")
    parser.add_argument("--return_mode", default="diff", choices=["diff", "pct", "log"])
    parser.add_argument("--value_scale", default="orig", choices=["orig", "scaled"])
    parser.add_argument("--tau", type=float, default=0.60)
    parser.add_argument("--move_tau", type=float, default=0.0)
    parser.add_argument("--min_edge_thresh", type=float, default=0.0)
    parser.add_argument("--cost_fixed", type=float, default=0.0)
    parser.add_argument("--cost_fixed_bps", type=float, default=None)
    parser.add_argument("--cost_unit", default="return", choices=["return", "bps"])
    parser.add_argument("--cost_k", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--rebalance_hours", type=int, default=1)
    parser.add_argument("--rank_mode", default="long_short", choices=["long_short", "long_only", "short_only"])
    parser.add_argument("--rank_normalize", action="store_true")
    parser.add_argument("--rank_gross_exposure", type=float, default=1.0)
    parser.add_argument(
        "--score_transform",
        default="none",
        choices=["none", "cs_demean", "cs_zscore", "ts_zscore"],
    )
    parser.add_argument("--ts_lookback", type=int, default=20)
    parser.add_argument("--ts_min_count", type=int, default=10)
    parser.add_argument("--ts_stat", default="median", choices=["mean", "median"])
    parser.add_argument("--ts_scale", default="std", choices=["std", "mad"])
    parser.add_argument("--ts_z_entry", type=float, default=1.0)
    parser.add_argument("--ts_z_exit", type=float, default=None)
    parser.add_argument("--ts_z_cap", type=float, default=2.0)
    parser.add_argument("--ts_use_vol", action="store_true")
    parser.add_argument("--ts_vol_lookback", type=int, default=240)
    parser.add_argument("--ts_vol_mode", default="std", choices=["std", "mad"])
    parser.add_argument("--ts_vol_floor", type=float, default=1e-6)
    parser.add_argument("--width_drop_q", type=float, default=0.0)
    parser.add_argument("--hold_min_hours", type=int, default=0)
    parser.add_argument("--flip_penalty_bps", type=float, default=0.0)
    parser.add_argument(
        "--size_mode",
        default="fixed",
        choices=["fixed", "edge", "edge_over_var", "inv_width", "edge_over_width2", "mu_over_sigma"],
    )
    parser.add_argument("--size_cap", type=float, default=5.0)
    parser.add_argument("--size_zero_k", type=float, default=0.0)
    parser.add_argument("--group_by", default="origin", choices=["origin", "timestamp"])
    parser.add_argument("--out_asset", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("data", {}).get("target_mode", "level")
    use_return_target = target_mode != "level"
    preds = np.load(args.preds)
    if "origin_t" not in preds or "series_idx" not in preds:
        raise ValueError("preds npz must include origin_t and series_idx.")
    y = preds["y"]
    q10 = preds["q10"]
    q50 = preds["q50"]
    q90 = preds["q90"]
    mask = preds["mask"] if "mask" in preds else np.isfinite(y).astype(np.float32)
    origin_t = preds["origin_t"].astype(np.int64)
    series_idx = preds["series_idx"].astype(np.int64)

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
    if args.value_scale == "orig":
        if use_return_target:
            y = pre.inverse_return(y)
            q10 = pre.inverse_return(q10)
            q50 = pre.inverse_return(q50)
            q90 = pre.inverse_return(q90)
        else:
            y = pre.inverse_y(y)
            q10 = pre.inverse_y(q10)
            q50 = pre.inverse_y(q50)
            q90 = pre.inverse_y(q90)
    y0, origin_mask = _origin_values(series_list, series_idx, origin_t)
    if args.value_scale == "orig":
        if use_return_target:
            y0 = pre.inverse_return(y0)
        else:
            y0 = pre.inverse_y(y0)
    if use_return_target:
        y0 = np.zeros_like(y0)

    H = q50.shape[1]
    horizons = _parse_horizons(args.horizons, H)
    if not horizons:
        raise ValueError("No valid horizons selected.")

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
    time_key_hours = time_key.copy()
    if args.group_by == "timestamp":
        time_key_hours = (time_key_hours // (3600 * 1_000_000_000)).astype(np.int64)
    time_key_hours = time_key_hours - int(np.min(time_key_hours))

    summaries = []
    trades_rows: List[Tuple] = []
    overall_valid_list = []
    overall_side_list = []
    overall_size_list = []
    overall_pnl_list = []
    overall_gross_list = []
    overall_cost_list = []
    overall_time_list = []
    overall_time_hours_list = []
    overall_edge_list = []
    overall_bucket_list = []
    overall_hold_list = []
    overall_series_list = []

    cost_fixed = float(args.cost_fixed)
    if args.cost_fixed_bps is not None:
        cost_fixed = float(args.cost_fixed_bps) / 1e4
    elif args.cost_unit == "bps":
        cost_fixed = float(args.cost_fixed) / 1e4
    if args.return_mode == "log":
        cost_fixed = float(np.log1p(cost_fixed))
    flip_penalty = 0.0
    if args.flip_penalty_bps > 0.0:
        flip_penalty = float(args.flip_penalty_bps) / 1e4
        if args.return_mode == "log":
            flip_penalty = float(np.log1p(flip_penalty))

    if args.min_edge_thresh > 0.0 and args.rule != "rank":
        raise ValueError("min_edge_thresh is only supported with rule=rank.")

    for h in horizons:
        h_idx = h - 1
        valid = (mask[:, h_idx] > 0) & (origin_mask > 0)
        if not np.any(valid):
            summaries.append(
                _summarize(
                    f"h{h}",
                    valid,
                    np.zeros_like(valid),
                    np.zeros_like(valid, dtype=np.float32),
                    np.zeros_like(valid, dtype=np.float32),
                    time_key,
                    gross_pnl=np.zeros_like(valid, dtype=np.float32),
                    cost_pnl=np.zeros_like(valid, dtype=np.float32),
                    edge_prob=np.full_like(valid, np.nan, dtype=np.float32),
                    time_key_hours=time_key_hours,
                    rank_bucket=np.zeros_like(valid, dtype=np.int8),
                )
            )
            continue

        y_true = y[:, h_idx]
        q10_h = q10[:, h_idx]
        q50_h = q50[:, h_idx]
        q90_h = q90[:, h_idx]

        if not use_return_target:
            if args.return_mode == "pct":
                denom = np.maximum(np.abs(y0), 1e-8)
                y_true = (y_true - y0) / denom
                q10_h = (q10_h - y0) / denom
                q50_h = (q50_h - y0) / denom
                q90_h = (q90_h - y0) / denom
            elif args.return_mode == "log":
                valid &= (y0 > 0) & (y_true > 0) & (q10_h > 0) & (q50_h > 0) & (q90_h > 0)
                y_true = np.log(y_true) - np.log(y0)
                q10_h = np.log(q10_h) - np.log(y0)
                q50_h = np.log(q50_h) - np.log(y0)
                q90_h = np.log(q90_h) - np.log(y0)
            else:
                y_true = y_true - y0
                q10_h = q10_h - y0
                q50_h = q50_h - y0
                q90_h = q90_h - y0
        if not np.all(valid):
            y_true = np.where(valid, y_true, 0.0)
            q10_h = np.where(valid, q10_h, 0.0)
            q50_h = np.where(valid, q50_h, 0.0)
            q90_h = np.where(valid, q90_h, 0.0)

        mu_r = q50_h
        width = np.maximum(q90_h - q10_h, 1e-12)
        sigma = np.maximum(width / 2.563, 1e-12)
        cost = cost_fixed + args.cost_k * sigma

        p_plus = None
        p_minus = None
        p_move = None
        if args.rule in {"prob_cost", "rank"} or args.move_tau > 0.0:
            z_plus = (mu_r - cost) / sigma
            z_minus = (-cost - mu_r) / sigma
            p_plus = _normal_cdf(z_plus)
            p_minus = _normal_cdf(z_minus)
            p_move = p_plus + p_minus

        if args.width_drop_q > 0.0 and np.any(valid):
            thr = float(np.quantile(width[valid], args.width_drop_q))
            valid &= width <= thr

        if args.rebalance_hours > 1:
            valid &= (time_key_hours % args.rebalance_hours) == 0

        side = np.zeros_like(mu_r, dtype=np.int8)
        rank_bucket = np.zeros_like(mu_r, dtype=np.int8)
        size = None
        if args.rule == "ts":
            z_entry = float(args.ts_z_entry)
            z_exit = z_entry if args.ts_z_exit is None else float(args.ts_z_exit)
            history: Dict[int, List[float]] = {}
            last_side: Dict[int, int] = {}
            size = np.zeros_like(mu_r, dtype=np.float32)
            order = np.lexsort((origin_t, series_idx))
            for idx in order:
                if not valid[idx]:
                    continue
                s_idx = int(series_idx[idx])
                t_idx = int(origin_t[idx])
                hist = history.get(s_idx, [])
                z = 0.0
                if len(hist) >= args.ts_min_count:
                    if args.ts_lookback > 0:
                        window = np.asarray(hist[-args.ts_lookback :], dtype=np.float32)
                    else:
                        window = np.asarray(hist, dtype=np.float32)
                    center, sc = _center_scale(window, args.ts_stat, args.ts_scale)
                    if sc > 0:
                        z = (float(mu_r[idx]) - center) / sc
                hist.append(float(mu_r[idx]))
                history[s_idx] = hist
                curr = last_side.get(s_idx, 0)
                if curr == 0:
                    if abs(z) >= z_entry:
                        curr = 1 if z > 0 else -1
                else:
                    if abs(z) < z_exit:
                        curr = 0
                    elif np.sign(z) != curr and abs(z) >= z_entry:
                        curr = 1 if z > 0 else -1
                side[idx] = curr
                last_side[s_idx] = curr
                if curr != 0:
                    base = abs(z) / max(float(args.ts_z_cap), 1e-8)
                    if base > 1.0:
                        base = 1.0
                    if args.ts_use_vol:
                        series = series_list[s_idx].y_raw if series_list[s_idx].y_raw is not None else series_list[s_idx].y
                        m = series_list[s_idx].mask
                        vol = _realized_vol(
                            series,
                            m,
                            t_idx,
                            int(args.ts_vol_lookback),
                            args.return_mode,
                            args.ts_vol_mode,
                            float(args.ts_vol_floor),
                        )
                        base = base / max(vol, float(args.ts_vol_floor))
                    size[idx] = base
        elif args.rule == "prob_cost":
            take_long = (p_plus > args.tau) & (p_plus >= p_minus)
            take_short = (p_minus > args.tau) & (p_minus > p_plus)
            side[take_long] = 1
            side[take_short] = -1
        elif args.rule == "conformal_bounds":
            side[q10_h > cost] = 1
            side[q90_h < -cost] = -1
        elif args.rule == "rank":
            score = _select_scores(args.score_mode, mu_r, p_plus, p_minus)
            score = _apply_score_transform(
                score,
                valid,
                series_idx,
                time_key,
                args.score_transform,
                args.ts_lookback,
            )
            side = _assign_rank_sides(score, time_key, valid, args.topk, args.rank_mode)
            rank_bucket = _assign_rank_buckets(score, time_key, valid, args.topk, args.rank_mode)
            if args.min_edge_thresh > 0.0:
                if args.score_mode not in {"prob_edge", "edge_prob"}:
                    raise ValueError("min_edge_thresh requires score_mode=prob_edge or edge_prob.")
                if p_plus is None or p_minus is None:
                    raise ValueError("min_edge_thresh requires p_plus/p_minus.")
                drop_long = (side > 0) & (p_plus < args.min_edge_thresh)
                drop_short = (side < 0) & (p_minus < args.min_edge_thresh)
                side[drop_long | drop_short] = 0

        if p_move is not None and args.move_tau > 0.0:
            side[p_move < args.move_tau] = 0

        if size is None and args.size_mode == "mu_over_sigma":
            pos = mu_r / (sigma + 1e-12)
            if args.size_zero_k > 0.0:
                pos = np.where(np.abs(mu_r) >= args.size_zero_k * sigma, pos, 0.0)
            side = np.sign(pos).astype(np.int8)
            size = np.abs(pos).astype(np.float32)

        side, extra_cost = _apply_hold_and_flip(
            side,
            valid,
            series_idx,
            time_key_hours,
            args.hold_min_hours,
            flip_penalty,
        )
        cost = cost + extra_cost

        if size is None:
            if args.size_mode == "fixed":
                size = np.ones_like(mu_r, dtype=np.float32)
            elif args.size_mode == "edge":
                size = np.abs(mu_r)
            elif args.size_mode == "edge_over_var":
                size = np.abs(mu_r) / (sigma**2 + 1e-12)
            elif args.size_mode == "inv_width":
                size = 1.0 / (width + 1e-12)
            elif args.size_mode == "edge_over_width2":
                size = np.abs(mu_r) / (width**2 + 1e-12)
            else:
                raise ValueError(f"Unknown size_mode: {args.size_mode}")
        size = np.minimum(size, args.size_cap)

        side[~valid] = 0
        size = np.where(valid, size, 0.0)
        if args.rule == "rank" and args.rank_normalize:
            size = _normalize_rank_exposure(
                size,
                side,
                valid,
                time_key,
                args.rank_gross_exposure,
            )
        gross_pnl = side.astype(np.float32) * size * y_true
        cost_pnl = cost * size * (side != 0)
        pnl = gross_pnl - cost_pnl

        edge_prob = np.full_like(mu_r, np.nan, dtype=np.float32)
        if p_plus is not None and p_minus is not None:
            edge_prob = np.where(side > 0, p_plus, np.where(side < 0, p_minus, np.nan))

        hold_hours = np.full_like(y_true, h, dtype=np.int32)
        summaries.append(
            _summarize(
                f"h{h}",
                valid,
                side,
                size,
                pnl,
                time_key,
                gross_pnl=gross_pnl,
                cost_pnl=cost_pnl,
                edge_prob=edge_prob,
                time_key_hours=time_key_hours,
                rank_bucket=rank_bucket,
                series_idx=series_idx,
                hold_hours=hold_hours,
            )
        )
        overall_valid_list.append(valid)
        overall_side_list.append(side)
        overall_size_list.append(size)
        overall_pnl_list.append(pnl)
        overall_gross_list.append(gross_pnl)
        overall_cost_list.append(cost_pnl)
        overall_time_list.append(time_key.copy())
        overall_time_hours_list.append(time_key_hours.copy())
        overall_edge_list.append(edge_prob)
        overall_bucket_list.append(rank_bucket)
        overall_hold_list.append(hold_hours)
        overall_series_list.append(series_idx.copy())

        if args.out_trades:
            for i in np.where(valid & (side != 0))[0]:
                trades_rows.append(
                    (
                        int(series_idx[i]),
                        int(origin_t[i]),
                        h,
                        h,
                        int(side[i]),
                        float(size[i]),
                        float(y_true[i]),
                        float(pnl[i]),
                        float(cost[i]),
                        float(p_plus[i]) if p_plus is not None else float("nan"),
                        float(p_minus[i]) if p_minus is not None else float("nan"),
                    )
                )

    # Overall summary across all horizons (flattened)
    all_valid = None
    all_side = None
    all_size = None
    all_pnl = None
    all_gross = None
    all_cost = None
    all_time = None
    all_time_hours = None
    all_edge = None
    all_bucket = None
    all_hold = None
    all_series = None
    if overall_valid_list:
        all_valid = np.concatenate(overall_valid_list, axis=0)
        all_side = np.concatenate(overall_side_list, axis=0)
        all_size = np.concatenate(overall_size_list, axis=0)
        all_pnl = np.concatenate(overall_pnl_list, axis=0)
        all_gross = np.concatenate(overall_gross_list, axis=0)
        all_cost = np.concatenate(overall_cost_list, axis=0)
        all_time = np.concatenate(overall_time_list, axis=0)
        all_time_hours = np.concatenate(overall_time_hours_list, axis=0)
        all_edge = np.concatenate(overall_edge_list, axis=0) if overall_edge_list else None
        all_bucket = np.concatenate(overall_bucket_list, axis=0) if overall_bucket_list else None
        all_hold = np.concatenate(overall_hold_list, axis=0)
        all_series = np.concatenate(overall_series_list, axis=0)
        summaries.append(
            _summarize(
                "all",
                all_valid,
                all_side,
                all_size,
                all_pnl,
                all_time,
                gross_pnl=all_gross,
                cost_pnl=all_cost,
                edge_prob=all_edge,
                time_key_hours=all_time_hours,
                rank_bucket=all_bucket,
                series_idx=all_series,
                hold_hours=all_hold,
            )
        )

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    if args.out_trades:
        out_trades = Path(args.out_trades)
        out_trades.parent.mkdir(parents=True, exist_ok=True)
        with out_trades.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "series_idx",
                    "origin_t",
                    "h",
                    "hold_hours",
                    "side",
                    "size",
                    "r_true",
                    "pnl",
                    "cost",
                    "p_plus",
                    "p_minus",
                ]
            )
            writer.writerows(trades_rows)

    if args.out_asset and all_valid is not None and all_series is not None:
        out_asset = Path(args.out_asset)
        out_asset.parent.mkdir(parents=True, exist_ok=True)
        with out_asset.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "series_idx",
                    "n_trades",
                    "trade_rate",
                    "hit_rate",
                    "mean_pnl_trade",
                    "net_pnl_sum",
                    "gross_exposure_mean",
                ]
            )
            for s_idx in np.unique(all_series):
                idx = (all_series == s_idx) & (all_valid > 0)
                if not np.any(idx):
                    continue
                trade = idx & (all_side != 0)
                n_valid = int(np.sum(idx))
                n_trades = int(np.sum(trade))
                trade_rate = n_trades / max(n_valid, 1)
                if n_trades:
                    hit_rate = float(np.mean(all_pnl[trade] > 0))
                    mean_pnl = float(np.mean(all_pnl[trade]))
                    net_pnl = float(np.sum(all_pnl[trade]))
                    gross_exposure_mean = float(np.mean(np.abs(all_size[trade] * all_side[trade])))
                else:
                    hit_rate = float("nan")
                    mean_pnl = float("nan")
                    net_pnl = 0.0
                    gross_exposure_mean = float("nan")
                writer.writerow(
                    [
                        int(s_idx),
                        n_trades,
                        trade_rate,
                        hit_rate,
                        mean_pnl,
                        net_pnl,
                        gross_exposure_mean,
                    ]
                )

    all_row = next((row for row in summaries if row.get("scope") == "all"), None)
    if all_row is not None:
        print(
            "run_summary",
            f"n_trades={all_row.get('n_trades')}",
            f"gross_pnl_sum={all_row.get('gross_pnl_sum')}",
            f"cost_pnl_sum={all_row.get('cost_pnl_sum')}",
            f"net_pnl_sum={all_row.get('net_pnl_sum')}",
            f"turnover_tax={all_row.get('turnover_tax')}",
            f"turnover_sum={all_row.get('turnover_sum')}",
            f"flip_rate={all_row.get('flip_rate')}",
            f"jaccard={all_row.get('jaccard')}",
            f"gross_exposure_p10={all_row.get('gross_exposure_p10')}",
            f"gross_exposure_p50={all_row.get('gross_exposure_p50')}",
            f"gross_exposure_p90={all_row.get('gross_exposure_p90')}",
            f"edge_prob_p10={all_row.get('edge_prob_p10')}",
            f"edge_prob_p50={all_row.get('edge_prob_p50')}",
            f"edge_prob_p90={all_row.get('edge_prob_p90')}",
            f"time_delta_min_hours={all_row.get('time_delta_min_hours')}",
            f"time_delta_median_hours={all_row.get('time_delta_median_hours')}",
            f"time_delta_max_hours={all_row.get('time_delta_max_hours')}",
        )


if __name__ == "__main__":
    main()
