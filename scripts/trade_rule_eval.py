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
    if score_mode == "mean":
        return mu_r
    if score_mode == "prob_edge":
        if p_plus is None or p_minus is None:
            raise ValueError("prob_edge requires p_plus/p_minus.")
        return p_plus - p_minus
    raise ValueError(f"Unknown score_mode: {score_mode}")


def _assign_rank_sides(
    scores: np.ndarray,
    times: np.ndarray,
    valid_mask: np.ndarray,
    topk: int,
) -> np.ndarray:
    sides = np.zeros_like(scores, dtype=np.int8)
    if topk <= 0:
        return sides
    uniq = np.unique(times)
    for t in uniq:
        idx = np.where((times == t) & valid_mask)[0]
        if idx.size < 2 * topk:
            continue
        order = np.argsort(scores[idx])
        bottom = idx[order[:topk]]
        top = idx[order[-topk:]]
        sides[top] = 1
        sides[bottom] = -1
    return sides


def _compute_time_metrics(origin_t: np.ndarray, pnl: np.ndarray) -> Tuple[float, float, float, float]:
    if origin_t.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    uniq, inv = np.unique(origin_t, return_inverse=True)
    pnl_time = np.zeros(uniq.shape[0], dtype=np.float64)
    np.add.at(pnl_time, inv, pnl.astype(np.float64))
    mean_t = float(np.mean(pnl_time))
    std_t = float(np.std(pnl_time))
    sharpe_t = mean_t / (std_t + 1e-12)
    cum = np.cumsum(pnl_time)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum)) if cum.size else float("nan")
    return mean_t, std_t, sharpe_t, max_dd


def _summarize(
    scope: str,
    valid_mask: np.ndarray,
    side: np.ndarray,
    size: np.ndarray,
    pnl: np.ndarray,
    origin_t: np.ndarray,
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
    mean_t, std_t, sharpe_t, max_dd = _compute_time_metrics(origin_t[valid], pnl[valid])
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
        "max_dd": max_dd,
        "avg_size": avg_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--out_trades", default=None)
    parser.add_argument("--rule", default="prob_cost", choices=["prob_cost", "conformal_bounds", "rank"])
    parser.add_argument("--score_mode", default="mean", choices=["mean", "prob_edge"])
    parser.add_argument("--horizons", default="")
    parser.add_argument("--return_mode", default="diff", choices=["diff", "pct", "log"])
    parser.add_argument("--value_scale", default="orig", choices=["orig", "scaled"])
    parser.add_argument("--tau", type=float, default=0.60)
    parser.add_argument("--move_tau", type=float, default=0.0)
    parser.add_argument("--cost_fixed", type=float, default=0.0)
    parser.add_argument("--cost_fixed_bps", type=float, default=None)
    parser.add_argument("--cost_unit", default="return", choices=["return", "bps"])
    parser.add_argument("--cost_k", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--size_mode",
        default="fixed",
        choices=["fixed", "edge", "edge_over_var", "inv_width", "edge_over_width2"],
    )
    parser.add_argument("--size_cap", type=float, default=5.0)
    parser.add_argument("--group_by", default="origin", choices=["origin", "timestamp"])
    args = parser.parse_args()

    cfg = load_config(args.config)
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
    pre = apply_scaling(series_list, split.train_end, scale_x=cfg["data"].get("scale_x", True))
    if args.value_scale == "orig":
        y = pre.inverse_y(y)
        q10 = pre.inverse_y(q10)
        q50 = pre.inverse_y(q50)
        q90 = pre.inverse_y(q90)
    y0, origin_mask = _origin_values(series_list, series_idx, origin_t)
    if args.value_scale == "orig":
        y0 = pre.inverse_y(y0)

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

    summaries = []
    trades_rows: List[Tuple] = []
    overall_valid_list = []
    overall_side_list = []
    overall_size_list = []
    overall_pnl_list = []
    overall_time_list = []

    cost_fixed = float(args.cost_fixed)
    if args.cost_fixed_bps is not None:
        cost_fixed = float(args.cost_fixed_bps) / 1e4
    elif args.cost_unit == "bps":
        cost_fixed = float(args.cost_fixed) / 1e4
    if args.return_mode == "log":
        cost_fixed = float(np.log1p(cost_fixed))

    for h in horizons:
        h_idx = h - 1
        valid = (mask[:, h_idx] > 0) & (origin_mask > 0)
        if not np.any(valid):
            summaries.append(
                _summarize(f"h{h}", valid, np.zeros_like(valid), np.zeros_like(valid, dtype=np.float32), np.zeros_like(valid, dtype=np.float32), time_key)
            )
            continue

        y_true = y[:, h_idx]
        q10_h = q10[:, h_idx]
        q50_h = q50[:, h_idx]
        q90_h = q90[:, h_idx]

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

        side = np.zeros_like(mu_r, dtype=np.int8)
        if args.rule == "prob_cost":
            take_long = (p_plus > args.tau) & (p_plus >= p_minus)
            take_short = (p_minus > args.tau) & (p_minus > p_plus)
            side[take_long] = 1
            side[take_short] = -1
        elif args.rule == "conformal_bounds":
            side[q10_h > cost] = 1
            side[q90_h < -cost] = -1
        elif args.rule == "rank":
            score = _select_scores(args.score_mode, mu_r, p_plus, p_minus)
            side = _assign_rank_sides(score, time_key, valid, args.topk)

        if p_move is not None and args.move_tau > 0.0:
            side[p_move < args.move_tau] = 0

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
        pnl = side.astype(np.float32) * size * y_true - cost * size * (side != 0)

        summaries.append(_summarize(f"h{h}", valid, side, size, pnl, time_key))
        overall_valid_list.append(valid)
        overall_side_list.append(side)
        overall_size_list.append(size)
        overall_pnl_list.append(pnl)
        overall_time_list.append(time_key.copy())

        if args.out_trades:
            for i in np.where(valid & (side != 0))[0]:
                trades_rows.append(
                    (
                        int(series_idx[i]),
                        int(origin_t[i]),
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
    if overall_valid_list:
        all_valid = np.concatenate(overall_valid_list, axis=0)
        all_side = np.concatenate(overall_side_list, axis=0)
        all_size = np.concatenate(overall_size_list, axis=0)
        all_pnl = np.concatenate(overall_pnl_list, axis=0)
        all_time = np.concatenate(overall_time_list, axis=0)
        summaries.append(_summarize("all", all_valid, all_side, all_size, all_pnl, all_time))

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


if __name__ == "__main__":
    main()
