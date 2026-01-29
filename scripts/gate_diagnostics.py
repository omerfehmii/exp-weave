import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import compress_series_observed, load_panel_npz
from eval import apply_scaling
from utils import load_config


def _load_series(cfg: Dict) -> list:
    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    return series_list


def _summary(pnl: np.ndarray) -> Tuple[float, float, float]:
    if pnl.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(pnl))
    std = float(np.std(pnl)) + 1e-12
    sharpe = mean / std
    hit = float(np.mean(pnl > 0))
    return mean, sharpe, hit


def _bin_stats(values: np.ndarray, pnl: np.ndarray, turnover: np.ndarray, n_bins: int) -> List[Tuple[int, float, float, float, float, int]]:
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (values >= lo) & (values < hi)
        if np.any(mask):
            mean, sharpe, hit = _summary(pnl[mask])
            tmean = float(np.mean(turnover[mask]))
            out.append((i, mean, sharpe, hit, tmean, int(np.sum(mask))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--score_transform", default="none", choices=["none", "cs_zscore"])
    parser.add_argument("--use_cs", action="store_true")
    parser.add_argument("--ret_cs", action="store_true")
    parser.add_argument("--disp_metric", default="std", choices=["std", "p90p10"])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("data", {}).get("target_mode", "level")
    use_return_target = target_mode != "level"

    preds = np.load(args.preds)
    y = preds["y"]
    q50 = preds["q50"]
    q50_std = preds["q50_std"] if "q50_std" in preds else None
    mask = preds["mask"] if "mask" in preds else np.isfinite(y).astype(np.float32)
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
    if use_return_target:
        y = pre.inverse_return(y)
        q50 = pre.inverse_return(q50)
    else:
        y = pre.inverse_y(y)
        q50 = pre.inverse_y(q50)

    h_idx = args.h - 1
    valid = mask[:, h_idx] > 0
    mu = q50[:, h_idx]
    ret = y[:, h_idx]
    mu_std = q50_std[:, h_idx] if q50_std is not None else None

    if origin_t is None or series_idx is None:
        raise ValueError("preds must include origin_t and series_idx.")

    time_key = origin_t
    n_series = int(np.max(series_idx)) + 1
    prev_w = np.zeros(n_series, dtype=np.float64)
    prev_scores = np.full(n_series, np.nan, dtype=np.float64)

    rows = []
    for t in np.unique(time_key):
        idx = np.where((time_key == t) & valid)[0]
        if idx.size == 0:
            continue
        mu_t = mu[idx].astype(np.float64, copy=True)
        ret_t = ret[idx].astype(np.float64, copy=True)
        if args.use_cs:
            mu_t = mu_t - np.mean(mu_t)
        if args.ret_cs:
            ret_t = ret_t - np.mean(ret_t)
        if args.score_transform == "cs_zscore":
            std = np.std(mu_t)
            if std > 1e-12:
                mu_t = (mu_t - np.mean(mu_t)) / std
        if args.disp_metric == "p90p10":
            d_t = float(np.percentile(mu_t, 90) - np.percentile(mu_t, 10))
        else:
            d_t = float(np.std(mu_t))
        u_t = float(np.mean(mu_std[idx])) if mu_std is not None else float("nan")
        s_t = float("nan")
        prev = prev_scores[series_idx[idx]]
        valid_prev = np.isfinite(prev)
        if np.sum(valid_prev) >= 2:
            a = mu_t[valid_prev]
            b = prev[valid_prev]
            a = a - np.mean(a)
            b = b - np.mean(b)
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12
            s_t = float(np.sum(a * b) / denom)
        # value-weighted pnl and turnover
        denom = np.sum(np.abs(mu_t))
        if denom <= 1e-12:
            continue
        w = mu_t / denom
        w_full = prev_w.copy()
        w_full[series_idx[idx]] = w
        pnl_t = float(np.sum(w * ret_t))
        turnover_t = float(np.sum(np.abs(w_full - prev_w)))
        prev_w = w_full
        prev_scores[series_idx[idx]] = mu_t
        rows.append((t, d_t, u_t, s_t, pnl_t, turnover_t))

    rows = np.asarray(rows, dtype=np.float64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "per_time.csv", rows, delimiter=",", header="time,D,U,S,pnl,turnover", comments="")

    D = rows[:, 1]
    U = rows[:, 2]
    S = rows[:, 3]
    P = rows[:, 4]
    T = rows[:, 5]

    # bin stats
    d_stats = _bin_stats(D, P, T, args.bins)
    np.savetxt(out_dir / "bin_D.csv", d_stats, delimiter=",", header="bin,mean,sharpe,hit,turnover,n", comments="")
    if np.isfinite(U).any():
        u_stats = _bin_stats(U[np.isfinite(U)], P[np.isfinite(U)], T[np.isfinite(U)], args.bins)
        np.savetxt(out_dir / "bin_U.csv", u_stats, delimiter=",", header="bin,mean,sharpe,hit,turnover,n", comments="")
    if np.isfinite(S).any():
        s_stats = _bin_stats(S[np.isfinite(S)], P[np.isfinite(S)], T[np.isfinite(S)], args.bins)
        np.savetxt(out_dir / "bin_S.csv", s_stats, delimiter=",", header="bin,mean,sharpe,hit,turnover,n", comments="")

    # 2D heatmap D x U
    if np.isfinite(U).any():
        d_edges = np.quantile(D, np.linspace(0, 1, args.bins + 1))
        u_edges = np.quantile(U[np.isfinite(U)], np.linspace(0, 1, args.bins + 1))
        heat_rows = []
        for i in range(args.bins):
            for j in range(args.bins):
                mask = (D >= d_edges[i]) & (D < d_edges[i + 1]) & (U >= u_edges[j]) & (U < u_edges[j + 1])
                if np.any(mask):
                    mean, sharpe, hit = _summary(P[mask])
                    tmean = float(np.mean(T[mask]))
                    heat_rows.append((i, j, mean, sharpe, hit, tmean, int(np.sum(mask))))
        np.savetxt(out_dir / "heat_DU.csv", heat_rows, delimiter=",", header="d_bin,u_bin,mean,sharpe,hit,turnover,n", comments="")


if __name__ == "__main__":
    main()
