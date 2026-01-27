from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import compress_series_observed, load_panel_npz
from eval import apply_scaling
from utils import load_config


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    try:
        erf = np.erf
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    except AttributeError:
        from math import erf as _erf

        return 0.5 * (1.0 + np.vectorize(_erf)(x / np.sqrt(2.0)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # average ranks for ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for val in np.unique(inv):
            idx = np.where(inv == val)[0]
            if idx.size > 1:
                ranks[idx] = ranks[idx].mean()
    return ranks


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


def _load_series(cfg: Dict) -> list:
    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    return series_list


def _compute_time_key(series_list: list, series_idx: np.ndarray, origin_t: np.ndarray, mode: str) -> np.ndarray:
    if mode == "origin":
        return origin_t.astype(np.int64)
    time_key = np.zeros_like(origin_t, dtype=np.int64)
    for i, (s_idx, t) in enumerate(zip(series_idx, origin_t)):
        s = series_list[int(s_idx)]
        if s.timestamps is None:
            time_key[i] = int(t)
        else:
            time_key[i] = int(np.asarray(s.timestamps[int(t)]).astype("datetime64[ns]").astype(np.int64))
    return time_key


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


def _summarize(values: np.ndarray, label: str) -> None:
    if values.size == 0:
        print(f"{label}: empty")
        return
    print(
        f"{label}: mean={values.mean():.6f} median={np.median(values):.6f} "
        f"p10={np.quantile(values, 0.1):.6f} p90={np.quantile(values, 0.9):.6f} "
        f"pos_frac={(values > 0).mean():.3f} n={values.size}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--group_by", default="origin", choices=["origin", "timestamp"])
    parser.add_argument("--score_mode", default="q50", choices=["q50", "mean", "prob_edge", "edge_prob"])
    parser.add_argument(
        "--score_transform",
        default="none",
        choices=["none", "cs_demean", "cs_zscore", "ts_zscore"],
    )
    parser.add_argument("--ts_lookback", type=int, default=20)
    parser.add_argument("--return_mode", default="diff", choices=["diff", "pct", "log"])
    parser.add_argument("--value_scale", default="orig", choices=["orig", "scaled"])
    parser.add_argument("--cost_fixed_bps", type=float, default=0.0)
    parser.add_argument("--cost_k", type=float, default=0.0)
    parser.add_argument("--min_assets", type=int, default=5)
    parser.add_argument("--out_csv", default=None)
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

    h_idx = args.h - 1
    valid = (mask[:, h_idx] > 0) & (origin_mask > 0)
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

    mu_r = q50_h
    width = np.maximum(q90_h - q10_h, 1e-12)
    sigma = np.maximum(width / 2.563, 1e-12)
    cost = (float(args.cost_fixed_bps) / 1e4) + float(args.cost_k) * sigma

    if args.score_mode in {"mean", "q50"}:
        score = mu_r
    else:
        z_plus = (mu_r - cost) / sigma
        z_minus = (-cost - mu_r) / sigma
        p_plus = _normal_cdf(z_plus)
        p_minus = _normal_cdf(z_minus)
        score = p_plus - p_minus

    time_key = _compute_time_key(series_list, series_idx, origin_t, args.group_by)
    score = _apply_score_transform(score, valid, series_idx, time_key, args.score_transform, args.ts_lookback)

    pearsons = []
    spearmans = []
    pearsons_cs = []
    spearmans_cs = []
    n_assets = []
    out_rows = []
    for t in np.unique(time_key):
        idx = np.where((time_key == t) & valid)[0]
        if idx.size < args.min_assets:
            continue
        s = score[idx]
        r = y_true[idx]
        if np.std(s) < 1e-12 or np.std(r) < 1e-12:
            continue
        s_cs = s - np.mean(s)
        r_cs = r - np.mean(r)
        pear = float(np.corrcoef(s, r)[0, 1])
        s_rank = _rankdata(s)
        r_rank = _rankdata(r)
        spear = float(np.corrcoef(s_rank, r_rank)[0, 1])
        pear_cs = float(np.corrcoef(s_cs, r_cs)[0, 1]) if np.std(s_cs) > 1e-12 and np.std(r_cs) > 1e-12 else float("nan")
        s_rank_cs = _rankdata(s_cs)
        r_rank_cs = _rankdata(r_cs)
        spear_cs = float(np.corrcoef(s_rank_cs, r_rank_cs)[0, 1]) if np.std(s_rank_cs) > 1e-12 and np.std(r_rank_cs) > 1e-12 else float("nan")
        pearsons.append(pear)
        spearmans.append(spear)
        pearsons_cs.append(pear_cs)
        spearmans_cs.append(spear_cs)
        n_assets.append(idx.size)
        if args.out_csv:
            out_rows.append((int(t), idx.size, pear, spear, pear_cs, spear_cs))

    pearsons = np.asarray(pearsons, dtype=np.float64)
    spearmans = np.asarray(spearmans, dtype=np.float64)
    pearsons_cs = np.asarray(pearsons_cs, dtype=np.float64)
    spearmans_cs = np.asarray(spearmans_cs, dtype=np.float64)
    n_assets = np.asarray(n_assets, dtype=np.int64)

    print(f"cs_ic h={args.h} group_by={args.group_by} score={args.score_mode} transform={args.score_transform}")
    _summarize(pearsons, "pearson")
    _summarize(spearmans, "spearman")
    _summarize(pearsons_cs, "pearson_cs")
    _summarize(spearmans_cs, "spearman_cs")
    if n_assets.size:
        print(f"assets_per_time: mean={n_assets.mean():.2f} median={np.median(n_assets):.2f} n_times={n_assets.size}")

    # pooled IC: raw and cs-demeaned
    all_idx = valid
    s_all = score[all_idx]
    r_all = y_true[all_idx]
    s_all_cs = s_all.copy()
    r_all_cs = r_all.copy()
    for t in np.unique(time_key[all_idx]):
        m = time_key[all_idx] == t
        if np.any(m):
            s_all_cs[m] = s_all[m] - np.mean(s_all[m])
            r_all_cs[m] = r_all[m] - np.mean(r_all[m])
    if s_all.size > 1:
        pooled_raw = float(np.corrcoef(s_all, r_all)[0, 1])
        pooled_cs = float(np.corrcoef(s_all_cs, r_all_cs)[0, 1])
        pooled_raw_s = float(np.corrcoef(_rankdata(s_all), _rankdata(r_all))[0, 1])
        pooled_cs_s = float(np.corrcoef(_rankdata(s_all_cs), _rankdata(r_all_cs))[0, 1])
        print(f"pooled_raw: pearson={pooled_raw:.6f} spearman={pooled_raw_s:.6f}")
        print(f"pooled_cs:  pearson={pooled_cs:.6f} spearman={pooled_cs_s:.6f}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("time_key,n_assets,pearson,spearman,pearson_cs,spearman_cs\n")
            for row in out_rows:
                f.write(f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}\n")


if __name__ == "__main__":
    main()
