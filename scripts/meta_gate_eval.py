from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import compress_series_observed, load_panel_npz
from eval import apply_scaling
from utils import load_config
from scripts import trade_rule_eval as tre


@dataclass
class CandidatePool:
    h_idx: int
    time_key_hours: int
    split_id: int
    cand_idx: np.ndarray
    side: np.ndarray
    base_score: np.ndarray
    features: np.ndarray


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
        from math import erf as _erf

        return 0.5 * (1.0 + np.vectorize(_erf)(x / np.sqrt(2.0)))


def _origin_values(
    series_list: list,
    series_idx: np.ndarray,
    origin_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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


def _build_time_key(
    series_list: list,
    series_idx: np.ndarray,
    origin_t: np.ndarray,
    group_by: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if group_by == "timestamp":
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
    if group_by == "timestamp":
        time_key_hours = (time_key_hours // (3600 * 1_000_000_000)).astype(np.int64)
    time_key_hours = time_key_hours - int(np.min(time_key_hours))
    return time_key, time_key_hours


def _split_time_keys(
    time_key_hours: np.ndarray,
    valid_mask: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> np.ndarray:
    uniq = np.unique(time_key_hours[valid_mask])
    if uniq.size == 0:
        uniq = np.unique(time_key_hours)
    uniq = np.sort(uniq)
    n = uniq.shape[0]
    if n < 3:
        split_ids = np.zeros_like(time_key_hours, dtype=np.int8)
        if n > 1:
            split_ids[np.isin(time_key_hours, uniq[-1:])] = 2
        return split_ids
    n_train = max(int(n * train_frac), 1)
    n_val = max(int(n * val_frac), 1)
    if n_train + n_val >= n:
        n_val = max(n - n_train - 1, 1)
    train_keys = uniq[:n_train]
    val_keys = uniq[n_train : n_train + n_val]
    test_keys = uniq[n_train + n_val :]
    split_ids = np.full_like(time_key_hours, 2, dtype=np.int8)
    split_ids[np.isin(time_key_hours, train_keys)] = 0
    split_ids[np.isin(time_key_hours, val_keys)] = 1
    split_ids[np.isin(time_key_hours, test_keys)] = 2
    return split_ids


def _rank_percentiles(scores: np.ndarray, ascending: bool) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    order = np.argsort(scores)
    if not ascending:
        order = order[::-1]
    pct = np.zeros_like(scores, dtype=np.float32)
    if scores.size == 1:
        pct[0] = 1.0
        return pct
    pct[order] = (np.arange(scores.size, dtype=np.float32) + 1.0) / float(scores.size)
    return pct


def _gate_score(mode: str, base_strength: np.ndarray, p_meta: np.ndarray, alpha: float) -> np.ndarray:
    if mode == "pure":
        return p_meta
    if mode == "blend_mul":
        return base_strength * p_meta
    if mode == "blend_add":
        return base_strength + alpha * p_meta
    raise ValueError(f"Unknown gate_mode: {mode}")


def _standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean, std


def _train_logistic_l1(
    x: np.ndarray,
    y: np.ndarray,
    l1_lambda: float,
    epochs: int,
    lr: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    x = x[order]
    y = y[order]
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(x.shape[1], dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        logits = x_t @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)
        loss = loss + l1_lambda * torch.sum(torch.abs(w))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return w.detach().numpy(), float(b.detach().item())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bootstrap_ci(
    pnl_time: np.ndarray,
    iters: int,
    seed: int,
) -> Tuple[float, float]:
    if iters <= 0 or pnl_time.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = pnl_time.shape[0]
    idx = rng.integers(0, n, size=(iters, n))
    samples = pnl_time[idx]
    sums = np.sum(samples, axis=1)
    return float(np.quantile(sums, 0.025)), float(np.quantile(sums, 0.975))


def _bootstrap_ci_mean(
    pnl_time: np.ndarray,
    iters: int,
    seed: int,
) -> Tuple[float, float]:
    if iters <= 0 or pnl_time.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = pnl_time.shape[0]
    idx = rng.integers(0, n, size=(iters, n))
    samples = pnl_time[idx]
    means = np.mean(samples, axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_ci_uplift(
    pnl_time_a: np.ndarray,
    pnl_time_b: np.ndarray,
    iters: int,
    seed: int,
) -> Tuple[float, float]:
    if iters <= 0 or pnl_time_a.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = pnl_time_a.shape[0]
    idx = rng.integers(0, n, size=(iters, n))
    diff = np.sum(pnl_time_b[idx], axis=1) - np.sum(pnl_time_a[idx], axis=1)
    return float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))


def _pnl_time_series(time_key: np.ndarray, pnl: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return np.zeros(0, dtype=np.float64)
    times = time_key[valid]
    uniq, inv = np.unique(times, return_inverse=True)
    pnl_time = np.zeros(uniq.shape[0], dtype=np.float64)
    np.add.at(pnl_time, inv, pnl[valid].astype(np.float64))
    return pnl_time


def _build_summary(
    side_list: List[np.ndarray],
    size_list: List[np.ndarray],
    horizon_data: List[Dict[str, np.ndarray]],
) -> Dict[str, float | str]:
    valid_all = np.concatenate([h["valid"] for h in horizon_data], axis=0)
    side_all = np.concatenate(side_list, axis=0)
    size_all = np.concatenate(size_list, axis=0)
    pnl_list = []
    gross_list = []
    cost_list = []
    edge_list = []
    time_list = []
    time_hours_list = []
    hold_list = []
    series_list = []
    for side, size, h in zip(side_list, size_list, horizon_data):
        y_true = h["y_true"]
        cost = h["cost"]
        gross_pnl = side.astype(np.float32) * size * y_true
        cost_pnl = cost * size * (side != 0)
        pnl = gross_pnl - cost_pnl
        edge_prob = np.where(side > 0, h["p_plus"], np.where(side < 0, h["p_minus"], np.nan))
        pnl_list.append(pnl)
        gross_list.append(gross_pnl)
        cost_list.append(cost_pnl)
        edge_list.append(edge_prob)
        time_list.append(h["time_key"])
        time_hours_list.append(h["time_key_hours"])
        hold_list.append(h["hold_hours"])
        series_list.append(h["series_idx"])
    pnl_all = np.concatenate(pnl_list, axis=0)
    gross_all = np.concatenate(gross_list, axis=0)
    cost_all = np.concatenate(cost_list, axis=0)
    edge_all = np.concatenate(edge_list, axis=0)
    time_all = np.concatenate(time_list, axis=0)
    time_hours_all = np.concatenate(time_hours_list, axis=0)
    hold_all = np.concatenate(hold_list, axis=0)
    series_all = np.concatenate(series_list, axis=0)
    rank_bucket = np.zeros_like(side_all, dtype=np.int8)
    summary = tre._summarize(
        "all",
        valid_all,
        side_all,
        size_all,
        pnl_all,
        time_all,
        gross_pnl=gross_all,
        cost_pnl=cost_all,
        edge_prob=edge_all,
        time_key_hours=time_hours_all,
        rank_bucket=rank_bucket,
        series_idx=series_all,
        hold_hours=hold_all,
    )
    return summary


def _collect_pnl_time(
    side_list: List[np.ndarray],
    size_list: List[np.ndarray],
    horizon_data: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    pnl_all = []
    time_all = []
    valid_all = []
    for side, size, h in zip(side_list, size_list, horizon_data):
        y_true = h["y_true"]
        cost = h["cost"]
        gross_pnl = side.astype(np.float32) * size * y_true
        cost_pnl = cost * size * (side != 0)
        pnl = gross_pnl - cost_pnl
        pnl_all.append(pnl)
        time_all.append(h["time_key"])
        valid_all.append(h["valid"])
    pnl_all = np.concatenate(pnl_all, axis=0)
    time_all = np.concatenate(time_all, axis=0)
    valid_all = np.concatenate(valid_all, axis=0)
    return _pnl_time_series(time_all, pnl_all, valid_all)


def _split_horizon_data(
    horizon_data: List[Dict[str, np.ndarray]],
    split_ids: np.ndarray,
    split_id: int,
) -> List[Dict[str, np.ndarray]]:
    out: List[Dict[str, np.ndarray]] = []
    for h in horizon_data:
        h_split = h.copy()
        h_split["valid"] = h["valid"] & (split_ids == split_id)
        out.append(h_split)
    return out


def _select_baseline(
    pools_by_h: List[List[CandidatePool]],
    horizon_data: List[Dict[str, np.ndarray]],
    k_trade: int,
    split_id: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    side_list = []
    size_list = []
    size_val = 1.0 / (2.0 * k_trade)
    for h_idx, pools in enumerate(pools_by_h):
        side = np.zeros_like(horizon_data[h_idx]["valid"], dtype=np.int8)
        size = np.zeros_like(horizon_data[h_idx]["valid"], dtype=np.float32)
        for pool in pools:
            if pool.split_id != split_id:
                continue
            cand_idx = pool.cand_idx
            side_c = pool.side
            score = pool.base_score
            long_mask = side_c > 0
            short_mask = side_c < 0
            long_idx = cand_idx[long_mask]
            short_idx = cand_idx[short_mask]
            long_score = score[long_mask]
            short_score = score[short_mask]
            if long_idx.size < k_trade or short_idx.size < k_trade:
                continue
            long_sel = long_idx[np.argsort(long_score)[-k_trade:]]
            short_sel = short_idx[np.argsort(short_score)[:k_trade]]
            side[long_sel] = 1
            side[short_sel] = -1
            size[long_sel] = size_val
            size[short_sel] = size_val
        side_list.append(side)
        size_list.append(size)
    return side_list, size_list


def _select_gate(
    pools_by_h: List[List[CandidatePool]],
    horizon_data: List[Dict[str, np.ndarray]],
    k_trade: int,
    split_id: int,
    w: np.ndarray,
    b: float,
    mean: np.ndarray,
    std: np.ndarray,
    gate_mode: str,
    alpha: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    side_list = []
    size_list = []
    size_val = 1.0 / (2.0 * k_trade)
    for h_idx, pools in enumerate(pools_by_h):
        side = np.zeros_like(horizon_data[h_idx]["valid"], dtype=np.int8)
        size = np.zeros_like(horizon_data[h_idx]["valid"], dtype=np.float32)
        for pool in pools:
            if pool.split_id != split_id:
                continue
            cand_idx = pool.cand_idx
            side_c = pool.side
            feat = pool.features
            feat = (feat - mean) / std
            p_meta = _sigmoid(feat @ w + b)
            base_strength = side_c.astype(np.float32) * pool.base_score
            score = _gate_score(gate_mode, base_strength, p_meta, alpha)
            long_mask = side_c > 0
            short_mask = side_c < 0
            long_idx = cand_idx[long_mask]
            short_idx = cand_idx[short_mask]
            long_score = score[long_mask]
            short_score = score[short_mask]
            if long_idx.size < k_trade or short_idx.size < k_trade:
                continue
            long_sel = long_idx[np.argsort(long_score)[-k_trade:]]
            short_sel = short_idx[np.argsort(short_score)[-k_trade:]]
            side[long_sel] = 1
            side[short_sel] = -1
            size[long_sel] = size_val
            size[short_sel] = size_val
        side_list.append(side)
        size_list.append(size)
    return side_list, size_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out_summary", required=True)
    parser.add_argument("--out_uplift", default=None)
    parser.add_argument("--horizons", default="")
    parser.add_argument("--return_mode", default="diff", choices=["diff", "pct", "log"])
    parser.add_argument("--value_scale", default="orig", choices=["orig", "scaled"])
    parser.add_argument("--cost_fixed_bps", type=float, default=10.0)
    parser.add_argument("--cost_k", type=float, default=0.0)
    parser.add_argument("--group_by", default="timestamp", choices=["origin", "timestamp"])
    parser.add_argument("--k_cand", type=int, default=20)
    parser.add_argument("--k_trade", type=int, default=3)
    parser.add_argument("--margin_bps", default="0,5")
    parser.add_argument("--gate_mode", default="blend_mul", choices=["pure", "blend_mul", "blend_add"])
    parser.add_argument("--blend_alpha", type=float, default=0.5)
    parser.add_argument("--blend_alpha_grid", default="")
    parser.add_argument("--l1_lambda", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap_iters", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    preds = np.load(args.preds)
    y = preds["y"]
    q10 = preds["q10"]
    q50 = preds["q50"]
    q90 = preds["q90"]
    mask = preds["mask"] if "mask" in preds else np.isfinite(y).astype(np.float32)
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
    if args.value_scale == "orig":
        y = pre.inverse_y(y)
        q10 = pre.inverse_y(q10)
        q50 = pre.inverse_y(q50)
        q90 = pre.inverse_y(q90)
    y0, origin_mask = _origin_values(series_list, series_idx, origin_t)
    if args.value_scale == "orig":
        y0 = pre.inverse_y(y0)

    time_key, time_key_hours = _build_time_key(series_list, series_idx, origin_t, args.group_by)
    split_ids = _split_time_keys(
        time_key_hours,
        origin_mask > 0,
        cfg["data"].get("train_frac", 0.7),
        cfg["data"].get("val_frac", 0.15),
    )

    H = q50.shape[1]
    horizons = _parse_horizons(args.horizons, H)
    if not horizons:
        raise ValueError("No valid horizons selected.")

    margins = []
    for part in args.margin_bps.split(","):
        part = part.strip()
        if part:
            margins.append(float(part))
    if not margins:
        margins = [0.0]

    pools_by_h: List[List[CandidatePool]] = [[] for _ in horizons]
    horizon_data: List[Dict[str, np.ndarray]] = []
    feature_list: List[np.ndarray] = []
    split_list: List[np.ndarray] = []
    labels_by_margin: Dict[float, List[int]] = {m: [] for m in margins}

    for h_idx, h in enumerate(horizons):
        h_index = h - 1
        valid = (mask[:, h_index] > 0) & (origin_mask > 0)
        if not np.any(valid):
            horizon_data.append(
                {
                    "valid": valid,
                    "time_key": time_key.copy(),
                    "time_key_hours": time_key_hours.copy(),
                    "y_true": np.zeros_like(y[:, h_index], dtype=np.float32),
                    "cost": np.zeros_like(y[:, h_index], dtype=np.float32),
                    "p_plus": np.zeros_like(y[:, h_index], dtype=np.float32),
                    "p_minus": np.zeros_like(y[:, h_index], dtype=np.float32),
                    "hold_hours": np.full_like(y[:, h_index], h, dtype=np.int32),
                    "series_idx": series_idx.copy(),
                }
            )
            continue

        y_true = y[:, h_index]
        q10_h = q10[:, h_index]
        q50_h = q50[:, h_index]
        q90_h = q90[:, h_index]

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
        cost = (args.cost_fixed_bps / 1e4) + args.cost_k * sigma
        if args.return_mode == "log":
            cost = np.log1p(cost)

        z_plus = (mu_r - cost) / sigma
        z_minus = (-cost - mu_r) / sigma
        p_plus = _normal_cdf(z_plus)
        p_minus = _normal_cdf(z_minus)
        base_score = p_plus - p_minus

        horizon_data.append(
            {
                "valid": valid,
                "time_key": time_key.copy(),
                "time_key_hours": time_key_hours.copy(),
                "y_true": y_true.astype(np.float32),
                "cost": cost.astype(np.float32),
                "p_plus": p_plus.astype(np.float32),
                "p_minus": p_minus.astype(np.float32),
                "hold_hours": np.full_like(y_true, h, dtype=np.int32),
                "series_idx": series_idx.copy(),
            }
        )

        uniq_times = np.unique(time_key[valid])
        for t in uniq_times:
            idx = np.where((time_key == t) & valid)[0]
            if idx.size < 2 * args.k_trade:
                continue
            k_cand = min(args.k_cand, idx.size // 2)
            if k_cand < args.k_trade:
                continue
            order = np.argsort(base_score[idx])
            bottom = idx[order[:k_cand]]
            top = idx[order[-k_cand:]]
            pool_time = int(time_key_hours[idx[0]])
            pool_split = int(split_ids[idx[0]])
            bottom_pct = _rank_percentiles(base_score[bottom], ascending=True)
            top_pct = _rank_percentiles(base_score[top], ascending=False)
            rank_pct = np.concatenate([bottom_pct, top_pct], axis=0)
            cand_idx = np.concatenate([bottom, top], axis=0)
            side = np.concatenate(
                [
                    -np.ones(bottom.shape[0], dtype=np.int8),
                    np.ones(top.shape[0], dtype=np.int8),
                ],
                axis=0,
            )
            cand_score = base_score[cand_idx]
            cand_mu = mu_r[cand_idx]
            cand_sigma = sigma[cand_idx]
            cand_width = width[cand_idx]
            cand_p_plus = p_plus[cand_idx]
            cand_p_minus = p_minus[cand_idx]
            cand_cost = cost[cand_idx]
            cand_h = np.full_like(cand_mu, h, dtype=np.float32)
            base_strength = side.astype(np.float32) * cand_score
            hour = float(pool_time % 24)
            dow = float((pool_time // 24) % 7)
            hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
            hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
            dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
            dow_cos = np.cos(2.0 * np.pi * dow / 7.0)
            time_feat = np.tile(
                np.array([hour_sin, hour_cos, dow_sin, dow_cos], dtype=np.float32),
                (cand_idx.shape[0], 1),
            )
            cand_feat = np.stack(
                [
                    cand_score,
                    rank_pct,
                    cand_p_plus,
                    cand_p_minus,
                    cand_mu,
                    cand_sigma,
                    cand_width,
                    np.abs(cand_mu),
                    cand_cost,
                    base_strength,
                    side.astype(np.float32),
                    cand_h,
                    time_feat[:, 0],
                    time_feat[:, 1],
                    time_feat[:, 2],
                    time_feat[:, 3],
                ],
                axis=1,
            ).astype(np.float32)
            feature_list.append(cand_feat)
            split_list.append(np.full(cand_idx.shape[0], pool_split, dtype=np.int8))
            net_return = side.astype(np.float32) * y_true[cand_idx] - cand_cost
            for margin in margins:
                threshold = margin / 1e4
                labels_by_margin[margin].append((net_return > threshold).astype(np.int8))
            pools_by_h[h_idx].append(
                CandidatePool(
                    h_idx=h_idx,
                    time_key_hours=pool_time,
                    split_id=pool_split,
                    cand_idx=cand_idx,
                    side=side,
                    base_score=cand_score,
                    features=cand_feat,
                )
            )

    if not feature_list:
        raise ValueError("No candidate pools available.")
    features = np.concatenate(feature_list, axis=0)
    splits = np.concatenate(split_list, axis=0)
    mask_f = np.all(np.isfinite(features), axis=1)
    if not np.all(mask_f):
        features = features[mask_f]
        splits = splits[mask_f]
        for margin in margins:
            labels_by_margin[margin] = [arr for arr in labels_by_margin[margin]]

    split_names = {0: "train", 1: "val", 2: "test"}
    split_horizons = {sid: _split_horizon_data(horizon_data, split_ids, sid) for sid in split_names}
    baseline_summaries = {}
    baseline_pnl_time = {}
    for sid in split_names:
        b_side, b_size = _select_baseline(pools_by_h, split_horizons[sid], args.k_trade, sid)
        baseline_summaries[sid] = _build_summary(b_side, b_size, split_horizons[sid])
        baseline_pnl_time[sid] = _collect_pnl_time(b_side, b_size, split_horizons[sid])

    rows = []
    uplift_rows = []

    train_mask = splits == 0
    if not np.any(train_mask):
        raise ValueError("No train split samples available.")

    for margin in margins:
        labels = np.concatenate(labels_by_margin[margin], axis=0).astype(np.float32)
        if not np.all(mask_f):
            labels = labels[mask_f]
        if labels.size != features.shape[0]:
            raise ValueError("Label and feature size mismatch.")
        x_train = features[train_mask]
        y_train = labels[train_mask]
        if x_train.shape[0] == 0:
            raise ValueError("Empty train split after filtering.")
        x_train_scaled, mean, std = _standardize(x_train)
        w, b = _train_logistic_l1(
            x_train_scaled,
            y_train,
            l1_lambda=args.l1_lambda,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
        )

        alpha_grid = [args.blend_alpha]
        if args.gate_mode == "blend_add" and args.blend_alpha_grid:
            alpha_grid = []
            for part in args.blend_alpha_grid.split(","):
                part = part.strip()
                if part:
                    alpha_grid.append(float(part))
        best_alpha = alpha_grid[0]
        if len(alpha_grid) > 1 and np.any(splits == 1):
            best_val = -1e30
            for alpha in alpha_grid:
                g_side, g_size = _select_gate(
                    pools_by_h,
                    split_horizons[1],
                    args.k_trade,
                    1,
                    w,
                    b,
                    mean,
                    std,
                    args.gate_mode,
                    alpha,
                )
                g_summary = _build_summary(g_side, g_size, split_horizons[1])
                val_net = float(g_summary.get("net_pnl_sum", 0.0))
                if val_net > best_val:
                    best_val = val_net
                    best_alpha = alpha

        gate_summaries = {}
        gate_pnl_time = {}
        for sid in split_names:
            g_side, g_size = _select_gate(
                pools_by_h,
                split_horizons[sid],
                args.k_trade,
                sid,
                w,
                b,
                mean,
                std,
                args.gate_mode,
                best_alpha,
            )
            gate_summaries[sid] = _build_summary(g_side, g_size, split_horizons[sid])
            gate_pnl_time[sid] = _collect_pnl_time(g_side, g_size, split_horizons[sid])

        for sid, split_name in split_names.items():
            base_summary = baseline_summaries[sid]
            gate_summary = gate_summaries[sid]
            base_net = float(base_summary.get("net_pnl_sum", 0.0))
            gate_net = float(gate_summary.get("net_pnl_sum", 0.0))
            base_trades = int(base_summary.get("n_trades", 0))
            gate_trades = int(gate_summary.get("n_trades", 0))
            base_ppt = base_net / max(base_trades, 1)
            gate_ppt = gate_net / max(gate_trades, 1)

            base_ci_low, base_ci_high = _bootstrap_ci(
                baseline_pnl_time[sid], args.bootstrap_iters, args.seed + sid
            )
            gate_ci_low, gate_ci_high = _bootstrap_ci(
                gate_pnl_time[sid], args.bootstrap_iters, args.seed + sid + 10
            )
            base_mean_ci_low, base_mean_ci_high = _bootstrap_ci_mean(
                baseline_pnl_time[sid], args.bootstrap_iters, args.seed + sid
            )
            gate_mean_ci_low, gate_mean_ci_high = _bootstrap_ci_mean(
                gate_pnl_time[sid], args.bootstrap_iters, args.seed + sid + 10
            )

            rows.append(
                {
                    "split": split_name,
                    "mode": "baseline",
                    "gate_mode": "",
                    "blend_alpha": "",
                    "margin_bps": margin,
                    "k_cand": args.k_cand,
                    "k_trade": args.k_trade,
                    "n_trades": base_summary.get("n_trades"),
                    "trade_rate": base_summary.get("trade_rate"),
                    "mean_pnl_time": base_summary.get("mean_pnl_time"),
                    "sharpe_time": base_summary.get("sharpe_time"),
                    "max_dd": base_summary.get("max_dd"),
                    "gross_pnl_sum": base_summary.get("gross_pnl_sum"),
                    "cost_pnl_sum": base_summary.get("cost_pnl_sum"),
                    "net_pnl_sum": base_summary.get("net_pnl_sum"),
                    "pnl_per_trade": base_ppt,
                    "turnover_tax": base_summary.get("turnover_tax"),
                    "turnover_sum": base_summary.get("turnover_sum"),
                    "flip_rate": base_summary.get("flip_rate"),
                    "avg_holding_time": base_summary.get("avg_holding_time"),
                    "jaccard": base_summary.get("jaccard"),
                    "net_pnl_ci_low": base_ci_low,
                    "net_pnl_ci_high": base_ci_high,
                    "mean_pnl_time_ci_low": base_mean_ci_low,
                    "mean_pnl_time_ci_high": base_mean_ci_high,
                }
            )
            rows.append(
                {
                    "split": split_name,
                    "mode": "gate",
                    "gate_mode": args.gate_mode,
                    "blend_alpha": best_alpha,
                    "margin_bps": margin,
                    "k_cand": args.k_cand,
                    "k_trade": args.k_trade,
                    "n_trades": gate_summary.get("n_trades"),
                    "trade_rate": gate_summary.get("trade_rate"),
                    "mean_pnl_time": gate_summary.get("mean_pnl_time"),
                    "sharpe_time": gate_summary.get("sharpe_time"),
                    "max_dd": gate_summary.get("max_dd"),
                    "gross_pnl_sum": gate_summary.get("gross_pnl_sum"),
                    "cost_pnl_sum": gate_summary.get("cost_pnl_sum"),
                    "net_pnl_sum": gate_summary.get("net_pnl_sum"),
                    "pnl_per_trade": gate_ppt,
                    "turnover_tax": gate_summary.get("turnover_tax"),
                    "turnover_sum": gate_summary.get("turnover_sum"),
                    "flip_rate": gate_summary.get("flip_rate"),
                    "avg_holding_time": gate_summary.get("avg_holding_time"),
                    "jaccard": gate_summary.get("jaccard"),
                    "net_pnl_ci_low": gate_ci_low,
                    "net_pnl_ci_high": gate_ci_high,
                    "mean_pnl_time_ci_low": gate_mean_ci_low,
                    "mean_pnl_time_ci_high": gate_mean_ci_high,
                }
            )

        base_summary = baseline_summaries[2]
        gate_summary = gate_summaries[2]
        base_net = float(base_summary.get("net_pnl_sum", 0.0))
        gate_net = float(gate_summary.get("net_pnl_sum", 0.0))
        base_trades = int(base_summary.get("n_trades", 0))
        gate_trades = int(gate_summary.get("n_trades", 0))
        base_ppt = base_net / max(base_trades, 1)
        gate_ppt = gate_net / max(gate_trades, 1)
        uplift_ci_low, uplift_ci_high = _bootstrap_ci_uplift(
            baseline_pnl_time[2], gate_pnl_time[2], args.bootstrap_iters, args.seed + 21
        )
        uplift_rows.append(
            {
                "margin_bps": margin,
                "gate_mode": args.gate_mode,
                "blend_alpha": best_alpha,
                "k_cand": args.k_cand,
                "k_trade": args.k_trade,
                "uplift_net_pnl_sum": gate_net - base_net,
                "uplift_pnl_per_trade": gate_ppt - base_ppt,
                "uplift_max_dd": float(gate_summary.get("max_dd", 0.0))
                - float(base_summary.get("max_dd", 0.0)),
                "uplift_net_ci_low": uplift_ci_low,
                "uplift_net_ci_high": uplift_ci_high,
            }
        )

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.out_uplift:
        out_uplift = Path(args.out_uplift)
        out_uplift.parent.mkdir(parents=True, exist_ok=True)
        fieldnames_uplift = list(uplift_rows[0].keys())
        with out_uplift.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_uplift)
            writer.writeheader()
            writer.writerows(uplift_rows)


if __name__ == "__main__":
    main()
