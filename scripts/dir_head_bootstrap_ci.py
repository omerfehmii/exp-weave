from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.features import append_direction_features
from data.loader import WindowedDataset, compress_series_observed, load_panel_npz
from eval import apply_scaling, build_model
from train import compute_delta_thresholds, filter_indices_with_observed
from utils import load_config, set_seed


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


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


def _metrics_from_arrays(y_true: np.ndarray, p_up: np.ndarray) -> Dict[str, float]:
    y_pred = (p_up >= 0.5).astype(int)
    return {
        "acc": float(np.mean(y_pred == y_true)) if y_true.size else float("nan"),
        "mcc": _mcc(y_true, y_pred),
        "auc": _roc_auc(y_true, p_up),
        "n": int(y_true.size),
    }


def _build_series_index_map(series_idx: np.ndarray, origin_t: np.ndarray) -> Dict[int, List[int]]:
    per_series: Dict[int, List[int]] = defaultdict(list)
    order = np.argsort(origin_t)
    for idx in order:
        per_series[int(series_idx[idx])].append(int(idx))
    return per_series


def _sample_blocks(idx_list: List[int], block_size: int, rng: np.random.Generator) -> List[int]:
    n = len(idx_list)
    if n == 0:
        return []
    block = min(block_size, n)
    n_blocks = int(math.ceil(n / block))
    out: List[int] = []
    for _ in range(n_blocks):
        start = int(rng.integers(0, n - block + 1))
        out.extend(idx_list[start : start + block])
    return out[:n]


def _bootstrap_indices(per_series: Dict[int, List[int]], block_size: int, rng: np.random.Generator) -> List[int]:
    out: List[int] = []
    for idx_list in per_series.values():
        out.extend(_sample_blocks(idx_list, block_size, rng))
    return out


def _compute_sigma_torch(y_past: torch.Tensor, mask: torch.Tensor, window: int) -> torch.Tensor:
    if y_past.ndim == 3:
        y = y_past[..., 0]
    else:
        y = y_past
    m = mask[..., 0] if mask.ndim == 3 else mask
    dy = y[:, 1:] - y[:, :-1]
    m_dy = m[:, 1:] * m[:, :-1]
    if window <= 0 or window > dy.shape[1]:
        window = dy.shape[1]
    dy = dy[:, -window:]
    m_dy = m_dy[:, -window:]
    denom = torch.clamp(m_dy.sum(dim=1, keepdim=True), min=1.0)
    mean = (dy * m_dy).sum(dim=1, keepdim=True) / denom
    var = ((dy - mean) ** 2 * m_dy).sum(dim=1, keepdim=True) / denom
    sigma = torch.sqrt(torch.clamp(var, min=0.0))
    return sigma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--delta_mode", default="origin", choices=["origin", "step"])
    parser.add_argument("--delta_quantile", type=float, default=0.33)
    parser.add_argument("--split_purge", type=int, default=None)
    parser.add_argument("--split_embargo", type=int, default=None)
    parser.add_argument("--horizons", default="1,12,24")
    parser.add_argument("--bootstrap_runs", type=int, default=200)
    parser.add_argument("--block_size", type=int, default=24)
    parser.add_argument("--metrics", default="mcc,auc")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = load_config(args.config)

    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    lengths = [len(s.y) for s in series_list]
    split = make_time_splits(min(lengths), cfg["data"].get("train_frac", 0.7), cfg["data"].get("val_frac", 0.15))
    split_purge = int(cfg["data"].get("split_purge", 0)) if args.split_purge is None else int(args.split_purge)
    split_embargo = int(cfg["data"].get("split_embargo", 0)) if args.split_embargo is None else int(args.split_embargo)
    dir_feat_cfg = cfg.get("data", {}).get("direction_features", {})
    if dir_feat_cfg.get("enabled", False):
        window = int(dir_feat_cfg.get("window", 24))
        append_direction_features(series_list, window=window, split_end=split.train_end)

    indices = generate_panel_origins(lengths, cfg["data"]["L"], cfg["data"]["H"], cfg["data"].get("step", cfg["data"]["H"]))
    horizon = cfg["data"]["H"]
    split_idx = select_indices_by_time(
        indices,
        split,
        args.split,
        horizon=horizon,
        purge=split_purge,
        embargo=split_embargo,
    )
    split_idx = filter_indices_with_observed(
        series_list,
        split_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        cfg["data"].get("min_past_obs", 1),
        cfg["data"].get("min_future_obs", 1),
    )
    apply_scaling(
        series_list,
        split.train_end,
        scale_x=cfg["data"].get("scale_x", True),
        scale_y=cfg["data"].get("scale_y", True),
    )

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
    tau_h = compute_delta_thresholds(series_list, train_idx, horizon, args.delta_quantile, delta_mode=args.delta_mode)

    target_mode = cfg["data"].get("target_mode", "level")
    target_log_eps = float(cfg["data"].get("target_log_eps", 1e-6))
    ds = WindowedDataset(
        series_list,
        split_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        target_mode=target_mode,
        target_log_eps=target_log_eps,
    )
    loader = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 64))

    device = torch.device(cfg["training"].get("device", "cpu"))
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    move_logits_list = []
    dir_logits_list = []
    logits3_list = []
    y_future_list = []
    y_last_list = []
    mask_list = []
    origin_mask_list = []
    sigma_list = []
    dir_cfg = cfg.get("training", {}).get("direction", {})
    epsilon_mode = dir_cfg.get("epsilon_mode", "quantile")
    epsilon_k = float(dir_cfg.get("epsilon_k", 1.0))
    epsilon_window = int(dir_cfg.get("epsilon_window", 24))
    with torch.no_grad():
        for batch, target in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            _, extras = model(batch)
            if "dir_logits3" in extras:
                logits3_list.append(extras["dir_logits3"].cpu().numpy())
            else:
                if "dir_move_logits" not in extras or "dir_dir_logits" not in extras:
                    raise RuntimeError("direction head logits not found in model output.")
                move_logits_list.append(extras["dir_move_logits"].cpu().numpy())
                dir_logits_list.append(extras["dir_dir_logits"].cpu().numpy())
            y_future_list.append(target[..., 0].cpu().numpy())
            y_last_list.append(batch["y_past"][:, -1, 0].cpu().numpy())
            mask_list.append(np.isfinite(target[..., 0].cpu().numpy()).astype(np.float32))
            origin_mask_list.append(batch["mask"][:, -1, 0].cpu().numpy())
            if epsilon_mode == "vol":
                sigma = _compute_sigma_torch(batch["y_past"], batch["mask"], epsilon_window)
                sigma_list.append(sigma.cpu().numpy())

    use_three_class = len(logits3_list) > 0
    if use_three_class:
        logits3 = np.concatenate(logits3_list, axis=0)
    else:
        move_logits = np.concatenate(move_logits_list, axis=0)
        dir_logits = np.concatenate(dir_logits_list, axis=0)
    y_future = np.concatenate(y_future_list, axis=0)
    y_last = np.concatenate(y_last_list, axis=0)
    mask = np.concatenate(mask_list, axis=0) > 0
    origin_mask = np.concatenate(origin_mask_list, axis=0) > 0
    mask = mask & origin_mask[:, None]

    H = y_future.shape[1]
    if args.delta_mode == "step":
        ref = np.concatenate([y_last[:, None], y_future[:, :-1]], axis=1)
    else:
        ref = y_last[:, None]
    delta = y_future - ref

    if epsilon_mode == "vol":
        if not sigma_list:
            raise RuntimeError("epsilon_mode=vol but sigma not computed.")
        sigma_arr = np.concatenate(sigma_list, axis=0).reshape(-1)
        eps = epsilon_k * sigma_arr[:, None] * np.sqrt(np.arange(1, H + 1, dtype=np.float32))
        move_label = (np.abs(delta) >= eps).astype(np.int32)
    else:
        move_label = (np.abs(delta) >= tau_h).astype(np.int32)

    if use_three_class:
        probs = _softmax(logits3)
        p_down = probs[:, :, 0]
        p_flat = probs[:, :, 1]
        p_up = probs[:, :, 2]
        if epsilon_mode == "vol":
            eps_h = eps
        else:
            eps_h = tau_h[None, :]
        dir_mask = mask & (np.abs(delta) >= eps_h)
        dir_label = (delta >= eps_h).astype(np.int32)
        dir_prob = p_up / np.maximum(p_up + p_down, 1e-6)
        move_prob = 1.0 - p_flat
    else:
        move_prob = _sigmoid(move_logits)
        p_up = _sigmoid(dir_logits) * move_prob
        dir_prob = p_up / np.maximum(move_prob, 1e-6)
        dir_label = (delta > 0).astype(np.int32)
        dir_mask = mask & (move_label > 0)

    if args.horizons.strip():
        h_list = [h for h in _parse_ints(args.horizons) if 1 <= h <= H]
    else:
        h_list = list(range(1, H + 1))
    h_idx = [h - 1 for h in h_list]

    series_idx = np.array([s for s, _ in split_idx], dtype=np.int64)
    origin_t = np.array([t for _, t in split_idx], dtype=np.int64)
    per_series = _build_series_index_map(series_idx, origin_t)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    boot = {m: np.full((args.bootstrap_runs, len(h_idx)), np.nan, dtype=np.float32) for m in metrics}
    estimate = {m: np.full(len(h_idx), np.nan, dtype=np.float32) for m in metrics}
    n_full = np.zeros(len(h_idx), dtype=np.int64)

    for i, h in enumerate(h_idx):
        m = dir_mask[:, h]
        y = dir_label[m, h]
        p = dir_prob[m, h]
        stats = _metrics_from_arrays(y, p)
        n_full[i] = stats["n"]
        for key in metrics:
            estimate[key][i] = stats.get(key, float("nan"))

    for b in range(args.bootstrap_runs):
        sample_idx = _bootstrap_indices(per_series, args.block_size, rng)
        idx = np.asarray(sample_idx, dtype=np.int64)
        for i, h in enumerate(h_idx):
            m = dir_mask[idx, h]
            y = dir_label[idx, h][m]
            p = dir_prob[idx, h][m]
            stats = _metrics_from_arrays(y, p)
            for key in metrics:
                boot[key][b, i] = stats.get(key, float("nan"))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["h", "metric", "estimate", "ci_lo", "ci_hi", "n"])
        for i, h in enumerate(h_list):
            for key in metrics:
                vals = boot[key][:, i]
                lo = float(np.nanpercentile(vals, 2.5)) if np.isfinite(vals).any() else float("nan")
                hi = float(np.nanpercentile(vals, 97.5)) if np.isfinite(vals).any() else float("nan")
                writer.writerow([h, key, float(estimate[key][i]), lo, hi, int(n_full[i])])


if __name__ == "__main__":
    main()
