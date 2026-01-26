from __future__ import annotations

import argparse
import csv
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


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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
    return {
        "acc": float(np.mean(y_pred == y_true)) if y_true.size else float("nan"),
        "mcc": _mcc(y_true, y_pred),
        "auc": _roc_auc(y_true, p_up),
        "brier": float(np.mean((p_up - y_true) ** 2)) if y_true.size else float("nan"),
        "ece": _ece(y_true, p_up),
        "pos_rate": float(np.mean(y_true)) if y_true.size else float("nan"),
        "n": int(y_true.size),
    }


def _collect_abs_deltas(series_list: list, indices: List[tuple], H: int, delta_mode: str) -> np.ndarray:
    out: List[float] = []
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
            out.append(abs(float(delta)))
    return np.asarray(out, dtype=np.float32)


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
    parser.add_argument("--out_metrics", required=True)
    parser.add_argument("--delta_mode", default="origin", choices=["origin", "step"])
    parser.add_argument("--delta_quantile", type=float, default=0.33)
    parser.add_argument("--bin_edges", default="")
    parser.add_argument("--bin_quantiles", default="0.2,0.4,0.6,0.8")
    parser.add_argument("--horizons", default="")
    parser.add_argument("--split_purge", type=int, default=None)
    parser.add_argument("--split_embargo", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
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
    abs_delta = np.abs(delta)

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
        h_list = [h for h in _parse_ints(args.horizons) if 1 <= h <= horizon]
    else:
        h_list = list(range(1, horizon + 1))
    h_idx = [h - 1 for h in h_list]

    if args.bin_edges.strip():
        edges = sorted(_parse_floats(args.bin_edges))
    else:
        qs = _parse_floats(args.bin_quantiles)
        eval_vals = abs_delta[:, h_idx][dir_mask[:, h_idx]]
        edges = [float(np.quantile(eval_vals, q)) for q in qs] if eval_vals.size else []
    edges = [0.0] + edges + [float("inf")]
    bins = list(zip(edges[:-1], edges[1:]))

    rows = []
    bin_summary = {}
    for lo, hi in bins:
        m = dir_mask[:, h_idx]
        bin_mask = (abs_delta[:, h_idx] >= lo) & (abs_delta[:, h_idx] < hi) & m
        y_b = dir_label[:, h_idx][bin_mask]
        p_b = dir_prob[:, h_idx][bin_mask]
        bin_summary[f"{lo:.6g}-{hi:.6g}"] = _metrics_from_arrays(y_b, p_b)
        for h in h_idx:
            m_h = dir_mask[:, h]
            bm = m_h & (abs_delta[:, h] >= lo) & (abs_delta[:, h] < hi)
            if not np.any(bm):
                rows.append([lo, hi, h + 1, "nan", "nan", "nan", "nan", "nan", "nan", 0])
                continue
            y_h = dir_label[bm, h]
            p_h = dir_prob[bm, h]
            metrics = _metrics_from_arrays(y_h, p_h)
            rows.append(
                [
                    lo,
                    hi,
                    h + 1,
                    metrics["acc"],
                    metrics["mcc"],
                    metrics["auc"],
                    metrics["brier"],
                    metrics["ece"],
                    metrics["pos_rate"],
                    metrics["n"],
                ]
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_lo", "bin_hi", "h", "acc", "mcc", "auc", "brier", "ece", "pos_rate", "n"])
        writer.writerows(rows)

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "bin_edges": edges,
        "horizons": h_list,
        "bin_summary": bin_summary,
        "delta_mode": args.delta_mode,
    }
    np.savez(out_metrics, metrics=metrics)
    print(metrics)


if __name__ == "__main__":
    main()
