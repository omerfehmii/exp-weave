from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.features import append_direction_features
from data.loader import WindowedDataset, load_panel_npz, compress_series_observed
from eval import apply_scaling, build_model
from train import compute_delta_thresholds, default_horizon_weights, filter_indices_with_observed
from utils import load_config, set_seed


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


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


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
    parser.add_argument("--split_purge", type=int, default=None)
    parser.add_argument("--split_embargo", type=int, default=None)
    parser.add_argument("--horizons", default="")
    parser.add_argument("--opt_threshold_split", default="")
    parser.add_argument("--opt_threshold_steps", type=int, default=101)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("training", {}).get("seed", 7)
    set_seed(seed)

    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    lengths = [len(s.y) for s in series_list]
    split = make_time_splits(min(lengths), cfg["data"].get("train_frac", 0.7), cfg["data"].get("val_frac", 0.15))
    split_purge = int(cfg["data"].get("split_purge", 0)) if args.split_purge is None else int(args.split_purge)
    split_embargo = int(cfg["data"].get("split_embargo", 0)) if args.split_embargo is None else int(args.split_embargo)
    dir_cfg = cfg.get("data", {}).get("direction_features", {})
    if dir_cfg.get("enabled", False):
        window = int(dir_cfg.get("window", 24))
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

    def collect_split_data(split_name: str) -> dict:
        split_idx = select_indices_by_time(
            indices,
            split,
            split_name,
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
        ds = WindowedDataset(
            series_list,
            split_idx,
            cfg["data"]["L"],
            cfg["data"]["H"],
            target_mode=target_mode,
            target_log_eps=target_log_eps,
        )
        loader = DataLoader(ds, batch_size=cfg["training"].get("batch_size", 64))

        move_logits_list = []
        dir_logits_list = []
        logits3_list = []
        y_future_list = []
        y_last_list = []
        mask_list = []
        origin_mask_list = []
        sigma_list = []
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
                y_last = batch["y_past"][:, -1, 0].cpu().numpy()
                if target_mode != "level":
                    y_last = np.zeros_like(y_last)
                y_last_list.append(y_last)
                mask_list.append(np.isfinite(target[..., 0].cpu().numpy()).astype(np.float32))
                origin_mask_list.append(batch["mask"][:, -1, 0].cpu().numpy())
                dir_cfg = cfg.get("training", {}).get("direction", {})
                if dir_cfg.get("epsilon_mode", "quantile") == "vol":
                    sigma = _compute_sigma_torch(batch["y_past"], batch["mask"], int(dir_cfg.get("epsilon_window", 24)))
                    sigma_list.append(sigma.cpu().numpy())

        use_three_class = len(logits3_list) > 0
        if use_three_class:
            logits3 = np.concatenate(logits3_list, axis=0)
            move_prob = None
            p_up = None
        else:
            logits3 = None
            move_logits = np.concatenate(move_logits_list, axis=0)
            dir_logits = np.concatenate(dir_logits_list, axis=0)
            move_prob = _sigmoid(move_logits)
            p_up = _sigmoid(dir_logits) * move_prob
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
        dir_cfg = cfg.get("training", {}).get("direction", {})
        epsilon_mode = dir_cfg.get("epsilon_mode", "quantile")
        epsilon_k = float(dir_cfg.get("epsilon_k", 1.0))
        if epsilon_mode == "vol":
            if not sigma_list:
                raise RuntimeError("epsilon_mode=vol but sigma not computed.")
            sigma_arr = np.concatenate(sigma_list, axis=0).reshape(-1)
            eps = epsilon_k * sigma_arr[:, None] * np.sqrt(np.arange(1, H + 1, dtype=np.float32))
            move_label = (np.abs(delta) >= eps).astype(np.int32)
        else:
            eps = None
            move_label = (np.abs(delta) >= tau_h).astype(np.int32)

        if use_three_class:
            probs = np.exp(logits3 - np.max(logits3, axis=2, keepdims=True))
            probs = probs / np.sum(probs, axis=2, keepdims=True)
            p_down = probs[:, :, 0]
            p_flat = probs[:, :, 1]
            p_up = probs[:, :, 2]
            if epsilon_mode == "vol":
                eps_h = eps
            else:
                eps_h = tau_h[None, :]
            dir_label = (delta >= eps_h).astype(np.int32)
            dir_mask = mask & (np.abs(delta) >= eps_h)
            dir_prob = p_up / np.maximum(p_up + p_down, 1e-6)
            move_prob = 1.0 - p_flat
        else:
            dir_label = (delta > 0).astype(np.int32)
            dir_mask = mask & (move_label > 0)
            dir_prob = p_up / np.maximum(move_prob, 1e-6)

        return {
            "use_three_class": use_three_class,
            "logits3": logits3,
            "move_prob": move_prob,
            "move_label": move_label,
            "dir_prob": dir_prob,
            "dir_label": dir_label,
            "dir_mask": dir_mask,
            "mask": mask,
            "delta": delta,
            "eps": eps,
            "epsilon_mode": epsilon_mode,
        }

    data = collect_split_data(args.split)
    use_three_class = data["use_three_class"]
    logits3 = data["logits3"]
    move_prob = data["move_prob"]
    move_label = data["move_label"]
    dir_prob_full = data["dir_prob"]
    dir_label_full = data["dir_label"]
    dir_mask_full = data["dir_mask"]
    mask = data["mask"]
    delta = data["delta"]
    eps = data["eps"]
    epsilon_mode = data["epsilon_mode"]
    H = delta.shape[1]
    rows = []
    move_mcc = np.full(H, np.nan, dtype=np.float32)
    move_auc = np.full(H, np.nan, dtype=np.float32)
    move_brier = np.full(H, np.nan, dtype=np.float32)
    dir_mcc = np.full(H, np.nan, dtype=np.float32)
    dir_auc = np.full(H, np.nan, dtype=np.float32)
    dir_brier = np.full(H, np.nan, dtype=np.float32)
    dir_acc = np.full(H, np.nan, dtype=np.float32)
    dir_ece = np.full(H, np.nan, dtype=np.float32)
    dir_f1 = np.full(H, np.nan, dtype=np.float32)
    dir_tp = np.full(H, np.nan, dtype=np.float32)
    dir_tn = np.full(H, np.nan, dtype=np.float32)
    dir_fp = np.full(H, np.nan, dtype=np.float32)
    dir_fn = np.full(H, np.nan, dtype=np.float32)

    for h in range(H):
        m = mask[:, h]
        if not np.any(m):
            rows.append([h + 1, "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 0])
            continue
        if use_three_class:
            logits_h = logits3[m, h]
            # labels: 0=down, 1=flat, 2=up
            delta_h = delta[m, h]
            if epsilon_mode == "vol":
                eps_h = eps[m, h]
            else:
                eps_h = tau_h[h]
            flat = np.abs(delta_h) < eps_h
            up = delta_h >= eps_h
            labels = np.where(flat, 1, np.where(up, 2, 0)).astype(int)
            probs = np.exp(logits_h - np.max(logits_h, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
            dir_acc[h] = float(np.mean(preds == labels))
            # macro F1
            f1s = []
            for c in [0, 1, 2]:
                tp = np.sum((preds == c) & (labels == c))
                fp = np.sum((preds == c) & (labels != c))
                fn = np.sum((preds != c) & (labels == c))
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-8)
                f1s.append(f1)
            dir_f1[h] = float(np.mean(f1s))
            rows.append([h + 1, "nan", "nan", "nan", "nan", "nan", "nan", dir_acc[h], dir_f1[h], "nan", "nan", "nan", "nan", "nan", int(np.sum(m))])
        else:
            move_y = move_label[m, h]
            move_p = move_prob[m, h]
            move_pred = (move_p >= 0.5).astype(int)
            move_mcc[h] = _mcc(move_y, move_pred)
            move_auc[h] = _roc_auc(move_y, move_p)
            move_brier[h] = float(np.mean((move_p - move_y) ** 2))

            dir_mask = dir_mask_full[:, h]
            if np.any(dir_mask):
                dir_y = dir_label_full[dir_mask, h]
                dir_p = dir_prob_full[dir_mask, h]
                dir_pred = (dir_p >= 0.5).astype(int)
                tp, tn, fp, fn = _confusion(dir_y, dir_pred)
                dir_mcc[h] = _mcc(dir_y, dir_pred)
                dir_auc[h] = _roc_auc(dir_y, dir_p)
                dir_brier[h] = float(np.mean((dir_p - dir_y) ** 2))
                dir_acc[h] = float(np.mean(dir_pred == dir_y))
                dir_ece[h] = _ece(dir_y, dir_p)
                dir_tp[h] = tp
                dir_tn[h] = tn
                dir_fp[h] = fp
                dir_fn[h] = fn
            rows.append(
                [
                    h + 1,
                    move_mcc[h],
                    move_auc[h],
                    move_brier[h],
                    dir_mcc[h],
                    dir_auc[h],
                    dir_brier[h],
                    dir_acc[h],
                    dir_ece[h],
                    "nan",
                    dir_tp[h],
                    dir_tn[h],
                    dir_fp[h],
                    dir_fn[h],
                    int(np.sum(m)),
                ]
            )

    weights = default_horizon_weights(H)
    if args.horizons.strip():
        h_list = [h for h in _parse_ints(args.horizons) if 1 <= h <= H]
    else:
        h_list = list(range(1, H + 1))
    h_idx = np.asarray([h - 1 for h in h_list], dtype=np.int64)

    opt_thresholds = np.full(H, 0.5, dtype=np.float32)
    dir_mcc_opt = np.full(H, np.nan, dtype=np.float32)
    opt_split = args.opt_threshold_split.strip()
    if opt_split:
        opt_data = collect_split_data(opt_split)
        opt_dir_prob = opt_data["dir_prob"]
        opt_dir_label = opt_data["dir_label"]
        opt_dir_mask = opt_data["dir_mask"]
        steps = max(int(args.opt_threshold_steps), 3)
        grid = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        for h in range(H):
            m = opt_dir_mask[:, h]
            if not np.any(m):
                continue
            y = opt_dir_label[m, h]
            p = opt_dir_prob[m, h]
            best_mcc = float("-inf")
            best_thr = 0.5
            for thr in grid:
                pred = (p >= thr).astype(int)
                score = _mcc(y, pred)
                if np.isnan(score):
                    continue
                if score > best_mcc or (score == best_mcc and abs(thr - 0.5) < abs(best_thr - 0.5)):
                    best_mcc = score
                    best_thr = float(thr)
            opt_thresholds[h] = best_thr
        for h in range(H):
            m = dir_mask_full[:, h]
            if not np.any(m):
                continue
            y = dir_label_full[m, h]
            p = dir_prob_full[m, h]
            pred = (p >= opt_thresholds[h]).astype(int)
            dir_mcc_opt[h] = _mcc(y, pred)
    if use_three_class:
        # assume class order: 0=down, 1=flat, 2=up
        dir_acc_ud = np.full(H, np.nan, dtype=np.float32)
        for h in range(H):
            m = mask[:, h] > 0
            if not np.any(m):
                continue
            logits_h = logits3[m, h]
            probs = np.exp(logits_h - np.max(logits_h, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
            delta_h = delta[m, h]
            if epsilon_mode == "vol":
                eps_h = eps[m, h]
            else:
                eps_h = tau_h[h]
            flat = np.abs(delta_h) < eps_h
            up = delta_h >= eps_h
            labels = np.where(flat, 1, np.where(up, 2, 0)).astype(int)
            ud_mask = labels != 1
            if np.any(ud_mask):
                dir_acc_ud[h] = float(np.mean(preds[ud_mask] == labels[ud_mask]))
        valid = ~np.isnan(dir_f1[h_idx])
        w_f1 = float(np.sum(dir_f1[h_idx][valid] * weights[h_idx][valid]) / max(np.sum(weights[h_idx][valid]), 1e-6))
        metrics = {
            "dirscore_wF1": w_f1,
            "dir_f1_mean": float(np.nanmean(dir_f1[h_idx])),
            "dir_acc_mean": float(np.nanmean(dir_acc[h_idx])),
            "dir_acc_ud_mean": float(np.nanmean(dir_acc_ud[h_idx])),
            "delta_mode": args.delta_mode,
            "delta_quantile": args.delta_quantile,
            "horizons": h_list,
        }
    else:
        valid = ~np.isnan(dir_mcc[h_idx])
        w_mcc = float(np.sum(dir_mcc[h_idx][valid] * weights[h_idx][valid]) / max(np.sum(weights[h_idx][valid]), 1e-6))
        w_auc = float(np.sum(dir_auc[h_idx][valid] * weights[h_idx][valid]) / max(np.sum(weights[h_idx][valid]), 1e-6))
        tp_sum = int(np.nansum(dir_tp[h_idx]))
        tn_sum = int(np.nansum(dir_tn[h_idx]))
        fp_sum = int(np.nansum(dir_fp[h_idx]))
        fn_sum = int(np.nansum(dir_fn[h_idx]))
        metrics = {
            "dirscore_wMCC": w_mcc,
            "dirscore_wAUC": w_auc,
            "dir_mcc_mean": float(np.nanmean(dir_mcc[h_idx])),
            "dir_auc_mean": float(np.nanmean(dir_auc[h_idx])),
            "dir_brier_mean": float(np.nanmean(dir_brier[h_idx])),
            "dir_acc_mean": float(np.nanmean(dir_acc[h_idx])),
            "dir_ece_mean": float(np.nanmean(dir_ece[h_idx])),
            "move_mcc_mean": float(np.nanmean(move_mcc[h_idx])),
            "move_auc_mean": float(np.nanmean(move_auc[h_idx])),
            "move_brier_mean": float(np.nanmean(move_brier[h_idx])),
            "dir_tp_sum": tp_sum,
            "dir_tn_sum": tn_sum,
            "dir_fp_sum": fp_sum,
            "dir_fn_sum": fn_sum,
            "delta_mode": args.delta_mode,
            "delta_quantile": args.delta_quantile,
            "horizons": h_list,
        }
    if opt_split:
        valid_opt = ~np.isnan(dir_mcc_opt[h_idx])
        w_mcc_opt = float(np.sum(dir_mcc_opt[h_idx][valid_opt] * weights[h_idx][valid_opt]) / max(np.sum(weights[h_idx][valid_opt]), 1e-6))
        metrics["dirscore_wMCC_opt"] = w_mcc_opt
        metrics["dir_mcc_opt_mean"] = float(np.nanmean(dir_mcc_opt[h_idx]))
        metrics["opt_threshold_split"] = opt_split
        metrics["opt_threshold_steps"] = steps

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "h",
                "move_mcc",
                "move_auc",
                "move_brier",
                "dir_mcc",
                "dir_auc",
                "dir_brier",
                "dir_acc",
                "dir_ece",
                "dir_f1",
                "dir_tp",
                "dir_tn",
                "dir_fp",
                "dir_fn",
                "n",
            ]
        )
        writer.writerows(rows)

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_metrics,
        metrics=metrics,
        dir_mcc=dir_mcc,
        dir_auc=dir_auc,
        dir_brier=dir_brier,
        dir_acc=dir_acc,
        dir_ece=dir_ece,
        dir_tp=dir_tp,
        dir_tn=dir_tn,
        dir_fp=dir_fp,
        dir_fn=dir_fn,
        dir_mcc_opt=dir_mcc_opt,
        opt_thresholds=opt_thresholds,
    )
    print(metrics)


if __name__ == "__main__":
    main()
