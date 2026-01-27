from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.loader import WindowedDataset, compress_series_observed, load_panel_npz
from eval import apply_scaling
from train import build_model, filter_indices_with_observed
from utils import load_config, set_seed


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _summarize(side: np.ndarray, pnl: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    trade = valid & (side != 0)
    n_trades = int(np.sum(trade))
    hit_rate = float(np.mean(pnl[trade] > 0)) if n_trades else float("nan")
    mean_pnl = float(np.mean(pnl[trade])) if n_trades else float("nan")
    std_pnl = float(np.std(pnl[trade])) if n_trades else float("nan")
    sharpe = mean_pnl / (std_pnl + 1e-12) if n_trades else float("nan")
    net_pnl_sum = float(np.sum(pnl[trade])) if n_trades else 0.0
    return {
        "n_trades": n_trades,
        "hit_rate": hit_rate,
        "mean_pnl_trade": mean_pnl,
        "std_pnl_trade": std_pnl,
        "sharpe_trade": sharpe,
        "net_pnl_sum": net_pnl_sum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--tau", type=float, default=0.55)
    parser.add_argument("--move_tau", type=float, default=0.0)
    parser.add_argument("--out_summary", default=None)
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
    indices = generate_panel_origins(lengths, cfg["data"]["L"], cfg["data"]["H"], cfg["data"].get("step", cfg["data"]["H"]))
    split_idx = select_indices_by_time(indices, split, args.split, horizon=cfg["data"]["H"])
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

    device_str = cfg["training"].get("device", "cpu")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        print("warning: cuda requested but not available; falling back to cpu")
        device_str = "cpu"
    device = torch.device(device_str)
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    logits3_list = []
    move_logits_list = []
    dir_logits_list = []
    y_future_list = []
    y_last_list = []
    mask_list = []
    origin_mask_list = []
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

    y_future = np.concatenate(y_future_list, axis=0)
    y_last = np.concatenate(y_last_list, axis=0)
    mask = np.concatenate(mask_list, axis=0) > 0
    origin_mask = np.concatenate(origin_mask_list, axis=0) > 0
    mask = mask & origin_mask[:, None]

    if logits3_list:
        logits3 = np.concatenate(logits3_list, axis=0)
        probs = np.exp(logits3 - np.max(logits3, axis=2, keepdims=True))
        probs = probs / np.sum(probs, axis=2, keepdims=True)
        p_down = probs[:, :, 0]
        p_flat = probs[:, :, 1]
        p_up = probs[:, :, 2]
        dir_prob = p_up / np.maximum(p_up + p_down, 1e-6)
        move_prob = 1.0 - p_flat
    else:
        move_logits = np.concatenate(move_logits_list, axis=0)
        dir_logits = np.concatenate(dir_logits_list, axis=0)
        move_prob = _sigmoid(move_logits)
        dir_prob = _sigmoid(dir_logits)

    h_idx = args.h - 1
    if h_idx < 0 or h_idx >= y_future.shape[1]:
        raise ValueError("Invalid horizon index.")

    ref = y_last
    delta = y_future[:, h_idx] - ref
    valid = mask[:, h_idx]
    side = np.zeros_like(delta, dtype=np.int8)
    if args.move_tau > 0.0:
        valid = valid & (move_prob[:, h_idx] >= args.move_tau)
    side[(dir_prob[:, h_idx] >= args.tau) & valid] = 1
    side[(dir_prob[:, h_idx] <= (1.0 - args.tau)) & valid] = -1
    pnl = side * delta

    summary = _summarize(side, pnl, valid)
    print(
        "dir_threshold_backtest",
        f"h={args.h}",
        f"tau={args.tau}",
        f"move_tau={args.move_tau}",
        f"n_trades={summary['n_trades']}",
        f"net_pnl_sum={summary['net_pnl_sum']}",
        f"hit_rate={summary['hit_rate']}",
        f"mean_pnl_trade={summary['mean_pnl_trade']}",
    )

    if args.out_summary:
        out_path = Path(args.out_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("metric,value\n")
            for k, v in summary.items():
                f.write(f"{k},{v}\n")


if __name__ == "__main__":
    main()
