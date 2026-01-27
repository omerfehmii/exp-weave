from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

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
    parser.add_argument("--ema_halflife", type=float, default=0.0)
    parser.add_argument("--min_hold", type=int, default=0)
    parser.add_argument("--turnover_cap", type=float, default=0.0)
    parser.add_argument("--walk_folds", type=int, default=0)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("data", {}).get("target_mode", "level")
    use_return_target = target_mode != "level"

    preds = np.load(args.preds)
    y = preds["y"]
    q50 = preds["q50"]
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
    ret = y[:, h_idx]

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
    if args.ema_halflife and args.ema_halflife > 0:
        alpha = 1.0 - 0.5 ** (1.0 / args.ema_halflife)
    else:
        alpha = None
    pnl_time = []
    turnover_time = []
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
        assets = series_idx[idx]
        if alpha is not None:
            for j, a in enumerate(assets):
                ema_state[a] = (1.0 - alpha) * ema_state[a] + alpha * mu_t[j]
                mu_t[j] = ema_state[a]
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
        denom_full = np.sum(np.abs(w_full[assets]))
        if denom_full > 1e-12:
            w_full = w_full / denom_full
        pnl_time.append(float(np.sum(w_full[assets] * ret_t)))
        turnover_time.append(float(np.sum(np.abs(w_full - prev_w))))
        prev_w = w_full

    pnl_time = np.asarray(pnl_time, dtype=np.float64)
    print(
        f"mu_value_weighted h={args.h} mode={args.weight_mode} "
        f"cs={args.use_cs} ret_cs={args.ret_cs} topk={args.topk} n_times={pnl_time.size}"
    )
    _summarize("portfolio", pnl_time)
    if turnover_time:
        avg_turnover = float(np.mean(turnover_time))
        print(f"turnover_mean={avg_turnover:.6f}")

    if args.walk_folds and args.walk_folds > 1 and pnl_time.size > 0:
        folds = int(args.walk_folds)
        fold_size = max(1, pnl_time.size // folds)
        print(f"walk_folds={folds} fold_size={fold_size}")
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
            print(f"fold{i}: mean={mean:.6f} sharpe={sharpe:.3f} hit={hit:.3f} n={chunk.size}")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("pnl\n")
            for v in pnl_time:
                f.write(f"{v:.8f}\n")


if __name__ == "__main__":
    main()
