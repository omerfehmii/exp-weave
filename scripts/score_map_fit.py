import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import load_config
from eval import apply_scaling
from backtest.harness import make_time_splits
from data.loader import load_panel_npz, compress_series_observed


def _load_series(cfg):
    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    return series_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--score_transform", default="none", choices=["none", "cs_zscore"])
    parser.add_argument("--use_cs", action="store_true")
    parser.add_argument("--ret_cs", action="store_true")
    parser.add_argument("--out_csv", required=True)
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
        raise ValueError("preds must include origin_t and series_idx.")

    # compute time_key (origin-based)
    time_key = origin_t
    rows = []
    for t in np.unique(time_key):
        idx = np.where((time_key == t) & valid)[0]
        if idx.size == 0:
            continue
        mu_t = mu[idx]
        ret_t = ret[idx]
        if args.use_cs:
            mu_t = mu_t - np.mean(mu_t)
        if args.ret_cs:
            ret_t = ret_t - np.mean(ret_t)
        if args.score_transform == "cs_zscore":
            std = np.std(mu_t)
            if std > 1e-12:
                mu_t = (mu_t - np.mean(mu_t)) / std
        rows.append((mu_t, ret_t))

    if not rows:
        raise RuntimeError("no valid rows for mapping")

    all_scores = np.concatenate([r[0] for r in rows], axis=0)
    edges = np.quantile(all_scores, np.linspace(0, 1, args.bins + 1))
    out = []
    for i in range(args.bins):
        lo = edges[i]
        hi = edges[i + 1]
        vals = []
        for mu_t, ret_t in rows:
            mask_bin = (mu_t >= lo) & (mu_t < hi)
            if np.any(mask_bin):
                vals.append(np.mean(ret_t[mask_bin]))
        mean_ret = float(np.mean(vals)) if vals else 0.0
        out.append((lo, hi, mean_ret))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("bin_lo,bin_hi,score\n")
        for lo, hi, mean_ret in out:
            f.write(f"{lo:.8f},{hi:.8f},{mean_ret:.8f}\n")


if __name__ == "__main__":
    main()
