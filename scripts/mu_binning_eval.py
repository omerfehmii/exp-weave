from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backtest.harness import make_time_splits
from data.loader import compress_series_observed, load_panel_npz
from regime import REGIME_NAMES
from eval import apply_scaling
from utils import load_config


def _load_series(cfg: Dict) -> list:
    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    return series_list


def _bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([values.min(), values.max()])
    return edges


def _bin_stats(mu: np.ndarray, y: np.ndarray, edges: np.ndarray) -> list[dict]:
    out = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            m = (mu >= lo) & (mu <= hi)
        else:
            m = (mu >= lo) & (mu < hi)
        n = int(np.sum(m))
        if n == 0:
            out.append(
                {
                    "bin": i,
                    "mu_lo": float(lo),
                    "mu_hi": float(hi),
                    "n": 0,
                    "mean_mu": float("nan"),
                    "mean_ret": float("nan"),
                    "hit_rate": float("nan"),
                }
            )
            continue
        mu_m = mu[m]
        y_m = y[m]
        hit = float(np.mean(np.sign(mu_m) == np.sign(y_m)))
        out.append(
            {
                "bin": i,
                "mu_lo": float(lo),
                "mu_hi": float(hi),
                "n": n,
                "mean_mu": float(np.mean(mu_m)),
                "mean_ret": float(np.mean(y_m)),
                "hit_rate": hit,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--h", type=int, default=24)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--value_scale", default="orig", choices=["orig", "scaled"])
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_mode = cfg.get("data", {}).get("target_mode", "level")
    use_return_target = target_mode != "level"

    preds = np.load(args.preds)
    y = preds["y"]
    q50 = preds["q50"]
    mask = preds["mask"] if "mask" in preds else np.isfinite(y).astype(np.float32)
    regime = preds["regime"] if "regime" in preds else None

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
    mu = q50[valid, h_idx]
    ret = y[valid, h_idx]
    if mu.size == 0:
        raise RuntimeError("No valid samples for requested horizon.")

    edges = _bin_edges(mu, args.bins)
    summary = _bin_stats(mu, ret, edges)

    print(f"mu_binning h={args.h} bins={args.bins} n={mu.size}")
    for row in summary:
        print(
            f"bin={row['bin']} n={row['n']} mu[{row['mu_lo']:.4f},{row['mu_hi']:.4f}] "
            f"mean_mu={row['mean_mu']:.6f} mean_ret={row['mean_ret']:.6f} hit={row['hit_rate']:.3f}"
        )

    rows = []
    for row in summary:
        rows.append({"regime": "ALL", **row})

    if regime is not None:
        for r in np.unique(regime):
            idx = valid & (regime == r)
            mu_r = q50[idx, h_idx]
            ret_r = y[idx, h_idx]
            if mu_r.size == 0:
                continue
            reg_stats = _bin_stats(mu_r, ret_r, edges)
            name = REGIME_NAMES[int(r)] if 0 <= int(r) < len(REGIME_NAMES) else f"REG_{int(r)}"
            for row in reg_stats:
                rows.append({"regime": name, **row})

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("regime,bin,mu_lo,mu_hi,n,mean_mu,mean_ret,hit_rate\n")
            for row in rows:
                f.write(
                    f"{row['regime']},{row['bin']},{row['mu_lo']:.8f},{row['mu_hi']:.8f},"
                    f"{row['n']},{row['mean_mu']:.8f},{row['mean_ret']:.8f},{row['hit_rate']:.6f}\n"
                )


if __name__ == "__main__":
    main()
