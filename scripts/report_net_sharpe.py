#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _sharpe(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    std = float(np.std(x))
    return float(np.mean(x) / std) if std > 0 else float("nan")


def _summarize_metrics(path: Path, cost_bps: float) -> dict:
    df = pd.read_csv(path)
    if df.empty:
        return {"n": 0}
    pnl = df["pnl"].to_numpy(dtype=np.float64)
    gross = df["gross"].to_numpy(dtype=np.float64) if "gross" in df.columns else None
    turnover = df["turnover"].to_numpy(dtype=np.float64) if "turnover" in df.columns else None
    if turnover is None:
        turnover = np.zeros_like(pnl)
    cost = turnover * (float(cost_bps) / 1e4)
    net = pnl - cost

    gross_sharpe = float("nan")
    net_sharpe = float("nan")
    if gross is not None:
        gross_ret = pnl / np.clip(gross, 1e-12, None)
        net_ret = net / np.clip(gross, 1e-12, None)
        gross_sharpe = _sharpe(gross_ret)
        net_sharpe = _sharpe(net_ret)
    else:
        gross_sharpe = _sharpe(pnl)
        net_sharpe = _sharpe(net)

    return {
        "n": int(pnl.size),
        "sharpe_gross": gross_sharpe,
        "sharpe_net": net_sharpe,
        "turnover_mean": float(np.mean(turnover)) if turnover.size else float("nan"),
    }


def _find_metrics_files(run_dir: Path) -> List[Path]:
    files = []
    for p in run_dir.glob("fold_*/metrics.csv"):
        files.append(p)
    if not files and (run_dir / "metrics_all.csv").exists():
        files.append(run_dir / "metrics_all.csv")
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--cost_bps", type=float, default=10.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    files = _find_metrics_files(run_dir)
    if not files:
        raise SystemExit("No metrics.csv found under run_dir.")

    rows = []
    for p in files:
        row = _summarize_metrics(p, args.cost_bps)
        row["path"] = str(p)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
