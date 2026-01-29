#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

# Ensure repo root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_panel_npz


@dataclass
class FoldSpec:
    fold: int
    train_end_t: int
    val_end_t: int
    test_end_t: int
    train_end_idx: int
    val_end_idx: int
    test_start_idx: int
    test_end_idx: int


def _load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_cfg(cfg: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _compute_origin_info(cfg: Dict) -> Tuple[int, int, int, int]:
    data = cfg["data"]
    L = int(data["L"])
    H = int(data["H"])
    step = int(data.get("step", H))
    series = load_panel_npz(str(data["path"]))
    lengths = [len(s.y) for s in series]
    T = min(lengths)
    origin_min = L - 1
    origin_max = T - H - 1
    if origin_max < origin_min:
        raise ValueError("Not enough history for given L/H.")
    n_origins = (origin_max - origin_min) // step + 1
    return T, origin_min, origin_max, n_origins


def _compute_valid_origin_max(cfg: Dict, horizon: int) -> int:
    data = cfg["data"]
    L = int(data["L"])
    H = int(data["H"])
    step = int(data.get("step", H))
    series = load_panel_npz(str(data["path"]))
    T = min(len(s.y) for s in series)
    # Generate all origin candidates.
    origins = []
    t = L - 1
    last = T - H - 1
    while t <= last:
        origins.append(t)
        t += step
    valid = []
    for t in origins:
        any_valid = False
        for s in series:
            y = s.y
            if y.ndim == 1:
                y = y[:, None]
            if t + horizon < y.shape[0] and np.isfinite(y[t + horizon]).all():
                any_valid = True
                break
        if any_valid:
            valid.append(t)
    if not valid:
        raise ValueError("No valid origins with observed horizon.")
    return valid[-1]


def _build_folds(n_origins: int, origin_min: int, step: int, fold_size: int, n_folds: int, val_size: int) -> List[FoldSpec]:
    start_test_idx = n_origins - n_folds * fold_size
    if start_test_idx <= val_size:
        raise ValueError("Not enough origins for requested folds/val_size.")
    folds: List[FoldSpec] = []
    for k in range(n_folds):
        test_start_idx = start_test_idx + k * fold_size
        test_end_idx = test_start_idx + fold_size - 1
        val_end_idx = test_start_idx - 1
        val_start_idx = max(0, val_end_idx - val_size + 1)
        train_end_idx = val_start_idx - 1
        if train_end_idx < 0:
            raise ValueError("Train window empty. Reduce val_size or number of folds.")
        train_end_t = origin_min + train_end_idx * step + 1
        val_end_t = origin_min + val_end_idx * step + 1
        test_end_t = origin_min + test_end_idx * step + 1
        folds.append(
            FoldSpec(
                fold=k,
                train_end_t=train_end_t,
                val_end_t=val_end_t,
                test_end_t=test_end_t,
                train_end_idx=train_end_idx,
                val_end_idx=val_end_idx,
                test_start_idx=test_start_idx,
                test_end_idx=test_end_idx,
            )
        )
    return folds


def _policy_args_dyn_cap_2(policy_config: Path, preds_path: Path, out_csv: Path, out_metrics: Path) -> List[str]:
    return [
        "scripts/mu_value_weighted_backtest.py",
        "--config",
        str(policy_config),
        "--preds",
        str(preds_path),
        "--h",
        "24",
        "--use_cs",
        "--ret_cs",
        "--disp_metric",
        "std",
        "--disp_hist_window",
        "200",
        "--disp_scale_q_low",
        "0.15",
        "--disp_scale_q_high",
        "0.40",
        "--disp_scale_floor",
        "0.25",
        "--disp_scale_power",
        "2.0",
        "--disagree_q_low",
        "0.4",
        "--disagree_q_high",
        "0.7",
        "--disagree_hist_window",
        "200",
        "--disagree_scale",
        "0.3",
        "--consistency_min",
        "0.05",
        "--consistency_scale",
        "0.6",
        "--gate_combine",
        "avg",
        "--gate_avg_weights",
        "0.45,0.35,0.20",
        "--ema_halflife_min",
        "2",
        "--ema_halflife_max",
        "8",
        "--ema_disp_lo",
        "0.02",
        "--ema_disp_hi",
        "0.10",
        "--min_hold",
        "2",
        "--turnover_budget",
        "0.15",
        "--pos_cap",
        "0.04",
        "--gross_target",
        "1.0",
        "--optimize",
        "--opt_lambda",
        "1.0",
        "--opt_kappa",
        "0.0",
        "--opt_steps",
        "20",
        "--opt_risk_window",
        "240",
        "--opt_dollar_neutral",
        "--topn_cap",
        "0.15",
        "--topn_cap_low",
        "0.12",
        "--topn_n",
        "10",
        "--topn_dyn_q_hi",
        "0.85",
        "--topn_dyn_q_lo",
        "0.70",
        "--shock_metric",
        "p90",
        "--shock_hist_window",
        "200",
        "--out_csv",
        str(out_csv),
        "--out_metrics",
        str(out_metrics),
    ]


def _summarize_metrics(path: Path) -> Dict[str, float]:
    import pandas as pd

    df = pd.read_csv(path)
    pnl = df["pnl"].to_numpy()
    if pnl.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sharpe": float("nan"),
            "sharpe_gross": float("nan"),
            "sharpe_vol": float("nan"),
            "top5_sum": float("nan"),
            "top5_over_total": float("nan"),
        }
    gross = df["gross"].to_numpy() if "gross" in df.columns else None
    vol = df["vol_mkt"].to_numpy() if "vol_mkt" in df.columns else None
    mean = float(np.mean(pnl))
    std = float(np.std(pnl))
    sharpe = mean / std if std > 0 else float("nan")
    gross_sharpe = float("nan")
    vol_sharpe = float("nan")
    if gross is not None:
        r = pnl / np.clip(gross, 1e-8, None)
        gross_sharpe = float(np.mean(r) / np.std(r)) if np.std(r) > 0 else float("nan")
    if vol is not None:
        r = pnl / np.clip(vol, 1e-8, None)
        vol_sharpe = float(np.mean(r) / np.std(r)) if np.std(r) > 0 else float("nan")
    s = np.sort(pnl)[::-1]
    top5 = float(s[:5].sum()) if len(s) >= 5 else float(s.sum())
    total = float(pnl.sum())
    return {
        "n": int(len(pnl)),
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "sharpe_gross": gross_sharpe,
        "sharpe_vol": vol_sharpe,
        "top5_sum": top5,
        "top5_over_total": top5 / total if total != 0 else float("nan"),
    }


def _ic_count_stats(
    preds_path: Path, horizon: int
) -> Tuple[Optional[float], Optional[float], Optional[float], np.ndarray]:
    with np.load(preds_path) as d:
        if "mask" not in d or "origin_t" not in d:
            return None, None, None, np.array([], dtype=np.int64)
        mask = d["mask"]
        origin_t = d["origin_t"]
        q50 = d["q50"] if "q50" in d else None
        if mask.ndim != 2 or horizon < 1 or horizon > mask.shape[1]:
            return None, None, None, np.array([], dtype=np.int64)
        h_idx = horizon - 1
        valid = mask[:, h_idx] > 0
        times = origin_t[valid]
        if times.size == 0:
            return None, None, None, np.array([], dtype=np.int64)
        _, counts = np.unique(times, return_counts=True)
        if counts.size == 0:
            return None, None, None, np.array([], dtype=np.int64)
        mean = float(np.mean(counts))
        p10 = float(np.percentile(counts, 10))
        ge20 = float(np.mean(counts >= 20))
        return mean, p10, ge20, counts


def _ignore_ratio(preds_path: Path) -> Optional[float]:
    with np.load(preds_path) as d:
        if "ignore" in d:
            return float(np.mean(d["ignore"]))
    return None


def _coverage_stats(preds_path: Path, horizon: int) -> Dict[str, float]:
    with np.load(preds_path) as d:
        if "mask" not in d or "origin_t" not in d or "q50" not in d:
            return {}
        mask = d["mask"]
        origin_t = d["origin_t"]
        q50 = d["q50"]
        if mask.ndim != 2 or horizon < 1 or horizon > mask.shape[1]:
            return {}
        h_idx = horizon - 1
        valid = mask[:, h_idx] > 0
        if not np.any(valid):
            return {}
        times = origin_t[valid]
        # active_count: mask>0 at horizon
        time_vals, active_counts = np.unique(times, return_counts=True)
        # ic_count: mask>0 AND finite q50 at horizon
        if q50.ndim == 2 and q50.shape[1] > h_idx:
            ic_valid = valid & np.isfinite(q50[:, h_idx])
        else:
            ic_valid = valid
        ic_times = origin_t[ic_valid]
        if ic_times.size:
            _, ic_counts = np.unique(ic_times, return_counts=True)
        else:
            ic_counts = np.array([], dtype=np.int64)
        stats = {
            "active_count_mean": float(np.mean(active_counts)),
            "active_count_p10": float(np.percentile(active_counts, 10)),
            "active_count_median": float(np.median(active_counts)),
            "ic_count_mean": float(np.mean(ic_counts)) if ic_counts.size else float("nan"),
            "ic_count_p10": float(np.percentile(ic_counts, 10)) if ic_counts.size else float("nan"),
            "ic_count_median": float(np.median(ic_counts)) if ic_counts.size else float("nan"),
            "ic_count_ge20_ratio": float(np.mean(ic_counts >= 20)) if ic_counts.size else float("nan"),
        }
        # Save time-level coverage for fold
        cov_rows = []
        ic_map = {}
        if ic_counts.size:
            _, ic_counts_full = np.unique(ic_times, return_counts=True)
            # align by time_vals order
            ic_map = {int(t): int(c) for t, c in zip(np.unique(ic_times), ic_counts_full)}
        for t, a in zip(time_vals, active_counts):
            cov_rows.append((int(t), int(a), int(ic_map.get(int(t), 0))))
        stats["_coverage_rows"] = cov_rows
        return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--policy_config", required=True)
    parser.add_argument("--out_dir", default="runs/wf")
    parser.add_argument("--fold_size", type=int, default=32, help="Test block size in origins.")
    parser.add_argument("--n_folds", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=64, help="Validation block size in origins.")
    parser.add_argument("--seeds", default="7,13,21,42,77")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--policy_preset", default="dynCap2")
    parser.add_argument("--min_future_obs", type=int, default=None, help="Override data.min_future_obs for folds.")
    parser.add_argument("--min_past_obs", type=int, default=None, help="Override data.min_past_obs for folds.")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config)
    policy_cfg_path = Path(args.policy_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(base_cfg_path)
    data = cfg["data"]
    horizon = int(data["H"])
    step = int(data.get("step", data["H"]))
    T, origin_min, origin_max, n_origins = _compute_origin_info(cfg)
    # Restrict to origins with observed horizon to avoid empty test windows.
    try:
        origin_max_valid = _compute_valid_origin_max(cfg, horizon)
        if origin_max_valid < origin_max:
            origin_max = origin_max_valid
            n_origins = (origin_max - origin_min) // step + 1
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not compute valid origin max: {exc}")

    folds = _build_folds(n_origins, origin_min, step, args.fold_size, args.n_folds, args.val_size)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    py = str(Path.cwd() / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = "python"

    summary_rows: List[Dict] = []
    combined_metrics_paths: List[Path] = []
    ic_counts_all: List[np.ndarray] = []
    ignore_vals: List[float] = []
    coverage_all: List[Dict[str, float]] = []

    for fold in folds:
        fold_dir = out_dir / f"fold_{fold.fold}"
        (fold_dir / "configs").mkdir(parents=True, exist_ok=True)
        (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (fold_dir / "preds").mkdir(parents=True, exist_ok=True)
        (fold_dir / "logs").mkdir(parents=True, exist_ok=True)

        meta = {
            "fold": fold.fold,
            "train_end_t": fold.train_end_t,
            "val_end_t": fold.val_end_t,
            "test_end_t": fold.test_end_t,
            "train_end_idx": fold.train_end_idx,
            "val_end_idx": fold.val_end_idx,
            "test_start_idx": fold.test_start_idx,
            "test_end_idx": fold.test_end_idx,
            "origin_min": origin_min,
            "origin_max": origin_max,
            "n_origins": n_origins,
            "T_min": T,
        }
        # Protocol checklist per fold
        series_list = load_panel_npz(str(cfg["data"]["path"]))
        ts = series_list[0].timestamps if series_list and series_list[0].timestamps is not None else None
        protocol = {
            "fold": fold.fold,
            "train_range_idx": [0, fold.train_end_idx],
            "val_range_idx": [fold.train_end_idx + 1, fold.val_end_idx],
            "test_range_idx": [fold.test_start_idx, fold.test_end_idx],
            "train_end_t": fold.train_end_t,
            "val_end_t": fold.val_end_t,
            "test_end_t": fold.test_end_t,
            "train_end_ts": str(ts[fold.train_end_t]) if ts is not None else None,
            "val_end_ts": str(ts[fold.val_end_t]) if ts is not None else None,
            "test_end_ts": str(ts[fold.test_end_t]) if ts is not None else None,
            "purge_len": int(cfg["data"].get("split_purge", 0)),
            "embargo_len": int(cfg["data"].get("split_embargo", 0)),
            "horizon": horizon,
            "step": step,
            "scaler_train_only": True,
            "decision_time": cfg["data"].get("decision_time", "close"),
            "feature_lag": int(cfg["data"].get("feature_lag", 0)),
        }
        with (fold_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with (fold_dir / "protocol.json").open("w", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2)

        preds_paths: List[Path] = []
        for seed in seeds:
            cfg_fold = _load_cfg(base_cfg_path)
            cfg_fold.setdefault("data", {})
            cfg_fold["data"]["split_train_end"] = fold.train_end_t
            # Extend val/test split ends by horizon so test targets are in-range.
            cfg_fold["data"]["split_val_end"] = fold.val_end_t + horizon
            cfg_fold["data"]["split_test_end"] = fold.test_end_t + horizon
            # Ensure the eval horizon is observed (avoid all-zero mask at h=H).
            cfg_fold["data"]["min_future_obs"] = horizon
            if args.min_future_obs is not None:
                cfg_fold["data"]["min_future_obs"] = int(args.min_future_obs)
            if args.min_past_obs is not None:
                cfg_fold["data"]["min_past_obs"] = int(args.min_past_obs)
            cfg_fold.setdefault("training", {})
            cfg_fold["training"]["seed"] = seed
            ckpt = fold_dir / "checkpoints" / f"cs_l1_w10_s{seed}.pt"
            log_path = fold_dir / "logs" / f"train_s{seed}.jsonl"
            cfg_fold["training"]["output_path"] = str(ckpt)
            cfg_fold["training"]["log_path"] = str(log_path)
            cfg_path = fold_dir / "configs" / f"cs_l1_w10_s{seed}.yaml"
            _save_cfg(cfg_fold, cfg_path)

            if not (args.skip_existing and ckpt.exists()):
                subprocess.run([py, "train.py", "--config", str(cfg_path)], check=True)

            preds_path = fold_dir / "preds" / f"preds_s{seed}.npz"
            if not (args.skip_existing and preds_path.exists()):
                subprocess.run(
                    [
                        py,
                        "eval.py",
                        "--config",
                        str(cfg_path),
                        "--checkpoint",
                        str(ckpt),
                        "--split",
                        "test",
                        "--out_npz",
                        str(preds_path),
                    ],
                    check=True,
                )
            preds_paths.append(preds_path)

        ens_path = fold_dir / "ens_preds.npz"
        if not (args.skip_existing and ens_path.exists()):
            subprocess.run([py, "scripts/ensemble_preds.py", "--out", str(ens_path), "--preds", *map(str, preds_paths)], check=True)

        out_csv = fold_dir / "backtest.csv"
        out_metrics = fold_dir / "metrics.csv"
        if not (args.skip_existing and out_metrics.exists()):
            if args.policy_preset.lower() != "dyncap2":
                raise ValueError("Only policy_preset=dynCap2 is supported right now.")
            cmd = [py, *_policy_args_dyn_cap_2(policy_cfg_path, ens_path, out_csv, out_metrics)]
            subprocess.run(cmd, check=True)

        fold_summary = _summarize_metrics(out_metrics)
        ic_mean, ic_p10, ic_ge20, counts = _ic_count_stats(ens_path, horizon)
        if counts.size:
            ic_counts_all.append(counts)
        fold_summary["ic_count_mean"] = ic_mean if ic_mean is not None else float("nan")
        fold_summary["ic_count_p10"] = ic_p10 if ic_p10 is not None else float("nan")
        fold_summary["ic_count_ge20_ratio"] = ic_ge20 if ic_ge20 is not None else float("nan")
        cov = _coverage_stats(ens_path, horizon)
        if cov:
            coverage_all.append(cov)
            fold_summary.update(
                {
                    "active_count_mean": cov.get("active_count_mean"),
                    "active_count_p10": cov.get("active_count_p10"),
                    "active_count_median": cov.get("active_count_median"),
                    "ic_count_median": cov.get("ic_count_median"),
                }
            )
            rows = cov.get("_coverage_rows", [])
            if rows:
                cov_path = fold_dir / "coverage.csv"
                with cov_path.open("w", encoding="utf-8") as f:
                    f.write("time_key,active_count,ic_count\n")
                    for t, a, ic in rows:
                        f.write(f"{t},{a},{ic}\n")
        # Fail-fast warning flag
        if fold_summary.get("ic_count_ge20_ratio", float("nan")) != fold_summary.get("ic_count_ge20_ratio", 0.0):
            fold_summary["coverage_warn"] = True
        else:
            fold_summary["coverage_warn"] = fold_summary["ic_count_ge20_ratio"] < 0.5
        ignore_ratio = _ignore_ratio(ens_path)
        if ignore_ratio is not None:
            ignore_vals.append(ignore_ratio)
            fold_summary["ignore_ratio"] = ignore_ratio
        else:
            fold_summary["ignore_ratio"] = float("nan")
        fold_summary["cost_included"] = False
        fold_summary["fold"] = fold.fold
        summary_rows.append(fold_summary)
        combined_metrics_paths.append(out_metrics)

    # write summary
    import pandas as pd

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # combined metrics
    dfs = [pd.read_csv(p) for p in combined_metrics_paths]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_dir / "metrics_all.csv", index=False)
    comb = _summarize_metrics(out_dir / "metrics_all.csv")
    ic_all = np.concatenate(ic_counts_all) if ic_counts_all else np.array([], dtype=np.int64)
    comb["ic_count_mean"] = float(np.mean(ic_all)) if ic_all.size else float("nan")
    comb["ic_count_p10"] = float(np.percentile(ic_all, 10)) if ic_all.size else float("nan")
    comb["ic_count_ge20_ratio"] = float(np.mean(ic_all >= 20)) if ic_all.size else float("nan")
    if coverage_all:
        comb["active_count_mean"] = float(np.mean([c["active_count_mean"] for c in coverage_all if "active_count_mean" in c]))
        comb["active_count_p10"] = float(np.mean([c["active_count_p10"] for c in coverage_all if "active_count_p10" in c]))
        comb["active_count_median"] = float(np.mean([c["active_count_median"] for c in coverage_all if "active_count_median" in c]))
        comb["ic_count_median"] = float(np.mean([c["ic_count_median"] for c in coverage_all if "ic_count_median" in c]))
    else:
        comb["active_count_mean"] = float("nan")
        comb["active_count_p10"] = float("nan")
        comb["active_count_median"] = float("nan")
        comb["ic_count_median"] = float("nan")
    comb["ignore_ratio"] = float(np.mean(ignore_vals)) if ignore_vals else float("nan")
    comb["cost_included"] = False
    comb_path = out_dir / "summary_all.json"
    with comb_path.open("w", encoding="utf-8") as f:
        json.dump(comb, f, indent=2)
    wf_summary = {
        "n_folds": len(folds),
        "coverage_warn_folds": int(np.sum([1 for row in summary_rows if row.get("coverage_warn")])),
        "protocol": {
            "purge_len": int(cfg["data"].get("split_purge", 0)),
            "embargo_len": int(cfg["data"].get("split_embargo", 0)),
            "horizon": horizon,
            "step": step,
            "scaler_train_only": True,
            "decision_time": cfg["data"].get("decision_time", "close"),
            "feature_lag": int(cfg["data"].get("feature_lag", 0)),
        },
        "coverage_summary": {
            "active_count_mean": comb["active_count_mean"],
            "active_count_p10": comb["active_count_p10"],
            "active_count_median": comb["active_count_median"],
            "ic_count_mean": comb["ic_count_mean"],
            "ic_count_p10": comb["ic_count_p10"],
            "ic_count_median": comb["ic_count_median"],
            "ic_count_ge20_ratio": comb["ic_count_ge20_ratio"],
        },
    }
    with (out_dir / "wf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(wf_summary, f, indent=2)


if __name__ == "__main__":
    main()
