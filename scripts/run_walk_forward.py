#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

import numpy as np
import yaml

# Ensure repo root is on sys.path when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import load_panel_npz, compress_series_observed, filter_series_by_active_ratio, filter_series_by_future_ratio


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


def _load_series(cfg: Dict) -> List:
    series = load_panel_npz(str(cfg["data"]["path"]))
    if cfg.get("data", {}).get("observed_only", False):
        series = compress_series_observed(series)
    min_ratio = float(cfg.get("data", {}).get("universe_min_active_ratio", 0.0))
    min_points = int(cfg.get("data", {}).get("universe_min_active_points", 0))
    active_end = cfg.get("data", {}).get("universe_active_end")
    min_future_ratio = float(cfg.get("data", {}).get("universe_min_future_ratio", 0.0))
    future_horizon = int(cfg.get("data", {}).get("H", cfg.get("data", {}).get("future_horizon", 0)))
    if min_ratio or min_points:
        series = filter_series_by_active_ratio(series, min_ratio, min_points, active_end)
    if min_future_ratio and future_horizon > 0:
        series = filter_series_by_future_ratio(series, min_future_ratio, future_horizon, active_end)
    return series


def _compute_origin_info(cfg: Dict) -> Tuple[int, int, int, int]:
    data = cfg["data"]
    L = int(data["L"])
    H = int(data["H"])
    step = int(data.get("step", H))
    series = _load_series(cfg)
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
    series = _load_series(cfg)
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


def _policy_params_dyn_cap_2(horizon: int) -> Dict[str, str]:
    return {
        "h": str(int(horizon)),
        "disp_metric": "std",
        "disp_hist_window": "200",
        "disp_scale_q_low": "0.15",
        "disp_scale_q_high": "0.40",
        "disp_scale_floor": "0.25",
        "disp_scale_power": "2.0",
        "disagree_q_low": "0.4",
        "disagree_q_high": "0.7",
        "disagree_hist_window": "200",
        "disagree_scale": "0.3",
        "consistency_min": "0.05",
        "consistency_scale": "0.6",
        "gate_combine": "avg",
        "gate_avg_weights": "0.45,0.35,0.20",
        "ema_halflife_min": "2",
        "ema_halflife_max": "8",
        "ema_disp_lo": "0.02",
        "ema_disp_hi": "0.10",
        "min_hold": "2",
        "turnover_budget": "0.15",
        "pos_cap": "0.04",
        "gross_target": "1.0",
        "optimize": "true",
        "opt_lambda": "1.0",
        "opt_kappa": "0.0",
        "opt_steps": "20",
        "opt_risk_window": "240",
        "opt_dollar_neutral": "true",
        "topn_cap": "0.15",
        "topn_cap_low": "0.12",
        "topn_n": "10",
        "topn_dyn_q_hi": "0.85",
        "topn_dyn_q_lo": "0.70",
        "shock_metric": "p90",
        "shock_hist_window": "200",
        "min_ic_count": "20",
    }


def _policy_args_dyn_cap_2(
    policy_config: Path, preds_path: Path, out_csv: Path, out_metrics: Path, horizon: int
) -> List[str]:
    p = _policy_params_dyn_cap_2(horizon)
    return [
        "scripts/mu_value_weighted_backtest.py",
        "--config",
        str(policy_config),
        "--preds",
        str(preds_path),
        "--h",
        p["h"],
        "--use_cs",
        "--ret_cs",
        "--disp_metric",
        p["disp_metric"],
        "--disp_hist_window",
        p["disp_hist_window"],
        "--disp_scale_q_low",
        p["disp_scale_q_low"],
        "--disp_scale_q_high",
        p["disp_scale_q_high"],
        "--disp_scale_floor",
        p["disp_scale_floor"],
        "--disp_scale_power",
        p["disp_scale_power"],
        "--disagree_q_low",
        p["disagree_q_low"],
        "--disagree_q_high",
        p["disagree_q_high"],
        "--disagree_hist_window",
        p["disagree_hist_window"],
        "--disagree_scale",
        p["disagree_scale"],
        "--consistency_min",
        p["consistency_min"],
        "--consistency_scale",
        p["consistency_scale"],
        "--gate_combine",
        p["gate_combine"],
        "--gate_avg_weights",
        p["gate_avg_weights"],
        "--ema_halflife_min",
        p["ema_halflife_min"],
        "--ema_halflife_max",
        p["ema_halflife_max"],
        "--ema_disp_lo",
        p["ema_disp_lo"],
        "--ema_disp_hi",
        p["ema_disp_hi"],
        "--min_hold",
        p["min_hold"],
        "--turnover_budget",
        p["turnover_budget"],
        "--pos_cap",
        p["pos_cap"],
        "--gross_target",
        p["gross_target"],
        "--optimize",
        "--opt_lambda",
        p["opt_lambda"],
        "--opt_kappa",
        p["opt_kappa"],
        "--opt_steps",
        p["opt_steps"],
        "--opt_risk_window",
        p["opt_risk_window"],
        "--opt_dollar_neutral",
        "--topn_cap",
        p["topn_cap"],
        "--topn_cap_low",
        p["topn_cap_low"],
        "--topn_n",
        p["topn_n"],
        "--topn_dyn_q_hi",
        p["topn_dyn_q_hi"],
        "--topn_dyn_q_lo",
        p["topn_dyn_q_lo"],
        "--shock_metric",
        p["shock_metric"],
        "--shock_hist_window",
        p["shock_hist_window"],
        "--min_ic_count",
        p["min_ic_count"],
        "--out_csv",
        str(out_csv),
        "--out_metrics",
        str(out_metrics),
    ]


def _overall_active_coverage(cfg: Dict, horizon: int) -> Dict[str, float]:
    data = cfg["data"]
    L = int(data["L"])
    H = int(data["H"])
    step = int(data.get("step", H))
    future_mode = cfg["data"].get("future_obs_mode", "count")
    series = _load_series(cfg)
    T = min(len(s.y) for s in series)
    origin_min = L - 1
    origin_max = T - H - 1
    origins = np.arange(origin_min, origin_max + 1, step, dtype=np.int64)
    counts = np.zeros(origins.shape[0], dtype=np.int64)
    total = np.zeros(origins.shape[0], dtype=np.int64)
    for s in series:
        y = s.y
        if y.ndim == 2:
            y = y[:, 0]
        if future_mode == "exact":
            idx = origins + horizon
            in_range = idx < y.shape[0]
            valid = in_range & np.isfinite(y[idx])
        else:
            # For nearest/count mode, treat any finite point in the window as available.
            valid = np.zeros(origins.shape[0], dtype=bool)
            in_range = np.zeros(origins.shape[0], dtype=bool)
            for k in range(1, horizon + 1):
                idx = origins + k
                in_k = idx < y.shape[0]
                in_range |= in_k
                v = in_k & np.isfinite(y[idx])
                valid |= v
        counts += valid.astype(np.int64)
        total += in_range.astype(np.int64)
    if counts.size == 0:
        return {
            "active_mean": float("nan"),
            "active_p10": float("nan"),
            "active_median": float("nan"),
            "active_ge20_ratio": float("nan"),
            "exact_missing_ratio": float("nan"),
            "future_obs_mode": future_mode,
        }
    missing_ratio = float(1.0 - np.sum(counts) / np.maximum(np.sum(total), 1.0))
    min_time_ic = int(cfg["data"].get("min_time_ic_count", 0))
    time_ge_ratio = float(np.mean(counts >= min_time_ic)) if min_time_ic > 0 else float("nan")
    return {
        "active_mean": float(np.mean(counts)),
        "active_p10": float(np.percentile(counts, 10)),
        "active_median": float(np.median(counts)),
        "active_ge20_ratio": float(np.mean(counts >= 20)),
        "time_ge_min_ic_ratio": time_ge_ratio,
        "exact_missing_ratio": missing_ratio,
        "future_obs_mode": future_mode,
    }


def _infer_obs_interval(series_list: List) -> Dict[str, float]:
    for s in series_list:
        if s.timestamps is None or len(s.timestamps) < 2:
            continue
        dt = np.diff(s.timestamps).astype("timedelta64[h]").astype(np.float32)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            return {
                "obs_interval_hours_median": float(np.median(dt)),
                "obs_interval_hours_p10": float(np.percentile(dt, 10)),
                "obs_interval_hours_p90": float(np.percentile(dt, 90)),
            }
    return {}


def _infer_freq_hours(freq: Optional[str]) -> Optional[float]:
    if not freq:
        return None
    f = freq.lower()
    if "hour" in f:
        return 1.0
    if "day" in f:
        return 24.0
    if "min" in f:
        return 1.0 / 60.0
    return None


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
    summary = {
        "n": int(len(pnl)),
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "sharpe_gross": gross_sharpe,
        "sharpe_vol": vol_sharpe,
        "top5_sum": top5,
        "top5_over_total": top5 / total if total != 0 else float("nan"),
    }
    # Dominance proxy: top5 pnl share (same as top5_over_total).
    summary["dominance"] = summary["top5_over_total"]
    if "turnover" in df.columns:
        summary["turnover_mean"] = float(np.nanmean(df["turnover"].to_numpy()))
    return summary


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


def _coverage_stats(preds_path: Path, horizon: int, cfg: Optional[Dict] = None) -> Dict[str, float]:
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

        # Nearest gap stats (if indices are available and cfg provided)
        if cfg is not None and "series_idx" in d and "origin_t" in d:
            series_idx = d["series_idx"].astype(np.int64)
            origin_t_full = d["origin_t"].astype(np.int64)
            series = _load_series(cfg)
            base_hours = _infer_freq_hours(cfg["data"].get("data_freq")) or 1.0
            gaps = []
            for s_idx, t in zip(series_idx, origin_t_full):
                if s_idx >= len(series):
                    continue
                y = series[s_idx].y
                if y.ndim == 2:
                    y = y[:, 0]
                if t + horizon >= y.shape[0]:
                    continue
                k = None
                for step_k in range(1, horizon + 1):
                    if t + step_k < y.shape[0] and np.isfinite(y[t + step_k]):
                        k = step_k
                        break
                if k is None:
                    continue
                ts = series[s_idx].timestamps
                if ts is not None and t + k < len(ts):
                    try:
                        delta = (ts[t + k] - ts[t]).astype("timedelta64[h]")
                        gaps.append(float(delta))
                        continue
                    except Exception:
                        pass
                gaps.append(float(k) * base_hours)
            if gaps:
                stats["nearest_gap_p50"] = float(np.percentile(gaps, 50))
                stats["nearest_gap_p90"] = float(np.percentile(gaps, 90))
                stats["nearest_gap_mean"] = float(np.mean(gaps))
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

    policy_params = _policy_params_dyn_cap_2(horizon)
    print("policy_params:", json.dumps(policy_params, sort_keys=True))
    overall_cov = _overall_active_coverage(cfg, horizon)
    print("overall_active_coverage:", json.dumps(overall_cov, sort_keys=True))
    raw_series = load_panel_npz(str(cfg["data"]["path"]))
    series_list_for_stats = _load_series(cfg)
    universe_filter = {
        "n_series_raw": int(len(raw_series)),
        "n_series_filtered": int(len(series_list_for_stats)),
        "min_active_ratio": float(cfg["data"].get("universe_min_active_ratio", 0.0)),
        "min_active_points": int(cfg["data"].get("universe_min_active_points", 0)),
        "min_future_ratio": float(cfg["data"].get("universe_min_future_ratio", 0.0)),
        "future_horizon": int(cfg["data"].get("H", cfg["data"].get("future_horizon", 0))),
    }
    obs_interval = _infer_obs_interval(series_list_for_stats)
    base_hours = _infer_freq_hours(cfg["data"].get("freq"))
    observed_only = bool(cfg["data"].get("observed_only", False))
    time_units = {
        "data_freq": cfg["data"].get("freq"),
        "base_bar_hours": base_hours,
        "H_bars": horizon,
        "step_bars": step,
        "effective_horizon_hours": float(horizon * base_hours) if base_hours else None,
        "origin_stride_hours": float(step * base_hours) if base_hours else None,
        "observed_only": observed_only,
        "time_axis_mode": "compressed" if observed_only else "original",
        "future_obs_mode": cfg["data"].get("future_obs_mode", "count"),
    }
    if obs_interval:
        time_units.update(obs_interval)
    print("time_units:", json.dumps(time_units, sort_keys=True))
    universe_filter["mode"] = "fold_train_only"
    print("universe_filter:", json.dumps(universe_filter, sort_keys=True))
    if base_hours and step > 1 and horizon > 1:
        print("warning: step>1 and H>1; verify horizon units vs sampling interval")
    if observed_only:
        print("warning: observed_only=true compresses time axis; H counts observations, not hours")

    folds = _build_folds(n_origins, origin_min, step, args.fold_size, args.n_folds, args.val_size)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    py = str(Path.cwd() / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = "python"
    if py == "python" and shutil.which("python") is None and shutil.which("python3") is not None:
        py = "python3"

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
        # Fold-specific universe based on train window only.
        cfg_fold["data"]["universe_active_end"] = fold.train_end_t
        # Inherit policy data settings if present.
        cfg_fold["data"]["universe_min_active_ratio"] = cfg["data"].get("universe_min_active_ratio", 0.0)
        cfg_fold["data"]["universe_min_active_points"] = cfg["data"].get("universe_min_active_points", 0)
        cfg_fold["data"]["universe_min_future_ratio"] = cfg["data"].get("universe_min_future_ratio", 0.0)
        cfg_fold["data"]["future_horizon"] = horizon
        cfg_fold["data"]["future_obs_mode"] = cfg["data"].get("future_obs_mode", "count")

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
        series_list = _load_series(cfg_fold)
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
            "purge_len": int(cfg_fold["data"].get("split_purge", 0)),
            "embargo_len": int(cfg_fold["data"].get("split_embargo", 0)),
            "horizon": horizon,
            "step": step,
            "scaler_train_only": True,
            "decision_time": cfg_fold["data"].get("decision_time", "close"),
            "feature_lag": int(cfg_fold["data"].get("feature_lag", 0)),
            "future_obs_mode": cfg_fold["data"].get("future_obs_mode", "count"),
            "universe_filter_mode": "fold_train_only",
            "universe_active_end": int(cfg_fold["data"].get("universe_active_end", 0)),
            "universe_min_future_ratio": float(cfg_fold["data"].get("universe_min_future_ratio", 0.0)),
            "future_horizon": int(cfg_fold["data"].get("future_horizon", horizon)),
        }
        with (fold_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with (fold_dir / "protocol.json").open("w", encoding="utf-8") as f:
            json.dump(protocol, f, indent=2)

        preds_paths: List[Path] = []
        for seed in seeds:
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
                        "--save_indices",
                        "true",
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
            policy_fold_path = fold_dir / "configs" / "policy.yaml"
            policy_cfg = _load_cfg(policy_cfg_path)
            policy_cfg.setdefault("data", {})
            policy_cfg["data"]["universe_active_end"] = fold.train_end_t
            policy_cfg["data"]["universe_min_active_ratio"] = cfg_fold["data"].get("universe_min_active_ratio", 0.0)
            policy_cfg["data"]["universe_min_active_points"] = cfg_fold["data"].get("universe_min_active_points", 0)
            policy_cfg["data"]["universe_min_future_ratio"] = cfg_fold["data"].get("universe_min_future_ratio", 0.0)
            policy_cfg["data"]["H"] = horizon
            policy_cfg["data"]["step"] = horizon
            policy_cfg["data"]["future_horizon"] = cfg_fold["data"].get("future_horizon", horizon)
            policy_cfg["data"]["future_obs_mode"] = cfg_fold["data"].get("future_obs_mode", "count")
            _save_cfg(policy_cfg, policy_fold_path)
            cmd = [py, *_policy_args_dyn_cap_2(policy_fold_path, ens_path, out_csv, out_metrics, horizon)]
            subprocess.run(cmd, check=True)

        fold_summary = _summarize_metrics(out_metrics)
        ic_mean, ic_p10, ic_ge20, counts = _ic_count_stats(ens_path, horizon)
        if counts.size:
            ic_counts_all.append(counts)
        fold_summary["ic_count_mean"] = ic_mean if ic_mean is not None else float("nan")
        fold_summary["ic_count_p10"] = ic_p10 if ic_p10 is not None else float("nan")
        fold_summary["ic_count_ge20_ratio"] = ic_ge20 if ic_ge20 is not None else float("nan")
        cov = _coverage_stats(ens_path, horizon, cfg_fold)
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
        "policy_params": policy_params,
        "overall_active_coverage": overall_cov,
        "time_units": time_units,
        "observed_only": observed_only,
        "time_axis_mode": "compressed" if observed_only else "original",
        "universe_filter": universe_filter,
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
