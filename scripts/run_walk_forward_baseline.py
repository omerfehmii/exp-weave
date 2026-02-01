#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

# Ensure repo root on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.harness import TimeSplit, generate_panel_origins, select_indices_by_time  # noqa: E402
from data.loader import SeriesData  # noqa: E402
from eval import apply_scaling  # noqa: E402
from train import filter_indices_by_time_ic, filter_indices_with_observed  # noqa: E402
from scripts import run_walk_forward as wf  # noqa: E402


def _load_cfg(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_cfg(cfg: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _infer_future_mode(cfg: Dict) -> str:
    future_mode = cfg["data"].get("future_obs_mode", "count")
    if future_mode == "nearest":
        future_mode = "count"
    return future_mode


def _extract_sample(
    series: SeriesData,
    t: int,
    L: int,
    H: int,
    target_mode: str,
    target_log_eps: float,
    feature_mode: str,
    last_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y = series.y
    if y.ndim == 1:
        y = y[:, None]
    x_past = series.x_past_feats
    x_future = series.x_future_feats
    mask = series.mask
    delta_t = series.delta_t
    if x_past is None:
        x_past = np.zeros((y.shape[0], 0), dtype=np.float32)
    if x_future is None:
        x_future = np.zeros((y.shape[0], 0), dtype=np.float32)
    if mask is None:
        mask = np.ones_like(y, dtype=np.float32)
    if mask.ndim == 1:
        mask = mask[:, None]
    if delta_t is None:
        delta_t = np.zeros_like(y, dtype=np.float32)
    if delta_t.ndim == 1:
        delta_t = delta_t[:, None]

    past_slice = slice(t - L + 1, t + 1)
    future_slice = slice(t + 1, t + H + 1)

    y_past = y[past_slice]
    y_future = y[future_slice]
    y_last = y_past[-1]
    if np.isnan(y_last).any():
        y_last = y_last.copy()
        for d in range(y_past.shape[1]):
            col = y_past[:, d]
            idx = np.where(np.isfinite(col))[0]
            if idx.size:
                y_last[d] = col[idx[-1]]
    if np.isnan(y_past).any():
        y_past = np.nan_to_num(y_past, nan=0.0)

    x_past_feats = x_past[past_slice]
    x_future_feats = x_future[future_slice]
    mask_past = mask[past_slice]
    delta_past = delta_t[past_slice]

    if feature_mode == "lastk":
        k = max(1, min(int(last_k), y_past.shape[0]))
        y_past = y_past[-k:]
        x_past_feats = x_past_feats[-k:]
        mask_past = mask_past[-k:]
        delta_past = delta_past[-k:]

    feat = np.concatenate(
        [
            y_past.reshape(-1),
            x_past_feats.reshape(-1),
            mask_past.reshape(-1),
            delta_past.reshape(-1),
            x_future_feats.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    if target_mode == "return":
        target_arr = y_future - y_last[None, :]
    elif target_mode == "log_return":
        y_raw = y if series.y_raw is None else series.y_raw
        if y_raw.ndim == 1:
            y_raw = y_raw[:, None]
        y_past_raw = y_raw[past_slice]
        y_future_raw = y_raw[future_slice]
        y_last_raw = y_past_raw[-1]
        eps = target_log_eps
        target_arr = np.log(np.maximum(y_future_raw, eps) / np.maximum(y_last_raw[None, :], eps))
    else:
        target_arr = y_future

    return feat, target_arr.astype(np.float32)


def _build_matrix(
    series_list: List[SeriesData],
    indices: List[Tuple[int, int]],
    L: int,
    H: int,
    target_mode: str,
    target_log_eps: float,
    feature_mode: str,
    last_k: int,
    feat_dim: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not indices:
        if feat_dim is None:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0, H), dtype=np.float32)
        return np.zeros((0, int(feat_dim)), dtype=np.float32), np.zeros((0, H), dtype=np.float32)
    series_list[0].ensure_features()
    feat0, target0 = _extract_sample(
        series_list[indices[0][0]],
        indices[0][1],
        L,
        H,
        target_mode,
        target_log_eps,
        feature_mode,
        last_k,
    )
    d = int(feat0.size)
    X = np.zeros((len(indices), d), dtype=np.float32)
    y = np.full((len(indices), H), np.nan, dtype=np.float32)
    X[0] = feat0
    y[0] = target0
    for i in range(1, len(indices)):
        s_idx, t = indices[i]
        feat, target_arr = _extract_sample(
            series_list[s_idx],
            t,
            L,
            H,
            target_mode,
            target_log_eps,
            feature_mode,
            last_k,
        )
        X[i] = feat
        y[i] = target_arr
    return X, y


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    # X assumed standardized (zero mean, unit var). Fit intercept separately.
    y_mean = float(np.mean(y))
    y_c = y - y_mean
    X64 = X.astype(np.float64, copy=False)
    y64 = y_c.astype(np.float64, copy=False)
    xtx = X64.T @ X64
    xtx.flat[:: xtx.shape[0] + 1] += float(alpha)
    xty = X64.T @ y64
    w = np.linalg.solve(xtx, xty)
    return w.astype(np.float32), float(y_mean)


def _ridge_predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def _elastic_net_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    y_mean = float(np.mean(y))
    y_c = y - y_mean
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    y_pred = X @ w
    x_norm2 = np.sum(X * X, axis=0)
    l1 = float(alpha * l1_ratio) * n
    l2 = float(alpha * (1.0 - l1_ratio)) * n
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(d):
            y_pred -= X[:, j] * w[j]
            rho = float(np.dot(X[:, j], y_c - y_pred))
            if rho < -l1:
                w[j] = (rho + l1) / (x_norm2[j] + l2)
            elif rho > l1:
                w[j] = (rho - l1) / (x_norm2[j] + l2)
            else:
                w[j] = 0.0
            y_pred += X[:, j] * w[j]
        if np.max(np.abs(w - w_old)) < tol:
            break
    return w.astype(np.float32), float(y_mean)


def _select_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: List[float],
    model: str,
    l1_ratio: Optional[float] = None,
) -> Tuple[float, np.ndarray, float]:
    if y_val.size == 0:
        if model == "ridge":
            w, b = _ridge_fit(X_train, y_train, alpha_grid[0])
            return float(alpha_grid[0]), w, b
        if l1_ratio is None:
            raise ValueError("l1_ratio required for elasticnet")
        w, b = _elastic_net_fit(X_train, y_train, alpha_grid[0], l1_ratio)
        return float(alpha_grid[0]), w, b
    best_alpha = float(alpha_grid[0])
    best_mse = float("inf")
    best_w: Optional[np.ndarray] = None
    best_b: float = 0.0
    for alpha in alpha_grid:
        if model == "ridge":
            w, b = _ridge_fit(X_train, y_train, alpha)
        else:
            if l1_ratio is None:
                raise ValueError("l1_ratio required for elasticnet")
            w, b = _elastic_net_fit(X_train, y_train, alpha, l1_ratio)
        pred = _ridge_predict(X_val, w, b)
        mse = float(np.mean((pred - y_val) ** 2)) if y_val.size else float("inf")
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
            best_w = w
            best_b = b
    if best_w is None:
        raise RuntimeError("Failed to select alpha.")
    return best_alpha, best_w, best_b


def _fit_gbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[object, str]:
    try:
        import lightgbm as lgb  # type: ignore

        model = lgb.LGBMRegressor(
            num_leaves=31,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",
            early_stopping_rounds=50,
            verbose=False,
        )
        return model, "lightgbm"
    except Exception:
        pass
    try:
        import xgboost as xgb  # type: ignore

        model = xgb.XGBRegressor(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=50,
        )
        return model, "xgboost"
    except Exception as exc:
        raise RuntimeError("Neither lightgbm nor xgboost is available.") from exc


def _score_stats(preds_path: Path, horizon: int) -> Tuple[float, float]:
    with np.load(preds_path) as d:
        q50 = d["q50"]
        mask = d["mask"] if "mask" in d else None
        h_idx = horizon - 1
        scores = q50[:, h_idx]
        if mask is not None and mask.ndim == 2 and mask.shape[1] > h_idx:
            scores = scores[mask[:, h_idx] > 0]
        if scores.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--policy_config", required=True)
    parser.add_argument("--out_dir", default="runs/wf_baseline")
    parser.add_argument("--fold_size", type=int, default=64)
    parser.add_argument("--n_folds", type=int, default=2)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--seeds", default="7,13")
    parser.add_argument("--model", default="ridge", choices=["ridge", "elasticnet", "gbdt"])
    parser.add_argument("--feature_mode", default="full", choices=["full", "lastk"])
    parser.add_argument("--last_k", type=int, default=24)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    base_cfg_path = Path(args.base_config)
    policy_cfg_path = Path(args.policy_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_cfg(base_cfg_path)
    data = cfg["data"]
    horizon = int(data["H"])
    step = int(data.get("step", data["H"]))
    T, origin_min, origin_max, n_origins = wf._compute_origin_info(cfg)
    try:
        origin_max_valid = wf._compute_valid_origin_max(cfg, horizon)
        if origin_max_valid < origin_max:
            origin_max = origin_max_valid
            n_origins = (origin_max - origin_min) // step + 1
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not compute valid origin max: {exc}")

    folds = wf._build_folds(n_origins, origin_min, step, args.fold_size, args.n_folds, args.val_size)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    policy_cfg = _load_cfg(policy_cfg_path)
    policy_cfg.setdefault("data", {})
    policy_cfg["data"]["H"] = horizon
    policy_cfg["data"]["step"] = horizon
    policy_cfg["data"]["future_horizon"] = horizon
    policy_cfg["data"]["future_obs_mode"] = cfg["data"].get("future_obs_mode", "count")
    policy_fold_template = out_dir / "policy.yaml"
    _save_cfg(policy_cfg, policy_fold_template)

    summary_rows: List[Dict] = []
    combined_metrics_paths: List[Path] = []
    ic_counts_all: List[np.ndarray] = []
    ignore_vals: List[float] = []
    coverage_all: List[Dict[str, float]] = []

    alpha_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    enet_l1_grid = [0.1, 0.5, 0.9]

    for fold in folds:
        fold_dir = out_dir / f"fold_{fold.fold}"
        (fold_dir / "preds").mkdir(parents=True, exist_ok=True)
        (fold_dir / "configs").mkdir(parents=True, exist_ok=True)

        cfg_fold = _load_cfg(base_cfg_path)
        cfg_fold.setdefault("data", {})
        cfg_fold["data"]["split_train_end"] = fold.train_end_t
        cfg_fold["data"]["split_val_end"] = fold.val_end_t + horizon
        cfg_fold["data"]["split_test_end"] = fold.test_end_t + horizon
        cfg_fold["data"]["min_future_obs"] = horizon
        cfg_fold["data"]["universe_active_end"] = fold.train_end_t
        cfg_fold["data"]["universe_min_active_ratio"] = cfg["data"].get("universe_min_active_ratio", 0.0)
        cfg_fold["data"]["universe_min_active_points"] = cfg["data"].get("universe_min_active_points", 0)
        cfg_fold["data"]["universe_min_future_ratio"] = cfg["data"].get("universe_min_future_ratio", 0.0)
        cfg_fold["data"]["future_horizon"] = horizon
        cfg_fold["data"]["future_obs_mode"] = cfg["data"].get("future_obs_mode", "count")
        _save_cfg(cfg_fold, fold_dir / "configs" / "baseline.yaml")

        # build series list and apply scaling
        series_list = wf._load_series(cfg_fold)
        for s in series_list:
            s.ensure_features()
        split = TimeSplit(
            train_end=int(cfg_fold["data"]["split_train_end"]),
            val_end=int(cfg_fold["data"]["split_val_end"]),
            test_end=int(cfg_fold["data"]["split_test_end"]),
        )
        apply_scaling(
            series_list,
            split.train_end,
            scale_x=cfg_fold["data"].get("scale_x", True),
            scale_y=cfg_fold["data"].get("scale_y", True),
        )

        lengths = [len(s.y) for s in series_list]
        indices = generate_panel_origins(lengths, cfg_fold["data"]["L"], horizon, step)
        purge = int(cfg_fold["data"].get("split_purge", 0))
        embargo = int(cfg_fold["data"].get("split_embargo", 0))
        train_idx = select_indices_by_time(indices, split, "train", horizon=horizon, purge=purge, embargo=embargo)
        val_idx = select_indices_by_time(indices, split, "val", horizon=horizon, purge=purge, embargo=embargo)
        test_idx = select_indices_by_time(indices, split, "test", horizon=horizon, purge=purge, embargo=embargo)

        min_past_obs = int(cfg_fold["data"].get("min_past_obs", 1))
        min_future_obs = int(cfg_fold["data"].get("min_future_obs", 1))
        min_time_ic = int(cfg_fold["data"].get("min_time_ic_count", 0))
        future_mode = _infer_future_mode(cfg_fold)

        train_idx = filter_indices_with_observed(series_list, train_idx, cfg_fold["data"]["L"], horizon, min_past_obs, min_future_obs, future_mode)
        val_idx = filter_indices_with_observed(series_list, val_idx, cfg_fold["data"]["L"], horizon, min_past_obs, min_future_obs, future_mode)
        test_idx = filter_indices_with_observed(series_list, test_idx, cfg_fold["data"]["L"], horizon, min_past_obs, min_future_obs, future_mode)
        train_idx = filter_indices_by_time_ic(series_list, train_idx, horizon, min_time_ic, min_future_obs, future_mode)
        val_idx = filter_indices_by_time_ic(series_list, val_idx, horizon, min_time_ic, min_future_obs, future_mode)
        test_idx = filter_indices_by_time_ic(series_list, test_idx, horizon, min_time_ic, min_future_obs, future_mode)
        if not train_idx:
            raise RuntimeError(f"fold {fold.fold}: no training samples after filtering.")

        # build matrices
        X_train, y_train = _build_matrix(
            series_list,
            train_idx,
            cfg_fold["data"]["L"],
            horizon,
            cfg_fold["data"].get("target_mode", "level"),
            float(cfg_fold["data"].get("target_log_eps", 1e-6)),
            args.feature_mode,
            args.last_k,
        )
        X_val, y_val = _build_matrix(
            series_list,
            val_idx,
            cfg_fold["data"]["L"],
            horizon,
            cfg_fold["data"].get("target_mode", "level"),
            float(cfg_fold["data"].get("target_log_eps", 1e-6)),
            args.feature_mode,
            args.last_k,
            feat_dim=X_train.shape[1],
        )
        X_test, y_test = _build_matrix(
            series_list,
            test_idx,
            cfg_fold["data"]["L"],
            horizon,
            cfg_fold["data"].get("target_mode", "level"),
            float(cfg_fold["data"].get("target_log_eps", 1e-6)),
            args.feature_mode,
            args.last_k,
            feat_dim=X_train.shape[1],
        )

        mean, std = _standardize_fit(X_train)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        h_idx = horizon - 1
        train_mask = np.isfinite(y_train[:, h_idx])
        val_mask = np.isfinite(y_val[:, h_idx])
        test_mask = np.isfinite(y_test[:, h_idx])

        if not np.any(train_mask):
            raise RuntimeError(f"fold {fold.fold}: no finite y at horizon {horizon} in train.")

        X_train_h = X_train[train_mask]
        y_train_h = y_train[train_mask, h_idx]
        X_val_h = X_val[val_mask] if np.any(val_mask) else X_val[:0]
        y_val_h = y_val[val_mask, h_idx] if np.any(val_mask) else y_train_h[:0]

        preds_paths: List[Path] = []
        for seed in seeds:
            preds_path = fold_dir / "preds" / f"preds_s{seed}.npz"
            if args.skip_existing and preds_path.exists():
                preds_paths.append(preds_path)
                continue

            if args.model == "ridge":
                _, w, b = _select_alpha(X_train_h, y_train_h, X_val_h, y_val_h, alpha_grid, "ridge")
                pred = _ridge_predict(X_test, w, b)
            elif args.model == "elasticnet":
                best_alpha = alpha_grid[0]
                best_w = None
                best_b = 0.0
                best_mse = float("inf")
                for l1_ratio in enet_l1_grid:
                    alpha, w, b = _select_alpha(X_train_h, y_train_h, X_val_h, y_val_h, alpha_grid, "elasticnet", l1_ratio)
                    pred = _ridge_predict(X_val_h, w, b)
                    mse = float(np.mean((pred - y_val_h) ** 2)) if y_val_h.size else float("inf")
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = alpha
                        best_w = w
                        best_b = b
                if best_w is None:
                    if y_val_h.size == 0:
                        best_w, best_b = _elastic_net_fit(X_train_h, y_train_h, alpha_grid[0], enet_l1_grid[0])
                    else:
                        raise RuntimeError("ElasticNet failed to fit.")
                pred = _ridge_predict(X_test, best_w, best_b)
            else:
                model, backend = _fit_gbdt(X_train_h, y_train_h, X_val_h, y_val_h, seed)
                pred = model.predict(X_test)
                print(f"gbdt_backend={backend} seed={seed}")

            q50 = np.zeros((len(test_idx), horizon), dtype=np.float32)
            q50[:, h_idx] = pred.astype(np.float32)
            q10 = q50.copy()
            q90 = q50.copy()
            y_out = np.nan_to_num(y_test, nan=0.0)
            mask = np.isfinite(y_test).astype(np.float32)
            series_idx = np.array([i[0] for i in test_idx], dtype=np.int64)
            origin_t = np.array([i[1] for i in test_idx], dtype=np.int64)
            np.savez(
                preds_path,
                y=y_out,
                q10=q10,
                q50=q50,
                q90=q90,
                mask=mask,
                series_idx=series_idx,
                origin_t=origin_t,
            )
            preds_paths.append(preds_path)

        ens_path = fold_dir / "ens_preds.npz"
        if not (args.skip_existing and ens_path.exists()):
            subprocess.run([sys.executable, "scripts/ensemble_preds.py", "--out", str(ens_path), "--preds", *map(str, preds_paths)], check=True)

        out_csv = fold_dir / "backtest.csv"
        out_metrics = fold_dir / "metrics.csv"
        if not (args.skip_existing and out_metrics.exists()):
            policy_fold_path = fold_dir / "configs" / "policy.yaml"
            policy_cfg_fold = _load_cfg(policy_cfg_path)
            policy_cfg_fold.setdefault("data", {})
            policy_cfg_fold["data"]["H"] = horizon
            policy_cfg_fold["data"]["step"] = horizon
            policy_cfg_fold["data"]["universe_active_end"] = fold.train_end_t
            policy_cfg_fold["data"]["universe_min_active_ratio"] = cfg_fold["data"].get("universe_min_active_ratio", 0.0)
            policy_cfg_fold["data"]["universe_min_active_points"] = cfg_fold["data"].get("universe_min_active_points", 0)
            policy_cfg_fold["data"]["universe_min_future_ratio"] = cfg_fold["data"].get("universe_min_future_ratio", 0.0)
            policy_cfg_fold["data"]["future_horizon"] = horizon
            policy_cfg_fold["data"]["future_obs_mode"] = cfg_fold["data"].get("future_obs_mode", "count")
            _save_cfg(policy_cfg_fold, policy_fold_path)
            cmd = [sys.executable, *wf._policy_args_dyn_cap_2(policy_fold_path, ens_path, out_csv, out_metrics, horizon)]
            subprocess.run(cmd, check=True)

        fold_summary = wf._summarize_metrics(out_metrics)
        ic_mean, ic_p10, ic_ge20, counts = wf._ic_count_stats(ens_path, horizon)
        if counts.size:
            ic_counts_all.append(counts)
        fold_summary["ic_count_mean"] = ic_mean if ic_mean is not None else float("nan")
        fold_summary["ic_count_p10"] = ic_p10 if ic_p10 is not None else float("nan")
        fold_summary["ic_count_ge20_ratio"] = ic_ge20 if ic_ge20 is not None else float("nan")
        cov = wf._coverage_stats(ens_path, horizon, None)
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
        score_mean, score_std = _score_stats(ens_path, horizon)
        fold_summary["score_mean"] = score_mean
        fold_summary["score_std"] = score_std
        if fold_summary.get("ic_count_ge20_ratio", float("nan")) != fold_summary.get("ic_count_ge20_ratio", 0.0):
            fold_summary["coverage_warn"] = True
        else:
            fold_summary["coverage_warn"] = fold_summary["ic_count_ge20_ratio"] < 0.5
        ignore_ratio = wf._ignore_ratio(ens_path)
        fold_summary["ignore_ratio"] = ignore_ratio if ignore_ratio is not None else float("nan")
        fold_summary["cost_included"] = False
        fold_summary["fold"] = fold.fold
        summary_rows.append(fold_summary)
        combined_metrics_paths.append(out_metrics)

    # write summary
    import pandas as pd

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    all_df = pd.concat([pd.read_csv(p) for p in combined_metrics_paths], ignore_index=True)
    all_df.to_csv(out_dir / "metrics_all.csv", index=False)
    comb = wf._summarize_metrics(out_dir / "metrics_all.csv")
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
    with (out_dir / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(comb, f, indent=2)

    overall_cov = wf._overall_active_coverage(cfg, horizon)
    wf_summary = {
        "n_folds": len(summary_rows),
        "coverage_warn_folds": int(np.sum([1 for row in summary_rows if row.get("coverage_warn")])),
        "policy_params": wf._policy_params_dyn_cap_2(horizon),
        "overall_active_coverage": overall_cov,
        "time_units": {
            "data_freq": cfg["data"].get("freq"),
            "base_bar_hours": wf._infer_freq_hours(cfg["data"].get("freq")),
            "H_bars": horizon,
            "step_bars": step,
            "effective_horizon_hours": float(horizon * wf._infer_freq_hours(cfg["data"].get("freq"))) if wf._infer_freq_hours(cfg["data"].get("freq")) else None,
            "origin_stride_hours": float(step * wf._infer_freq_hours(cfg["data"].get("freq"))) if wf._infer_freq_hours(cfg["data"].get("freq")) else None,
            "observed_only": bool(cfg["data"].get("observed_only", False)),
            "time_axis_mode": "compressed" if cfg["data"].get("observed_only", False) else "original",
            "future_obs_mode": cfg["data"].get("future_obs_mode", "count"),
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
        "model": {
            "type": args.model,
            "feature_mode": args.feature_mode,
            "last_k": args.last_k,
            "alpha_grid": alpha_grid,
            "elasticnet_l1_grid": enet_l1_grid if args.model == "elasticnet" else None,
        },
    }
    with (out_dir / "wf_summary.json").open("w", encoding="utf-8") as f:
        json.dump(wf_summary, f, indent=2)


if __name__ == "__main__":
    main()
