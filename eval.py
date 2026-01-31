from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from backtest.harness import TimeSplit, generate_panel_origins, make_time_splits, select_indices_by_time
from backtest.metrics import horizon_collapse, quantile_crossing_rate
from backtest.diagnostics import (
    attention_similarity,
    band_summary,
    mean_offdiag_corr,
    mean_offdiag_cosine,
    width_stats_per_horizon,
)
from calibration.cqr import CQRCalibrator
from data.loader import SeriesData, WindowedDataset, load_panel_npz, compress_series_observed, filter_series_by_active_ratio, filter_series_by_future_ratio
from data.features import append_direction_features
from data.missingness import MissingnessConfig, MissingnessDataset
from data.preprocess import FoldFitPreprocessor
from model import ModelConfig, MultiScaleForecastModel, PatchScale
from utils import load_config, set_seed
from width_scaling import apply_width_scaling, fit_s_global, fit_s_per_horizon
from regime import (
    REGIME_NAMES,
    REGIME_TREND,
    REGIME_RANGE_MR,
    REGIME_CHOP_VOL,
    REGIME_JUMPY,
    compute_regime_features_np,
    compute_regime_features_torch,
    fit_regime_thresholds,
    label_regimes,
)
from interval_clamp import apply_min_half_clamp, fit_min_half_abs, fit_min_half_rel, half_stats


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    denom = np.sum(mask)
    return float(np.sum(values * mask) / max(denom, 1.0))


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return _masked_mean(np.abs(y_true - y_pred), mask)


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(_masked_mean((y_true - y_pred) ** 2, mask)))


def masked_pinball(y_true: np.ndarray, q_pred: np.ndarray, quantiles: List[float], mask: np.ndarray) -> float:
    y = y_true.reshape(-1)
    q = np.nan_to_num(q_pred, nan=0.0).reshape(-1, q_pred.shape[-1])
    mask_flat = mask.reshape(-1)
    losses = []
    for i, tau in enumerate(quantiles):
        diff = y - q[:, i]
        loss = np.maximum(tau * diff, (tau - 1) * diff)
        losses.append(loss * mask_flat)
    denom = max(np.sum(mask_flat) * len(quantiles), 1.0)
    return float(np.sum(np.stack(losses, axis=1)) / denom)


def masked_coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, mask: np.ndarray) -> float:
    within = (y_true >= q_low) & (y_true <= q_high) & (mask > 0)
    denom = np.sum(mask)
    return float(np.sum(within) / max(denom, 1.0))


def masked_interval_width(q_low: np.ndarray, q_high: np.ndarray, mask: np.ndarray) -> float:
    return _masked_mean(q_high - q_low, mask)


def masked_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    H = y_true.shape[1]
    mae_h = np.zeros(H, dtype=np.float32)
    rmse_h = np.zeros(H, dtype=np.float32)
    for h in range(H):
        m = mask[:, h]
        mae_h[h] = masked_mae(y_true[:, h], y_pred[:, h], m)
        rmse_h[h] = masked_rmse(y_true[:, h], y_pred[:, h], m)
    return {"mae": mae_h, "rmse": rmse_h}


def coverage_per_horizon(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inside = (y_true >= q_low) & (y_true <= q_high) & (mask > 0)
    denom = np.sum(mask, axis=0)
    return np.sum(inside, axis=0) / np.maximum(denom, 1.0)


def width_per_horizon(q_low: np.ndarray, q_high: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denom = np.sum(mask, axis=0)
    return np.sum((q_high - q_low) * mask, axis=0) / np.maximum(denom, 1.0)


def pinball_per_horizon(y_true: np.ndarray, q_pred: np.ndarray, tau: float, mask: np.ndarray) -> np.ndarray:
    diff = y_true - q_pred
    loss = np.maximum(tau * diff, (tau - 1.0) * diff)
    denom = np.sum(mask, axis=0)
    return np.sum(loss * mask, axis=0) / np.maximum(denom, 1.0)


def build_series_list(
    path: str,
    observed_only: bool = False,
    min_active_ratio: float = 0.0,
    min_active_points: int = 0,
    active_end: int | None = None,
    min_future_ratio: float = 0.0,
    future_horizon: int = 0,
) -> List[SeriesData]:
    series_list = load_panel_npz(path)
    if observed_only:
        series_list = compress_series_observed(series_list)
    if min_active_ratio or min_active_points:
        series_list = filter_series_by_active_ratio(series_list, min_active_ratio, min_active_points, active_end)
    if min_future_ratio and future_horizon > 0:
        series_list = filter_series_by_future_ratio(series_list, min_future_ratio, future_horizon, active_end)
        if not series_list:
            raise ValueError("Universe filter removed all series. Check universe_min_active_ratio/points.")
    for series in series_list:
        series.ensure_features()
    return series_list


def apply_scaling(
    series_list: List[SeriesData],
    split_end: int,
    scale_x: bool = True,
    scale_y: bool = True,
) -> FoldFitPreprocessor:
    y_train = np.concatenate([s.y[:split_end] for s in series_list], axis=0)
    mask_train = np.concatenate(
        [
            s.mask[:split_end]
            if s.mask is not None
            else np.ones_like(s.y[:split_end], dtype=np.float32)
            for s in series_list
        ],
        axis=0,
    )
    x_train = None
    if scale_x and all(s.x_past_feats is not None for s in series_list):
        x_train = np.concatenate([s.x_past_feats[:split_end] for s in series_list], axis=0)
    else:
        scale_x = False
    pre = FoldFitPreprocessor(scale_y=scale_y, scale_x=scale_x)
    pre.fit(y_train, x_train=x_train, mask=mask_train)
    warned = False
    for series in series_list:
        if series.y_raw is None:
            series.y_raw = series.y.copy()
        if scale_y:
            series.y = pre.y_scaler.transform(series.y)
        if scale_x and series.x_past_feats is not None:
            series.x_past_feats = pre.x_scaler.transform(series.x_past_feats)
            if series.x_future_feats is not None:
                if series.x_future_feats.shape[1] == series.x_past_feats.shape[1]:
                    series.x_future_feats = pre.x_scaler.transform(series.x_future_feats)
                elif not warned:
                    print("warning: skipping x_future scaling due to feature dim mismatch.")
                    warned = True
    return pre


def build_model(cfg: Dict) -> MultiScaleForecastModel:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    patch_cfg = cfg["patching"]
    decoder_cfg = cfg.get("decoder", {})
    dual_path_cfg = decoder_cfg.get("dual_path", {})
    head_cfg = cfg.get("head", {})
    scales = [PatchScale(**scale) for scale in patch_cfg["scales"]]
    head_cfg = cfg.get("head", {})
    missing_cfg = cfg.get("missingness", {})
    moe_cfg = cfg.get("moe", {})
    delta_t_mode = missing_cfg.get("delta_t_mode", missing_cfg.get("dt_embedding", "MLP_LOG1P"))
    summary_cfg = patch_cfg.get("summary", {})
    gate_cfg = patch_cfg.get("gate", {})
    scale_drop_cfg = patch_cfg.get("scale_drop", {})
    dir_head_cfg = cfg.get("direction_head", {})
    rank_head_cfg = model_cfg.get("rank_head", {})
    cumret_cfg = cfg.get("cumret24_head", {})
    cumret_enabled = False
    if isinstance(cumret_cfg, dict):
        cumret_enabled = bool(cumret_cfg.get("enabled", False))
    elif isinstance(cumret_cfg, bool):
        cumret_enabled = bool(cumret_cfg)
    cumret_model = model_cfg.get("cumret24_head", None)
    if cumret_model is not None:
        if isinstance(cumret_model, dict):
            cumret_enabled = bool(cumret_model.get("enabled", False))
        else:
            cumret_enabled = bool(cumret_model)
    config = ModelConfig(
        L=data_cfg["L"],
        H=data_cfg["H"],
        target_dim=data_cfg.get("target_dim", 1),
        past_feat_dim=data_cfg.get("past_feat_dim", 0),
        future_feat_dim=data_cfg.get("future_feat_dim", 0),
        quantiles=data_cfg["quantiles"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        mlp_hidden=model_cfg["mlp_hidden"],
        dropout=model_cfg["dropout"],
        fusion=patch_cfg["fusion"],
        gate_entropy_floor=gate_cfg.get("entropy_floor"),
        gate_temperature=gate_cfg.get("temperature", 1.0),
        gate_logit_clip=gate_cfg.get("logit_clip"),
        gate_fixed_weight=gate_cfg.get("fixed_weight"),
        gate_use_regime_features=bool(gate_cfg.get("use_regime_features", False)),
        gate_regime_window=int(gate_cfg.get("regime_window", data_cfg.get("L", 240))),
        gate_regime_eps=float(gate_cfg.get("regime_eps", 1e-6)),
        gate_disable_coarse=bool(gate_cfg.get("disable_coarse", False)),
        gate_disable_fine=bool(gate_cfg.get("disable_fine", False)),
        scale_drop_coarse=float(scale_drop_cfg.get("drop_coarse_p", scale_drop_cfg.get("coarse", 0.0))),
        scale_drop_fine=float(scale_drop_cfg.get("drop_fine_p", scale_drop_cfg.get("fine", 0.0))),
        dual_sum_weight=patch_cfg.get("dual_sum_weight", "equal"),
        decoder_mode=decoder_cfg.get("mode", "CA_ONLY"),
        cats_enabled=decoder_cfg.get("cats_masking", {}).get("enabled", True),
        cats_p_min=decoder_cfg.get("cats_masking", {}).get("p_min", 0.1),
        cats_p_max=decoder_cfg.get("cats_masking", {}).get("p_max", 0.7),
        cats_scaling=decoder_cfg.get("cats_masking", {}).get("scaling", "NONE"),
        dual_path=dual_path_cfg.get("enabled", False),
        dual_path_uncertainty_cats=dual_path_cfg.get("uncertainty_cats", False),
        head_type=head_cfg.get("type", "MONO"),
        head_delta_floor=head_cfg.get("delta_floor", 0.0),
        head_lsq_s_min=head_cfg.get("lsq_s_min", 0.0),
        head_detach=bool(model_cfg.get("head_detach", False)),
        mask_embedding=missing_cfg.get("mask_embedding", True),
        delta_t_mode=delta_t_mode,
        attn_logit_bias=missing_cfg.get("attn_logit_bias", "HARD_NEG_INF"),
        summary_enabled=summary_cfg.get("enabled", False),
        summary_window=summary_cfg.get("window", 24),
        summary_dropout=summary_cfg.get("dropout", 0.0),
        dir_head_enabled=dir_head_cfg.get("enabled", False),
        dir_head_type=dir_head_cfg.get("type", "hierarchical"),
        dir_head_detach=dir_head_cfg.get("detach", False),
        dir_head_dropout=dir_head_cfg.get("dropout", 0.0),
        rank_head_enabled=bool(rank_head_cfg.get("enabled", model_cfg.get("rank_head_enabled", False))),
        rank_head_detach=bool(rank_head_cfg.get("detach", model_cfg.get("rank_head_detach", False))),
        rank_head_dropout=float(rank_head_cfg.get("dropout", model_cfg.get("rank_head_dropout", 0.0))),
        cumret24_head=cumret_enabled,
        moe_enabled=bool(moe_cfg.get("enabled", False)),
        moe_gate_hidden=int(moe_cfg.get("gate_hidden", model_cfg.get("d_model", 256))),
        moe_gate_temperature=float(moe_cfg.get("gate_temperature", 1.0)),
        moe_gate_logit_clip=moe_cfg.get("gate_logit_clip"),
        moe_gate_use_regime_features=bool(moe_cfg.get("use_regime_features", False)),
        moe_gate_regime_window=int(moe_cfg.get("regime_window", data_cfg.get("L", 240))),
        moe_gate_regime_eps=float(moe_cfg.get("regime_eps", 1e-6)),
        moe_expert_drop_trend=float(moe_cfg.get("expert_drop_trend", 0.0)),
        moe_expert_drop_mr=float(moe_cfg.get("expert_drop_mr", 0.0)),
    )
    return MultiScaleForecastModel(config, scales)


def filter_indices_with_observed(
    series_list: List[SeriesData],
    indices: List[tuple],
    L: int,
    H: int,
    min_past_obs: int,
    min_future_obs: int,
    future_mode: str = "count",
) -> List[tuple]:
    if min_past_obs <= 0 and min_future_obs <= 0:
        return indices
    keep = []
    for s_idx, t in indices:
        y = series_list[s_idx].y
        if y.ndim == 1:
            y = y[:, None]
        past = y[t - L + 1 : t + 1]
        if future_mode == "exact":
            if t + H >= y.shape[0]:
                continue
            future_ok = np.isfinite(y[t + H]).all()
        else:
            future = y[t + 1 : t + H + 1]
            future_ok = np.isfinite(future).sum() >= min_future_obs
        if np.isfinite(past).sum() >= min_past_obs and future_ok:
            keep.append((s_idx, t))
    return keep


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def collect_predictions(
    model: MultiScaleForecastModel,
    loader: DataLoader,
    device: torch.device,
    diagnostics: bool = False,
    attn: bool = False,
    return_meta: bool = False,
    regime_window: int = 240,
    regime_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict | None]:
    y_list = []
    q_list = []
    dec_list = []
    attn_list = []
    meta: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for batch, target in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            q_hat, extras = model(batch, return_diagnostics=diagnostics, return_attn=attn)
            y_list.append(target[..., 0].cpu().numpy())
            q_list.append(q_hat.cpu().numpy())
            if diagnostics and "dec_out" in extras:
                dec_list.append(extras["dec_out"].cpu().numpy())
            if attn and "attn" in extras:
                attn_val = extras["attn"]
                if isinstance(attn_val, list):
                    attn_list.append([a.cpu().numpy() for a in attn_val])
                else:
                    attn_list.append(attn_val.cpu().numpy())
            if return_meta:
                feats = compute_regime_features_torch(
                    batch["y_past"],
                    batch.get("mask"),
                    window=regime_window,
                    eps=regime_eps,
                )
                for key, val in feats.items():
                    meta.setdefault(key, []).append(val.detach().cpu().numpy())
                if "gate_weights" in extras:
                    meta.setdefault("gate_weights", []).append(extras["gate_weights"].detach().cpu().numpy())
                if "gate_logits" in extras:
                    meta.setdefault("gate_logits", []).append(extras["gate_logits"].detach().cpu().numpy())
                if "moe_weights" in extras:
                    meta.setdefault("moe_weights", []).append(extras["moe_weights"].detach().cpu().numpy())
                if "moe_logits" in extras:
                    meta.setdefault("moe_logits", []).append(extras["moe_logits"].detach().cpu().numpy())
    y = np.concatenate(y_list, axis=0)
    q = np.concatenate(q_list, axis=0)
    dec = np.concatenate(dec_list, axis=0) if dec_list else None
    if not attn_list:
        attn_arr = None
    elif isinstance(attn_list[0], list):
        attn_arr = [np.concatenate([a[i] for a in attn_list], axis=0) for i in range(len(attn_list[0]))]
    else:
        attn_arr = np.concatenate(attn_list, axis=0)
    meta_out = None
    if return_meta:
        meta_out = {k: np.concatenate(v, axis=0) for k, v in meta.items()}
    return y, q, dec, attn_arr, meta_out


def collect_regime_features(
    loader: DataLoader,
    device: torch.device,
    window: int,
    eps: float,
) -> Dict[str, np.ndarray]:
    feats: Dict[str, List[np.ndarray]] = {}
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = compute_regime_features_torch(batch["y_past"], batch.get("mask"), window=window, eps=eps)
            for key, val in out.items():
                feats.setdefault(key, []).append(val.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in feats.items()}


def _monotonic_violations(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, eps: float = 1e-6) -> Dict[str, float]:
    total = q10.size
    low = np.sum(q10 > q50 + eps) / max(total, 1)
    high = np.sum(q50 > q90 + eps) / max(total, 1)
    return {"q10_gt_q50": float(low), "q50_gt_q90": float(high)}


def collect_baselines(
    loader: DataLoader,
    device: torch.device,
    seasonal_lag: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    last_list = []
    seasonal_list = []
    last_mask_list = []
    for batch, _ in loader:
        y_past = batch["y_past"].to(device)
        mask_past = batch["mask"].to(device)
        if y_past.ndim == 2:
            y_past = y_past.unsqueeze(-1)
            mask_past = mask_past.unsqueeze(-1)
        last = y_past[:, -1, 0]
        last_mask = mask_past[:, -1, 0]
        lag = seasonal_lag if y_past.shape[1] >= seasonal_lag else 1
        seasonal = y_past[:, -lag, 0]
        last_list.append(last.cpu().numpy())
        seasonal_list.append(seasonal.cpu().numpy())
        last_mask_list.append(last_mask.cpu().numpy())
    last_vals = np.concatenate(last_list, axis=0)
    seasonal_vals = np.concatenate(seasonal_list, axis=0)
    last_mask_vals = np.concatenate(last_mask_list, axis=0)
    return last_vals, seasonal_vals, last_mask_vals


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_last: np.ndarray,
    mask: np.ndarray,
) -> float:
    true_dir = np.sign(y_true - y_last[:, None])
    pred_dir = np.sign(y_pred - y_last[:, None])
    correct = (true_dir == pred_dir) & (mask > 0)
    denom = np.sum(mask)
    return float(np.sum(correct) / max(denom, 1.0))


def _num_patches(L: int, P: int, S: int) -> int:
    if L < P:
        return 0
    return int((L - P) // S + 1)


def compute_turning_delays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n, H = y_true.shape
    dy_true = np.zeros_like(y_true)
    dy_pred = np.zeros_like(y_pred)
    dy_true[:, 0] = y_true[:, 0] - ref
    dy_pred[:, 0] = y_pred[:, 0] - ref
    if H > 1:
        dy_true[:, 1:] = y_true[:, 1:] - y_true[:, :-1]
        dy_pred[:, 1:] = y_pred[:, 1:] - y_pred[:, :-1]
    true_turn = np.full(n, -1, dtype=np.int64)
    pred_turn = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for h in range(1, H):
            if mask[i, h] <= 0 or mask[i, h - 1] <= 0:
                continue
            s_prev = np.sign(dy_true[i, h - 1])
            s_cur = np.sign(dy_true[i, h])
            if true_turn[i] < 0 and s_prev != 0 and s_cur != 0 and s_cur != s_prev:
                true_turn[i] = h + 1
            s_prev_p = np.sign(dy_pred[i, h - 1])
            s_cur_p = np.sign(dy_pred[i, h])
            if pred_turn[i] < 0 and s_prev_p != 0 and s_cur_p != 0 and s_cur_p != s_prev_p:
                pred_turn[i] = h + 1
        if true_turn[i] >= 0 and pred_turn[i] < 0:
            pred_turn[i] = H + 1
    return true_turn, pred_turn


def compute_regime_report(
    eval_y: np.ndarray,
    q50: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    mask: np.ndarray,
    last_vals: np.ndarray,
    regime_labels: np.ndarray,
    target_mode: str,
    no_trade_quantile: float = 0.8,
    no_trade_min_count: int = 50,
    attn_ratio: np.ndarray | None = None,
) -> Dict[str, object]:
    n, H = eval_y.shape
    n_regimes = len(REGIME_NAMES)
    width = q90 - q10
    metrics: Dict[str, np.ndarray] = {}
    for name in [
        "mae",
        "pinball50",
        "dir_acc",
        "drift_slope",
        "turn_delay",
        "pnl_proxy",
        "turnover_proxy",
    ]:
        metrics[name] = np.full((n_regimes, H), np.nan, dtype=np.float32)
    width_thr = np.full((n_regimes, H), np.nan, dtype=np.float32)
    trade_mask = np.ones_like(mask, dtype=bool)
    for r in range(n_regimes):
        idx = regime_labels == r
        if not np.any(idx):
            continue
        for h in range(H):
            m = mask[idx, h] > 0
            w = width[idx, h][m]
            if w.size >= no_trade_min_count:
                thr = float(np.quantile(w, no_trade_quantile))
                width_thr[r, h] = thr
                trade_mask[idx, h] = width[idx, h] <= thr

    ref = np.zeros(n, dtype=np.float32) if target_mode != "level" else last_vals
    true_turn, pred_turn = compute_turning_delays(eval_y, q50, mask, ref)
    delay = np.full(n, np.nan, dtype=np.float32)
    valid_turn = true_turn >= 0
    delay[valid_turn] = np.maximum(pred_turn[valid_turn] - true_turn[valid_turn], 0).astype(np.float32)

    for r in range(n_regimes):
        idx = regime_labels == r
        if not np.any(idx):
            continue
        for h in range(H):
            m = mask[idx, h] > 0
            if not np.any(m):
                continue
            y_true = eval_y[idx, h][m]
            y_pred = q50[idx, h][m]
            ref_h = ref[idx][m]
            metrics["mae"][r, h] = float(np.mean(np.abs(y_true - y_pred)))
            metrics["pinball50"][r, h] = float(np.mean(0.5 * np.abs(y_true - y_pred)))
            true_dir = np.sign(y_true - ref_h)
            pred_dir = np.sign(y_pred - ref_h)
            metrics["dir_acc"][r, h] = float(np.mean(true_dir == pred_dir))
            metrics["drift_slope"][r, h] = float(np.mean(np.abs((y_pred - ref_h) / float(h + 1))))
            trade_m = trade_mask[idx, h][m]
            if np.any(trade_m):
                side = np.sign(y_pred[trade_m] - ref_h[trade_m])
                pnl = side * (y_true[trade_m] - ref_h[trade_m])
                metrics["pnl_proxy"][r, h] = float(np.mean(pnl))
                metrics["turnover_proxy"][r, h] = float(np.mean(trade_m))
            turn_h = true_turn[idx]
            delay_h = delay[idx]
            match = (turn_h == (h + 1)) & ~np.isnan(delay_h)
            if np.any(match):
                metrics["turn_delay"][r, h] = float(np.mean(delay_h[match]))
    report: Dict[str, object] = {
        "regime_names": REGIME_NAMES,
        "metrics": metrics,
        "width_thresholds": width_thr,
    }
    if attn_ratio is not None:
        report["attr_ratio_coarse"] = attn_ratio
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test", "calib"])
    parser.add_argument("--use_cqr", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--out_npz", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--metrics_out", default=None)
    parser.add_argument(
        "--min_half_mode",
        default="none",
        choices=["none", "abs", "rel", "calib_abs", "calib_rel"],
    )
    parser.add_argument("--min_half", type=float, default=0.0)
    parser.add_argument("--min_half_rel", type=float, default=0.0)
    parser.add_argument("--min_half_quantile", type=float, default=0.05)
    parser.add_argument("--min_half_scale", type=float, default=0.1)
    parser.add_argument("--width_scaling", default="false")
    parser.add_argument("--diagnostics", default="false")
    parser.add_argument("--attn_diagnostics", default="false")
    parser.add_argument("--width_scaling_mode", default="global", choices=["global", "per_horizon"])
    parser.add_argument("--target_coverage", type=float, default=0.90)
    parser.add_argument("--width_scaling_s_lo", type=float, default=0.10)
    parser.add_argument("--width_scaling_s_hi", type=float, default=1.50)
    parser.add_argument("--width_scaling_iters", type=int, default=40)
    parser.add_argument("--calib_npz", default=None)
    parser.add_argument("--out_s_npz", default=None)
    parser.add_argument("--save_indices", default="true")
    parser.add_argument("--missingness_pattern", default="none", choices=["none", "random", "burst"])
    parser.add_argument("--missingness_rate", type=float, default=0.0)
    parser.add_argument("--missingness_burst_prob", type=float, default=0.0)
    parser.add_argument("--missingness_burst_len", type=int, default=24)
    parser.add_argument("--missingness_seed", type=int, default=7)
    parser.add_argument("--missingness_apply_to_calib", default="false")
    parser.add_argument("--split_purge", type=int, default=None)
    parser.add_argument("--split_embargo", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("training", {}).get("seed", 7)
    set_seed(seed)
    regime_cfg = cfg.get("evaluation", {}).get("regime", {})
    regime_enabled = bool(regime_cfg.get("enabled", True))
    regime_window = int(regime_cfg.get("window", cfg["data"]["L"]))
    regime_eps = float(regime_cfg.get("eps", 1e-6))

    series_list = build_series_list(
        cfg["data"]["path"],
        observed_only=cfg["data"].get("observed_only", False),
        min_active_ratio=float(cfg["data"].get("universe_min_active_ratio", 0.0)),
        min_active_points=int(cfg["data"].get("universe_min_active_points", 0)),
        active_end=cfg["data"].get("universe_active_end"),
        min_future_ratio=float(cfg["data"].get("universe_min_future_ratio", 0.0)),
        future_horizon=int(cfg["data"].get("H", cfg["data"].get("future_horizon", 0))),
    )
    lengths = [len(s.y) for s in series_list]
    split_train_end = cfg["data"].get("split_train_end", None)
    split_val_end = cfg["data"].get("split_val_end", None)
    split_test_end = cfg["data"].get("split_test_end", None)
    if split_train_end is not None or split_val_end is not None or split_test_end is not None:
        if split_train_end is None or split_val_end is None:
            raise ValueError("split_train_end and split_val_end must both be set when using split overrides.")
        T_min = min(lengths)
        train_end = int(split_train_end)
        val_end = int(split_val_end)
        test_end = int(split_test_end) if split_test_end is not None else T_min
        if not (0 <= train_end <= val_end <= test_end <= T_min):
            raise ValueError(
                f"Invalid split overrides: train_end={train_end} val_end={val_end} "
                f"test_end={test_end} T_min={T_min}"
            )
        split = TimeSplit(train_end=train_end, val_end=val_end, test_end=test_end)
    else:
        split = make_time_splits(min(lengths), cfg["data"].get("train_frac", 0.7), cfg["data"].get("val_frac", 0.15))
    split_purge = int(cfg["data"].get("split_purge", 0)) if args.split_purge is None else int(args.split_purge)
    split_embargo = int(cfg["data"].get("split_embargo", 0)) if args.split_embargo is None else int(args.split_embargo)
    dir_cfg = cfg.get("data", {}).get("direction_features", {})
    if dir_cfg.get("enabled", False):
        window = int(dir_cfg.get("window", 24))
        append_direction_features(series_list, window=window, split_end=split.train_end)
        expected = int(cfg["data"].get("past_feat_dim", 0))
        actual = series_list[0].x_past_feats.shape[1] if series_list[0].x_past_feats is not None else 0
        if expected != actual:
            raise ValueError(f"past_feat_dim mismatch: config={expected} actual={actual}")
    indices = generate_panel_origins(lengths, cfg["data"]["L"], cfg["data"]["H"], cfg["data"].get("step", cfg["data"]["H"]))
    horizon = cfg["data"]["H"]
    val_idx = select_indices_by_time(indices, split, "val", horizon=horizon, purge=split_purge, embargo=split_embargo)
    test_idx = select_indices_by_time(indices, split, "test", horizon=horizon, purge=split_purge, embargo=split_embargo)
    train_idx = None
    if regime_enabled:
        train_idx = select_indices_by_time(
            indices,
            split,
            "train",
            horizon=horizon,
            purge=split_purge,
            embargo=split_embargo,
        )

    pre = apply_scaling(
        series_list,
        split.train_end,
        scale_x=cfg["data"].get("scale_x", True),
        scale_y=cfg["data"].get("scale_y", True),
    )
    min_past_obs = cfg["data"].get("min_past_obs", 1)
    min_future_obs = cfg["data"].get("min_future_obs", 1)
    future_mode = cfg["data"].get("future_obs_mode", "count")
    val_idx = filter_indices_with_observed(
        series_list, val_idx, cfg["data"]["L"], cfg["data"]["H"], min_past_obs, min_future_obs, future_mode
    )
    test_idx = filter_indices_with_observed(
        series_list, test_idx, cfg["data"]["L"], cfg["data"]["H"], min_past_obs, min_future_obs, future_mode
    )
    if regime_enabled and train_idx is not None:
        train_idx = filter_indices_with_observed(
            series_list,
            train_idx,
            cfg["data"]["L"],
            cfg["data"]["H"],
            min_past_obs,
            min_future_obs,
            future_mode,
        )

    target_mode = cfg["data"].get("target_mode", "level")
    target_log_eps = float(cfg["data"].get("target_log_eps", 1e-6))
    val_ds_base = WindowedDataset(
        series_list,
        val_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        target_mode=target_mode,
        target_log_eps=target_log_eps,
    )
    test_ds_base = WindowedDataset(
        series_list,
        test_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        target_mode=target_mode,
        target_log_eps=target_log_eps,
    )
    apply_missingness = args.missingness_pattern != "none"
    miss_cfg = None
    if apply_missingness:
        miss_cfg = MissingnessConfig(
            pattern=args.missingness_pattern,
            rate=args.missingness_rate,
            burst_prob=args.missingness_burst_prob,
            burst_len=args.missingness_burst_len,
            seed=args.missingness_seed,
        )
        val_ds = MissingnessDataset(val_ds_base, miss_cfg)
        test_ds = MissingnessDataset(test_ds_base, miss_cfg)
    else:
        val_ds = val_ds_base
        test_ds = test_ds_base
    batch_size = cfg["training"].get("batch_size", 32)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    if apply_missingness and parse_bool(args.missingness_apply_to_calib):
        calib_loader = DataLoader(MissingnessDataset(val_ds_base, miss_cfg), batch_size=batch_size)
    else:
        calib_loader = DataLoader(val_ds_base, batch_size=batch_size)
    train_loader = None
    if regime_enabled and train_idx:
        train_ds = WindowedDataset(
            series_list,
            train_idx,
            cfg["data"]["L"],
            cfg["data"]["H"],
            target_mode=target_mode,
            target_log_eps=target_log_eps,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size)

    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    device_str = cfg["training"].get("device", "cpu")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        print("warning: cuda requested but not available; falling back to cpu")
        device_str = "cpu"
    device = torch.device(device_str)
    model.to(device)
    model.eval()
    regime_thresholds = None
    if regime_enabled and train_loader is not None:
        train_feats = collect_regime_features(train_loader, device, regime_window, regime_eps)
        regime_thresholds = fit_regime_thresholds(
            train_feats,
            trend_q=float(regime_cfg.get("trend_q", 0.7)),
            mr_q=float(regime_cfg.get("mr_q", 0.3)),
            vol_q=float(regime_cfg.get("vol_q", 0.7)),
            chop_q=float(regime_cfg.get("chop_q", 0.7)),
            jump_q=float(regime_cfg.get("jump_q", 0.9)),
        )

    quantiles = cfg["data"]["quantiles"]
    q10_idx = quantiles.index(0.1)
    q90_idx = quantiles.index(0.9)
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    split_name = args.split
    if split_name == "calib":
        split_name = "val"
    eval_loader = val_loader if split_name == "val" else test_loader

    use_cqr = cfg.get("calibration", {}).get("cqr", {}).get("enabled", False)
    if args.use_cqr != "auto":
        use_cqr = parse_bool(args.use_cqr)
    width_scaling = parse_bool(args.width_scaling)
    diagnostics = parse_bool(args.diagnostics)
    attn_diag = parse_bool(args.attn_diagnostics)
    if width_scaling and args.calib_npz is None:
        raise ValueError("--calib_npz is required when --width_scaling is true.")
    min_half_mode = args.min_half_mode
    if min_half_mode in {"calib_abs", "calib_rel"} and args.calib_npz is None:
        raise ValueError("--calib_npz is required when --min_half_mode is calib_abs/calib_rel.")

    calibrator = None
    if use_cqr:
        val_y, val_q, _, _, _ = collect_predictions(model, calib_loader, device)
        val_q10 = val_q[..., q10_idx]
        val_q90 = val_q[..., q90_idx]
        cal_cfg = cfg.get("calibration", {}).get("cqr", {})
        per_h = cal_cfg.get("scope", "PER_HORIZON") == "PER_HORIZON"
        calibrator = CQRCalibrator(alpha=0.1, per_horizon=per_h)
        calibrator.fit(val_q10, val_q90, val_y)

    need_meta = regime_enabled or diagnostics
    eval_y, eval_q, dec_out, attn, meta = collect_predictions(
        model,
        eval_loader,
        device,
        diagnostics=diagnostics,
        attn=attn_diag,
        return_meta=need_meta,
        regime_window=regime_window,
        regime_eps=regime_eps,
    )
    base_last, base_seasonal, base_last_mask = collect_baselines(eval_loader, device)
    eval_q = np.nan_to_num(eval_q, nan=0.0)
    q10 = eval_q[..., q10_idx]
    q50 = eval_q[..., q50_idx]
    q90 = eval_q[..., q90_idx]
    if calibrator is not None:
        q10, q90 = calibrator.apply(q10, q90)
    q10 = np.nan_to_num(q10, nan=0.0, posinf=1e6, neginf=-1e6)
    q90 = np.nan_to_num(q90, nan=0.0, posinf=1e6, neginf=-1e6)

    eval_q_adj = eval_q.copy()
    eval_q_adj[..., q10_idx] = q10
    eval_q_adj[..., q90_idx] = q90

    min_half = None
    min_half_rel = None
    if min_half_mode == "abs":
        min_half = args.min_half
    elif min_half_mode == "rel":
        min_half_rel = args.min_half_rel
    elif min_half_mode in {"calib_abs", "calib_rel"}:
        calib = np.load(args.calib_npz)
        q10_c = np.nan_to_num(calib["q10"], nan=0.0)
        q50_c = np.nan_to_num(calib["q50"], nan=0.0)
        q90_c = np.nan_to_num(calib["q90"], nan=0.0)
        if min_half_mode == "calib_abs":
            min_half = fit_min_half_abs(q10_c, q50_c, q90_c, args.min_half_quantile, args.min_half_scale)
            print(f"min_half_abs={min_half:.6f}")
        else:
            min_half_rel = fit_min_half_rel(q10_c, q50_c, q90_c, args.min_half_quantile, args.min_half_scale)
            print(f"min_half_rel={min_half_rel:.6f}")

    if min_half_mode != "none":
        q10, q50, q90 = apply_min_half_clamp(q10, q50, q90, min_half=min_half, min_half_rel=min_half_rel)
        eval_q_adj[..., q10_idx] = q10
        eval_q_adj[..., q90_idx] = q90

    if width_scaling:
        calib = np.load(args.calib_npz)
        y_c = calib["y"]
        q10_c = calib["q10"]
        q50_c = calib["q50"]
        q90_c = calib["q90"]
        mask_c = calib["mask"] if "mask" in calib else np.isfinite(y_c).astype(np.float32)
        if min_half_mode != "none":
            q10_c, q50_c, q90_c = apply_min_half_clamp(q10_c, q50_c, q90_c, min_half=min_half, min_half_rel=min_half_rel)
        if args.width_scaling_mode == "per_horizon":
            s = fit_s_per_horizon(
                y_c,
                q10_c,
                q50_c,
                q90_c,
                target=args.target_coverage,
                s_lo=args.width_scaling_s_lo,
                s_hi=args.width_scaling_s_hi,
                iters=args.width_scaling_iters,
                mask=mask_c,
            )
            print(
                "width_scaling_s_stats",
                f"min={float(np.min(s)):.3f}",
                f"median={float(np.median(s)):.3f}",
                f"max={float(np.max(s)):.3f}",
            )
            if np.min(s) < 0.2 or np.max(s) > 2.0:
                print("warning: per-horizon s has extreme values; consider global scaling or more calib data.")
        else:
            s = fit_s_global(
                y_c,
                q10_c,
                q50_c,
                q90_c,
                target=args.target_coverage,
                s_lo=args.width_scaling_s_lo,
                s_hi=args.width_scaling_s_hi,
                iters=args.width_scaling_iters,
                mask=mask_c,
            )
            print(f"width_scaling_s={s:.3f}")
        q10, q50, q90 = apply_width_scaling(q10, q50, q90, s)
        eval_q_adj[..., q10_idx] = q10
        eval_q_adj[..., q90_idx] = q90
        if args.out_s_npz:
            Path(args.out_s_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out_s_npz, s=s)
        y_c = np.nan_to_num(y_c, nan=0.0)
        lo_c, _, hi_c = apply_width_scaling(q10_c, q50_c, q90_c, s)
        cov_c = masked_coverage(y_c, lo_c, hi_c, mask_c)
        print(f"width_scaling_calib_coverage={cov_c:.4f} target={args.target_coverage:.2f}")

    mask = np.isfinite(eval_y).astype(np.float32)
    eval_y = np.nan_to_num(eval_y, nan=0.0)
    regime_labels = None
    if regime_enabled and meta is not None:
        eval_feats = {k: meta[k] for k in ["trend_t", "autocorr", "vol", "chop", "jump"] if k in meta}
        if regime_thresholds is None:
            regime_thresholds = fit_regime_thresholds(
                eval_feats,
                trend_q=float(regime_cfg.get("trend_q", 0.7)),
                mr_q=float(regime_cfg.get("mr_q", 0.3)),
                vol_q=float(regime_cfg.get("vol_q", 0.7)),
                chop_q=float(regime_cfg.get("chop_q", 0.7)),
                jump_q=float(regime_cfg.get("jump_q", 0.9)),
            )
        regime_labels = label_regimes(eval_feats, regime_thresholds)
    last_mask = base_last_mask.reshape(-1, 1).astype(np.float32)
    mask_dir = mask * last_mask if target_mode == "level" else mask
    if target_mode == "level":
        if pre is not None:
            eval_y_orig = pre.inverse_y(eval_y)
            q50_orig = pre.inverse_y(q50)
            last_orig = pre.inverse_y(base_last)
            seasonal_orig = pre.inverse_y(base_seasonal)
        else:
            eval_y_orig = eval_y
            q50_orig = q50
            last_orig = base_last
            seasonal_orig = base_seasonal
        last_pred = np.repeat(last_orig[:, None], eval_y_orig.shape[1], axis=1)
        seasonal_pred = np.repeat(seasonal_orig[:, None], eval_y_orig.shape[1], axis=1)
    else:
        if pre is not None:
            eval_y_orig = pre.inverse_return(eval_y)
            q50_orig = pre.inverse_return(q50)
        else:
            eval_y_orig = eval_y
            q50_orig = q50
        last_orig = np.zeros_like(base_last)
        seasonal_orig = np.zeros_like(base_seasonal)
        last_pred = np.zeros_like(eval_y_orig)
        seasonal_pred = np.zeros_like(eval_y_orig)
    metrics = {
        "mae": masked_mae(eval_y, q50, mask),
        "rmse": masked_rmse(eval_y, q50, mask),
        "pinball": masked_pinball(eval_y, eval_q_adj, quantiles, mask),
        "coverage90": masked_coverage(eval_y, q10, q90, mask),
        "width90": masked_interval_width(q10, q90, mask),
        "collapse": horizon_collapse(q50),
        "crossing": quantile_crossing_rate(eval_q_adj),
        "mae_orig": masked_mae(eval_y_orig, q50_orig, mask),
        "rmse_orig": masked_rmse(eval_y_orig, q50_orig, mask),
        "mae_orig_last": masked_mae(eval_y_orig, last_pred, mask),
        "rmse_orig_last": masked_rmse(eval_y_orig, last_pred, mask),
        "mae_orig_seasonal": masked_mae(eval_y_orig, seasonal_pred, mask),
        "rmse_orig_seasonal": masked_rmse(eval_y_orig, seasonal_pred, mask),
        "dir_acc_model": directional_accuracy(eval_y_orig, q50_orig, last_orig, mask_dir),
        "dir_acc_last": directional_accuracy(eval_y_orig, last_pred, last_orig, mask_dir),
        "dir_acc_seasonal": directional_accuracy(eval_y_orig, seasonal_pred, last_orig, mask_dir),
    }
    metrics["coverage80"] = metrics["coverage90"]
    w = q90 - q10
    width_stats = width_stats_per_horizon(w, mask)
    metrics["width_zero_rate_mean"] = float(np.nanmean(width_stats["zero_rate"]))
    metrics["width_p95_mean"] = float(np.nanmean(width_stats["p95"]))
    metrics["width_p99_mean"] = float(np.nanmean(width_stats["p99"]))
    metrics["bimod_ratio_mean"] = float(np.nanmean(width_stats["bimod_ratio"]))
    bands = ((1, 6), (7, 12), (13, 24))
    for name, values in [
        ("zero_rate", width_stats["zero_rate"]),
        ("bimod_ratio", width_stats["bimod_ratio"]),
    ]:
        for band, val in band_summary(values, bands).items():
            metrics[f"{name}_band_{band}"] = val
    metrics["pred_sim"] = mean_offdiag_corr(q50)
    metrics["width_sim"] = mean_offdiag_corr(w)
    if diagnostics and dec_out is not None:
        metrics["tok_sim"] = mean_offdiag_cosine(dec_out)
    if attn_diag and attn is not None:
        if isinstance(attn, list):
            metrics["attn_sim"] = float(np.mean([attention_similarity(a) for a in attn]))
        else:
            metrics["attn_sim"] = attention_similarity(attn)
    half_overall = half_stats(q10, q50, q90, mask=mask, axis=None)
    metrics["half_zero_frac"] = float(half_overall["zero_frac"])
    metrics["half_p50"] = float(half_overall["p50"])
    metrics["half_p95"] = float(half_overall["p95"])
    metrics["half_p95_over_p50"] = float(half_overall["p95"] / max(half_overall["p50"], 1e-12))
    horizon = masked_horizon_metrics(eval_y, q50, mask)
    horizon["coverage90"] = coverage_per_horizon(eval_y, q10, q90, mask)
    horizon["width90"] = width_per_horizon(q10, q90, mask)
    horizon["pinball10"] = pinball_per_horizon(eval_y, q10, 0.1, mask)
    horizon["pinball50"] = pinball_per_horizon(eval_y, q50, 0.5, mask)
    horizon["pinball90"] = pinball_per_horizon(eval_y, q90, 0.9, mask)
    horizon["width_zero_rate"] = width_stats["zero_rate"]
    horizon["width_p95"] = width_stats["p95"]
    horizon["width_p99"] = width_stats["p99"]
    horizon["bimod_ratio"] = width_stats["bimod_ratio"]
    half_h = half_stats(q10, q50, q90, mask=mask, axis=0)
    horizon["half_zero_frac"] = half_h["zero_frac"]
    horizon["half_p50"] = half_h["p50"]
    horizon["half_p95"] = half_h["p95"]
    horizon["half_p95_over_p50"] = half_h["p95"] / np.maximum(half_h["p50"], 1e-12)
    metrics["coverage90_min"] = float(np.nanmin(horizon["coverage90"]))
    mask_full = np.all(mask > 0, axis=1)
    if np.any(mask_full):
        inside = (eval_y >= q10) & (eval_y <= q90)
        metrics["coverage_traj"] = float(np.mean(np.all(inside[mask_full], axis=1)))
    else:
        metrics["coverage_traj"] = float("nan")
    regime_report = None
    if regime_enabled and regime_labels is not None:
        no_trade_cfg = regime_cfg.get("no_trade", {})
        no_trade_quantile = float(no_trade_cfg.get("width_quantile", 0.8))
        no_trade_min = int(no_trade_cfg.get("min_count", 50))
        attn_ratio = None
        if attn_diag and attn is not None and cfg["patching"].get("fusion") == "TWO_ENCODERS_GATED":
            if not isinstance(attn, list):
                scales_cfg = cfg["patching"]["scales"]
                fine_cfg = next(s for s in scales_cfg if s["name"] == "fine")
                coarse_cfg = next(s for s in scales_cfg if s["name"] == "coarse")
                n_f = _num_patches(cfg["data"]["L"], fine_cfg["P"], fine_cfg["S"])
                n_c = _num_patches(cfg["data"]["L"], coarse_cfg["P"], coarse_cfg["S"])
                if cfg["patching"].get("summary", {}).get("enabled", False):
                    n_f += 1
                    n_c += 1
                attn_mean = attn.mean(axis=1)
                total = np.sum(attn_mean, axis=-1)
                coarse_sum = np.sum(attn_mean[..., n_f : n_f + n_c], axis=-1)
                ratio = coarse_sum / (total + 1e-8)
                attn_ratio = np.full((len(REGIME_NAMES), eval_y.shape[1]), np.nan, dtype=np.float32)
                for r in range(len(REGIME_NAMES)):
                    idx = regime_labels == r
                    if np.any(idx):
                        attn_ratio[r] = np.mean(ratio[idx], axis=0).astype(np.float32)
        regime_report = compute_regime_report(
            eval_y,
            q50,
            q10,
            q90,
            mask,
            base_last,
            regime_labels,
            target_mode,
            no_trade_quantile=no_trade_quantile,
            no_trade_min_count=no_trade_min,
            attn_ratio=attn_ratio,
        )
        if regime_thresholds is not None:
            regime_report["thresholds"] = regime_thresholds.as_dict()
        if meta is not None and "gate_logits" in meta:
            logit_diff = meta["gate_logits"][:, 1] - meta["gate_logits"][:, 0]
            bins = np.linspace(float(np.nanmin(logit_diff)), float(np.nanmax(logit_diff)), num=21)
            hist = np.zeros((len(REGIME_NAMES), bins.size - 1), dtype=np.float32)
            for r in range(len(REGIME_NAMES)):
                idx = regime_labels == r
                if np.any(idx):
                    hist[r] = np.histogram(logit_diff[idx], bins=bins)[0].astype(np.float32)
            regime_report["gate_logit_hist"] = hist
            regime_report["gate_logit_bins"] = bins.astype(np.float32)
        if meta is not None and "gate_weights" in meta:
            gate_w = meta["gate_weights"][:, 1]
            gate_mean = np.full(len(REGIME_NAMES), np.nan, dtype=np.float32)
            for r in range(len(REGIME_NAMES)):
                idx = regime_labels == r
                if np.any(idx):
                    gate_mean[r] = float(np.mean(gate_w[idx]))
            regime_report["gate_weight_mean"] = gate_mean
        if meta is not None and "moe_weights" in meta:
            moe_w = meta["moe_weights"][:, 0]
            moe_mean = np.full(len(REGIME_NAMES), np.nan, dtype=np.float32)
            for r in range(len(REGIME_NAMES)):
                idx = regime_labels == r
                if np.any(idx):
                    moe_mean[r] = float(np.mean(moe_w[idx]))
            regime_report["moe_trend_weight_mean"] = moe_mean
        if meta is not None and "moe_logits" in meta:
            logit_diff = meta["moe_logits"][:, 0] - meta["moe_logits"][:, 1]
            bins = np.linspace(float(np.nanmin(logit_diff)), float(np.nanmax(logit_diff)), num=21)
            hist = np.zeros((len(REGIME_NAMES), bins.size - 1), dtype=np.float32)
            for r in range(len(REGIME_NAMES)):
                idx = regime_labels == r
                if np.any(idx):
                    hist[r] = np.histogram(logit_diff[idx], bins=bins)[0].astype(np.float32)
            regime_report["moe_logit_hist"] = hist
            regime_report["moe_logit_bins"] = bins.astype(np.float32)
        h24 = min(eval_y.shape[1] - 1, 23)
        reg_summary = {}
        for r, name in enumerate(REGIME_NAMES):
            reg_summary[name] = {
                "mae": float(regime_report["metrics"]["mae"][r, h24]),
                "dir_acc": float(regime_report["metrics"]["dir_acc"][r, h24]),
                "pnl_proxy": float(regime_report["metrics"]["pnl_proxy"][r, h24]),
                "turnover": float(regime_report["metrics"]["turnover_proxy"][r, h24]),
            }
        print("regime_h24", reg_summary)
    output_path = Path(cfg.get("evaluation", {}).get("output_path", "metrics.npz"))
    if args.metrics_out:
        output_path = Path(args.metrics_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if regime_report is not None:
        np.savez(output_path, metrics=metrics, horizon=horizon, regime_report=regime_report)
    else:
        np.savez(output_path, metrics=metrics, horizon=horizon)
    print(metrics)

    if args.out_npz:
        out_npz = Path(args.out_npz)
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        save_indices = parse_bool(args.save_indices)
        npz_kwargs = {"y": eval_y, "q10": q10, "q50": q50, "q90": q90, "mask": mask}
        if regime_labels is not None:
            npz_kwargs["regime"] = regime_labels
        if save_indices:
            ds = eval_loader.dataset
            indices = None
            if hasattr(ds, "indices"):
                indices = ds.indices
            elif hasattr(ds, "base") and hasattr(ds.base, "indices"):
                indices = ds.base.indices
            if indices is not None:
                series_idx = np.array([i[0] for i in indices], dtype=np.int64)
                origin_t = np.array([i[1] for i in indices], dtype=np.int64)
                npz_kwargs["series_idx"] = series_idx
                npz_kwargs["origin_t"] = origin_t
        np.savez(out_npz, **npz_kwargs)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        H = eval_y.shape[1]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "h",
                    "coverage90",
                    "width90",
                    "width_zero_rate",
                    "width_p95",
                    "width_p99",
                    "bimod_ratio",
                    "pinball10",
                    "pinball50",
                    "pinball90",
                ]
            )
            for h in range(H):
                writer.writerow(
                    [
                        h + 1,
                        float(horizon["coverage90"][h]),
                        float(horizon["width90"][h]),
                        float(horizon["width_zero_rate"][h]),
                        float(horizon["width_p95"][h]),
                        float(horizon["width_p99"][h]),
                        float(horizon["bimod_ratio"][h]),
                        float(horizon["pinball10"][h]),
                        float(horizon["pinball50"][h]),
                        float(horizon["pinball90"][h]),
                    ]
                )

    violations = _monotonic_violations(q10, q50, q90)
    if violations["q10_gt_q50"] > 0.0 or violations["q50_gt_q90"] > 0.0:
        print("warning: quantile monotonicity violations", violations)


if __name__ == "__main__":
    main()
