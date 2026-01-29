from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from backtest.harness import TimeSplit, generate_panel_origins, make_time_splits, select_indices_by_time
from data.loader import SeriesData, WindowedDataset, load_panel_npz, compress_series_observed
from data.features import append_direction_features
from data.preprocess import FoldFitPreprocessor
from model import ModelConfig, MultiScaleForecastModel, PatchScale
from utils import load_config, set_seed


def masked_pinball_loss_torch(
    y_true: torch.Tensor,
    q_pred: torch.Tensor,
    quantiles: List[float],
    mask: torch.Tensor,
    weights: List[float] | None = None,
) -> torch.Tensor:
    q_pred = torch.nan_to_num(q_pred, nan=0.0, posinf=1e6, neginf=-1e6)
    if weights is None:
        weights = [1.0] * len(quantiles)
    losses = []
    for i, tau in enumerate(quantiles):
        diff = (y_true - q_pred[..., i]) * mask
        loss = torch.maximum(tau * diff, (tau - 1) * diff)
        losses.append(loss * mask * weights[i])
    stacked = torch.stack(losses, dim=-1)
    denom = torch.clamp(mask.sum() * sum(weights), min=1.0)
    return stacked.sum() / denom


def masked_coverage_torch(
    y_true: torch.Tensor,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    q_low = torch.nan_to_num(q_low, nan=0.0, posinf=1e6, neginf=-1e6)
    q_high = torch.nan_to_num(q_high, nan=0.0, posinf=1e6, neginf=-1e6)
    within = (y_true >= q_low) & (y_true <= q_high) & (mask > 0)
    denom = torch.clamp(mask.sum(), min=1.0)
    return within.sum() / denom


def masked_interval_width_torch(
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    q_low = torch.nan_to_num(q_low, nan=0.0, posinf=1e6, neginf=-1e6)
    q_high = torch.nan_to_num(q_high, nan=0.0, posinf=1e6, neginf=-1e6)
    denom = torch.clamp(mask.sum(), min=1.0)
    return torch.sum((q_high - q_low) * mask) / denom


def compute_sigma_torch(y_past: torch.Tensor, mask: torch.Tensor, window: int) -> torch.Tensor:
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


def compute_magnitude_weights(
    delta: torch.Tensor,
    mask: torch.Tensor,
    power: float,
    min_w: float,
    max_w: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    abs_delta = torch.abs(delta)
    denom = torch.clamp(mask.sum(dim=0, keepdim=True), min=1.0)
    mean = torch.sum(abs_delta * mask, dim=0, keepdim=True) / denom
    mean = torch.clamp(mean, min=eps)
    weights = torch.pow(abs_delta / mean, power)
    if min_w > 0.0 or max_w < float("inf"):
        weights = torch.clamp(weights, min=min_w, max=max_w)
    return weights


def build_shared_mask(model: torch.nn.Module, params: List[torch.nn.Parameter]) -> List[bool]:
    dir_param_ids = {id(p) for name, p in model.named_parameters() if name.startswith("dir_")}
    return [id(p) not in dir_param_ids for p in params]


def grad_dot(
    grads_a: List[torch.Tensor | None],
    grads_b: List[torch.Tensor | None],
    mask: List[bool],
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for g_a, g_b, use in zip(grads_a, grads_b, mask):
        if not use or g_a is None or g_b is None:
            continue
        total = total + torch.sum(g_a * g_b)
    return total


def grad_norm_sq(
    grads: List[torch.Tensor | None],
    mask: List[bool],
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for g, use in zip(grads, mask):
        if not use or g is None:
            continue
        total = total + torch.sum(g * g)
    return total


def grad_norm(
    grads: List[torch.Tensor | None],
    mask: List[bool],
    device: torch.device,
) -> torch.Tensor:
    return torch.sqrt(torch.clamp(grad_norm_sq(grads, mask, device), min=0.0))


def sample_pair_indices(
    labels: torch.Tensor,
    n_pairs: int,
    strategy: str,
    strat_bins: int,
    device: torch.device,
) -> torch.Tensor:
    if labels.numel() < 2:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    if n_pairs <= 0 or strategy == "all":
        return torch.triu_indices(labels.numel(), labels.numel(), 1, device=device)
    if strategy == "random":
        i = torch.randint(0, labels.numel(), (n_pairs,), device=device)
        j = torch.randint(0, labels.numel(), (n_pairs,), device=device)
        keep = i != j
        i = i[keep]
        j = j[keep]
        if i.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.stack([i, j], dim=0)
    # stratified
    n_bins = max(2, strat_bins)
    order = torch.argsort(labels)
    bins = torch.chunk(order, n_bins)
    combos = []
    for a in range(n_bins):
        for b in range(a + 1, n_bins):
            combos.append((a, b))
    if not combos:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    per = max(1, n_pairs // len(combos))
    pairs = []
    for a, b in combos:
        idx_a = bins[a]
        idx_b = bins[b]
        if idx_a.numel() == 0 or idx_b.numel() == 0:
            continue
        ia = idx_a[torch.randint(0, idx_a.numel(), (per,), device=device)]
        ib = idx_b[torch.randint(0, idx_b.numel(), (per,), device=device)]
        pairs.append(torch.stack([ia, ib], dim=0))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(pairs, dim=1)


def mcc_from_counts(tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    out = np.full_like(denom, np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = (tp[valid] * tn[valid] - fp[valid] * fn[valid]) / np.sqrt(denom[valid])
    return out


def project_conflicting(
    grads_a: List[torch.Tensor | None],
    grads_b: List[torch.Tensor | None],
    mask: List[bool],
    device: torch.device,
    eps: float = 1e-12,
) -> List[torch.Tensor | None]:
    dot = grad_dot(grads_a, grads_b, mask, device)
    if dot >= 0:
        return grads_a
    denom = grad_norm_sq(grads_b, mask, device)
    if denom <= 0:
        return grads_a
    scale = dot / (denom + eps)
    out: List[torch.Tensor | None] = []
    for g_a, g_b, use in zip(grads_a, grads_b, mask):
        if g_a is None:
            out.append(None)
        elif not use or g_b is None:
            out.append(g_a)
        else:
            out.append(g_a - scale * g_b)
    return out


def append_log(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_series_list(path: str, observed_only: bool = False) -> List[SeriesData]:
    series_list = load_panel_npz(path)
    if observed_only:
        series_list = compress_series_observed(series_list)
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
    mask_train = np.concatenate([s.mask[:split_end] for s in series_list], axis=0)
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
    scales = [PatchScale(**scale) for scale in patch_cfg["scales"]]
    head_cfg = cfg.get("head", {})
    dec_cfg = cfg.get("decoder", {})
    dual_path_cfg = dec_cfg.get("dual_path", {})
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
) -> List[tuple]:
    if min_past_obs <= 0 and min_future_obs <= 0:
        return indices
    keep = []
    for s_idx, t in indices:
        y = series_list[s_idx].y
        if y.ndim == 1:
            y = y[:, None]
        past = y[t - L + 1 : t + 1]
        future = y[t + 1 : t + H + 1]
        if np.isfinite(past).sum() >= min_past_obs and np.isfinite(future).sum() >= min_future_obs:
            keep.append((s_idx, t))
    return keep


def compute_delta_thresholds(
    series_list: List[SeriesData],
    indices: List[tuple],
    H: int,
    q: float,
    delta_mode: str = "origin",
) -> np.ndarray:
    buckets = [[] for _ in range(H)]
    for s_idx, t in indices:
        s = series_list[s_idx]
        y = s.y
        if y.ndim == 1:
            y = y[:, None]
        mask = s.mask
        if mask is None:
            mask = np.ones_like(y, dtype=np.float32)
        if mask.ndim == 1:
            mask = mask[:, None]
        if t + H >= y.shape[0]:
            continue
        y0 = y[t, 0]
        if mask[t, 0] <= 0:
            continue
        for h in range(H):
            idx = t + h + 1
            if mask[idx, 0] <= 0:
                continue
            if delta_mode == "step":
                prev_idx = t + h
                if mask[prev_idx, 0] <= 0:
                    continue
                prev = y[prev_idx, 0]
            else:
                prev = y0
            delta = y[idx, 0] - prev
            buckets[h].append(abs(float(delta)))
    tau = np.zeros(H, dtype=np.float32)
    for h in range(H):
        if buckets[h]:
            tau[h] = float(np.quantile(np.asarray(buckets[h], dtype=np.float32), q))
    return tau


def default_horizon_weights(H: int) -> np.ndarray:
    weights = np.ones(H, dtype=np.float32)
    weights[3:8] = 0.5
    weights[8:] = 0.2
    return weights


class TimestampBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        indices: List[tuple],
        min_assets: int = 5,
        max_assets: int = 0,
        shuffle: bool = True,
        seed: int = 7,
    ) -> None:
        self.max_assets = max_assets
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        by_t: Dict[int, List[int]] = {}
        for i, (_, t) in enumerate(indices):
            by_t.setdefault(int(t), []).append(i)
        self.groups = [g for g in by_t.values() if len(g) >= min_assets]

    def __iter__(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            self.rng.shuffle(order)
        for idx in order:
            g = self.groups[idx]
            if self.max_assets and len(g) > self.max_assets:
                sel = self.rng.choice(g, size=self.max_assets, replace=False)
                yield sel.tolist()
            else:
                yield g

    def __len__(self) -> int:
        return len(self.groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("training", {}).get("seed", 7)
    set_seed(seed)

    series_list = build_series_list(cfg["data"]["path"], observed_only=cfg["data"].get("observed_only", False))
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
    split_purge = int(cfg["data"].get("split_purge", 0))
    split_embargo = int(cfg["data"].get("split_embargo", 0))
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
    train_idx = select_indices_by_time(indices, split, "train", horizon=horizon, purge=split_purge, embargo=split_embargo)
    val_idx = select_indices_by_time(indices, split, "val", horizon=horizon, purge=split_purge, embargo=split_embargo)

    apply_scaling(
        series_list,
        split.train_end,
        scale_x=cfg["data"].get("scale_x", True),
        scale_y=cfg["data"].get("scale_y", True),
    )
    min_past_obs = cfg["data"].get("min_past_obs", 1)
    min_future_obs = cfg["data"].get("min_future_obs", 1)
    train_idx = filter_indices_with_observed(series_list, train_idx, cfg["data"]["L"], cfg["data"]["H"], min_past_obs, min_future_obs)
    val_idx = filter_indices_with_observed(series_list, val_idx, cfg["data"]["L"], cfg["data"]["H"], min_past_obs, min_future_obs)
    if not train_idx:
        raise RuntimeError("No training samples after filtering. Check min_past_obs/min_future_obs.")
    if not val_idx:
        print("warning: no validation samples after filtering.")

    target_mode = cfg["data"].get("target_mode", "level")
    target_log_eps = float(cfg["data"].get("target_log_eps", 1e-6))
    train_ds = WindowedDataset(
        series_list,
        train_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        target_mode=target_mode,
        target_log_eps=target_log_eps,
    )
    val_ds = WindowedDataset(
        series_list,
        val_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        target_mode=target_mode,
        target_log_eps=target_log_eps,
    )
    batch_size = cfg["training"].get("batch_size", 32)
    cs_batch_cfg = cfg.get("training", {}).get("cs_batching", {})
    cs_batch_enabled = bool(cs_batch_cfg.get("enabled", False))
    cs_batch_min_assets = int(cs_batch_cfg.get("min_assets", 5))
    cs_batch_max_assets = int(cs_batch_cfg.get("max_assets", 0))
    cs_batch_shuffle = bool(cs_batch_cfg.get("shuffle", True))
    cs_batch_val = bool(cs_batch_cfg.get("val_enabled", False))
    if cs_batch_enabled:
        train_sampler = TimestampBatchSampler(
            train_idx,
            min_assets=cs_batch_min_assets,
            max_assets=cs_batch_max_assets,
            shuffle=cs_batch_shuffle,
            seed=seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
        if cs_batch_val:
            val_sampler = TimestampBatchSampler(
                val_idx,
                min_assets=cs_batch_min_assets,
                max_assets=cs_batch_max_assets,
                shuffle=False,
                seed=seed,
            )
            val_loader = DataLoader(val_ds, batch_sampler=val_sampler)
        else:
            val_loader = DataLoader(val_ds, batch_size=batch_size)
        print(f"cs_batching enabled: train_groups={len(train_sampler)} val_groups={len(val_sampler) if cs_batch_val else 'n/a'}")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = build_model(cfg)
    device = torch.device(cfg["training"].get("device", "cpu"))
    model.to(device)

    all_params = [p for p in model.parameters() if p.requires_grad]
    optim_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        all_params,
        lr=optim_cfg.get("lr", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 0.01),
    )
    resume_from = cfg["training"].get("resume_from")
    resume_optimizer = bool(cfg["training"].get("resume_optimizer", True))
    resume_reset_epoch = bool(cfg["training"].get("resume_reset_epoch", False))
    resume_reset_lr = bool(cfg["training"].get("resume_reset_lr", True))
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if resume_optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if resume_reset_lr:
            lr = optim_cfg.get("lr", 1e-3)
            for group in optimizer.param_groups:
                group["lr"] = lr
        if not resume_reset_epoch and "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    quantiles = cfg["data"]["quantiles"]
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    q10_idx = quantiles.index(0.1) if 0.1 in quantiles else None
    q90_idx = quantiles.index(0.9) if 0.9 in quantiles else None
    epochs = cfg["training"].get("epochs", 10)
    if start_epoch >= epochs:
        raise RuntimeError(f"resume start_epoch={start_epoch} >= epochs={epochs}")
    early_cfg = cfg["training"].get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", False)
    early_patience = early_cfg.get("patience", 2)
    early_min_delta = early_cfg.get("min_delta", 0.0)
    early_metric = early_cfg.get("metric", "auto")
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    best_val_loss = float("inf")
    best_epoch_loss = -1
    best_state_loss = None
    best_val_dir = float("inf")
    best_epoch_dir = -1
    best_state_dir = None
    best_dir_wmcc = float("-inf")
    bad_epochs = 0

    loss_cfg = cfg.get("training", {}).get("loss", {})
    dir_cfg = cfg.get("training", {}).get("direction", {})
    gate_cfg = cfg.get("patching", {}).get("gate", {})
    dir_enabled = dir_cfg.get("enabled", False)
    dir_delta_mode = dir_cfg.get("delta_mode", "origin")
    dir_type = dir_cfg.get("type", "hierarchical")
    dir_delta_q = float(dir_cfg.get("delta_quantile", 0.33))
    dir_epsilon_mode = dir_cfg.get("epsilon_mode", "quantile")
    dir_epsilon_k = float(dir_cfg.get("epsilon_k", 1.0))
    dir_epsilon_window = int(dir_cfg.get("epsilon_window", 24))
    dir_loss_weight = float(dir_cfg.get("loss_weight", 0.1))
    dir_move_weight = float(dir_cfg.get("move_weight", 1.0))
    dir_dir_weight = float(dir_cfg.get("dir_weight", 1.0))
    dir_warmup_epochs = int(dir_cfg.get("warmup_epochs", 2))
    dir_weights_cfg = dir_cfg.get("horizon_weights")
    dir_mag_weighting = bool(dir_cfg.get("mag_weighting", False))
    dir_mag_power = float(dir_cfg.get("mag_weight_power", 1.0))
    dir_mag_min = float(dir_cfg.get("mag_weight_min", 0.0))
    dir_mag_max = float(dir_cfg.get("mag_weight_max", 5.0))
    pinball_weight = loss_cfg.get("pinball_weight", 1.0)
    mae_weight = loss_cfg.get("mae_weight", 0.0)
    quantile_weights = loss_cfg.get("quantile_weights")
    width_min = loss_cfg.get("width_min", 0.0)
    width_min_weight = loss_cfg.get("width_min_weight", 0.0)
    repulsion_weight = loss_cfg.get("repulsion_weight", 0.0)
    repulsion_scale = loss_cfg.get("repulsion_scale", 1.0)
    cumret24_weight = float(loss_cfg.get("cumret24_weight", 0.0))
    cumret24_warmup_epochs = int(loss_cfg.get("cumret24_warmup_epochs", 0))
    gate_entropy_weight = float(gate_cfg.get("entropy_weight", 0.0))
    moe_entropy_weight = float(loss_cfg.get("moe_entropy_weight", 0.0))
    moe_balance_weight = float(loss_cfg.get("moe_balance_weight", 0.0))
    cs_label = bool(cfg.get("training", {}).get("cs_label", False))
    cs_label_val = bool(cfg.get("training", {}).get("cs_label_val", False))
    cs_rank_cfg = cfg.get("training", {}).get("cs_rank", {})
    cs_rank_enabled = bool(cs_rank_cfg.get("enabled", False))
    cs_rank_weight = float(cs_rank_cfg.get("weight", 0.0))
    cs_rank_h = int(cs_rank_cfg.get("horizon", cfg["data"]["H"]))
    cs_rank_min_diff = float(cs_rank_cfg.get("min_diff", 0.0))
    cs_rank_use_residual = bool(cs_rank_cfg.get("use_residual", True))
    cs_rank_sample_pairs = int(cs_rank_cfg.get("sample_pairs", 0))
    cs_rank_sample_strategy = str(cs_rank_cfg.get("sample_strategy", "all")).lower()
    cs_rank_strat_bins = int(cs_rank_cfg.get("strat_bins", 3))
    if cs_rank_sample_strategy not in {"all", "random", "stratified"}:
        raise ValueError(f"Unknown cs_rank.sample_strategy: {cs_rank_sample_strategy}")
    cs_rank_mode = str(cs_rank_cfg.get("mode", "always")).lower()
    cs_rank_alternate_every = int(cs_rank_cfg.get("alternate_every", 1))
    if cs_rank_mode not in {"always", "alternate"}:
        raise ValueError(f"Unknown cs_rank.mode: {cs_rank_mode}")
    if cs_rank_alternate_every < 1:
        cs_rank_alternate_every = 1
    warned_cs_label = False

    grad_diag_cfg = cfg.get("training", {}).get("grad_diag", {})
    grad_diag_enabled = bool(grad_diag_cfg.get("enabled", False))
    grad_diag_every = int(grad_diag_cfg.get("every_n_steps", 200))
    grad_diag_shared_only = bool(grad_diag_cfg.get("shared_only", True))
    grad_diag_use_weighted = bool(grad_diag_cfg.get("use_weighted", True))

    mt_cfg = cfg.get("training", {}).get("multi_task", {})
    mt_mode = str(mt_cfg.get("mode", "none")).lower()
    if mt_mode not in {"none", "pcgrad", "gradnorm"}:
        raise ValueError(f"Unknown multi_task mode: {mt_mode}")
    mt_shared_only = bool(mt_cfg.get("shared_only", True))
    gradnorm_alpha = float(mt_cfg.get("gradnorm_alpha", 0.5))
    gradnorm_min = float(mt_cfg.get("gradnorm_min", 0.2))
    gradnorm_max = float(mt_cfg.get("gradnorm_max", 5.0))
    gradnorm_eps = float(mt_cfg.get("gradnorm_eps", 1e-6))
    gradnorm_weights = None
    gradnorm_init = None
    shared_mask = None
    shared_params: List[torch.nn.Parameter] = []
    shared_params_mask: List[bool] = []
    if mt_mode in {"pcgrad", "gradnorm"}:
        shared_mask = build_shared_mask(model, all_params) if mt_shared_only else [True] * len(all_params)
        if mt_mode == "gradnorm":
            gradnorm_weights = torch.ones(2, device=device)
        shared_params = [p for p, use in zip(all_params, shared_mask) if use]
        shared_params_mask = [True] * len(shared_params)
    diag_params: List[torch.nn.Parameter] = []
    diag_params_mask: List[bool] = []
    if grad_diag_enabled:
        diag_mask = build_shared_mask(model, all_params) if grad_diag_shared_only else [True] * len(all_params)
        diag_params = [p for p, use in zip(all_params, diag_mask) if use]
        diag_params_mask = [True] * len(diag_params)

    if dir_enabled:
        if dir_weights_cfg is None:
            dir_weights = default_horizon_weights(cfg["data"]["H"])
        else:
            dir_weights = np.asarray(dir_weights_cfg, dtype=np.float32)
            if dir_weights.shape[0] != cfg["data"]["H"]:
                raise ValueError("direction.horizon_weights length must equal H.")
        dir_weights_t = torch.tensor(dir_weights, device=device)
        if dir_epsilon_mode == "quantile":
            tau_h = compute_delta_thresholds(series_list, train_idx, cfg["data"]["H"], dir_delta_q, delta_mode=dir_delta_mode)
            tau_h_t = torch.tensor(tau_h, device=device)
        else:
            tau_h_t = None
            sqrt_h_t = torch.sqrt(torch.arange(1, cfg["data"]["H"] + 1, device=device, dtype=torch.float32))

    log_path = cfg["training"].get("log_path")
    log_every = cfg["training"].get("log_every", 200)
    log_path = Path(log_path) if log_path else None
    output_path = Path(cfg["training"].get("output_path", "model.pt"))
    save_every_epoch = bool(cfg["training"].get("save_every_epoch", False))
    cov_guard_min = cfg["training"].get("coverage_guard_min", None)
    cov_guard_max = cfg["training"].get("coverage_guard_max", None)
    cov_guard_fail = bool(cfg["training"].get("coverage_guard_fail", False))
    last_val_cov = None

    print(f"train_samples={len(train_idx)} val_samples={len(val_idx)}")

    global_step = 0
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        cumret24_weight_used = 0.0 if epoch < cumret24_warmup_epochs else cumret24_weight
        if cumret24_warmup_epochs > 0 or cumret24_weight > 0:
            print(f"cumret24_weight_used={cumret24_weight_used:.4f}")
        model.train()
        total_loss = 0.0
        running_loss = 0.0
        running_samples = 0
        rank_loss_sum = 0.0
        rank_batches = 0
        last_log_time = time.time()
        for batch_idx, (batch, target) in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            q_hat, extras = model(batch)
            y_true_raw = target[..., 0]
            mask = torch.isfinite(y_true_raw).float()
            origin_mask = batch["mask"][:, -1, 0]
            mask_dir = mask * origin_mask[:, None]
            y_true = torch.nan_to_num(y_true_raw, nan=0.0)
            q_hat_main = q_hat
            y_true_main = y_true
            y_true_cs = None
            if cs_batch_enabled and (cs_label or cs_rank_use_residual):
                denom_cs = torch.clamp(mask.sum(dim=0), min=1.0)
                y_mean = torch.sum(y_true * mask, dim=0) / denom_cs
                y_true_cs = y_true - y_mean
                if cs_label:
                    q_mean = torch.sum(q_hat * mask.unsqueeze(-1), dim=0) / denom_cs.unsqueeze(-1)
                    q_hat_main = q_hat - q_mean
                    y_true_main = y_true_cs
            elif cs_label and not warned_cs_label:
                print("warning: cs_label enabled but cs_batching disabled; skipping cs_label.")
                warned_cs_label = True
            point_loss = pinball_weight * masked_pinball_loss_torch(y_true_main, q_hat_main, quantiles, mask, quantile_weights)
            if mae_weight > 0:
                denom = torch.clamp(mask.sum(), min=1.0)
                point_loss = point_loss + mae_weight * (
                    torch.sum(torch.abs(y_true_main - q_hat_main[..., q50_idx]) * mask) / denom
                )
            if width_min_weight > 0 and width_min > 0 and q10_idx is not None and q90_idx is not None:
                width = q_hat_main[..., q90_idx] - q_hat_main[..., q10_idx]
                width_pen = torch.relu(width_min - width) * mask
                denom = torch.clamp(mask.sum(), min=1.0)
                point_loss = point_loss + width_min_weight * (width_pen.sum() / denom)
            if repulsion_weight > 0 and q10_idx is not None and q90_idx is not None:
                width = q_hat_main[..., q90_idx] - q_hat_main[..., q10_idx]
                repulsion = torch.exp(-width / max(repulsion_scale, 1e-6)) * mask
                denom = torch.clamp(mask.sum(), min=1.0)
                point_loss = point_loss + repulsion_weight * (repulsion.sum() / denom)
            if cumret24_weight_used > 0:
                if "cumret24" not in extras:
                    raise RuntimeError("cumret24_weight>0 but model did not return cumret24. Enable cumret24_head.")
                h_idx = min(y_true.shape[1] - 1, cfg["data"]["H"] - 1)
                target_h = y_true[:, h_idx]
                if target_mode == "level":
                    y_last = batch["y_past"][:, -1, 0]
                    target_h = target_h - y_last
                pred_h = extras["cumret24"]
                mask_h = mask[:, h_idx]
                denom = torch.clamp(mask_h.sum(), min=1.0)
                cumret_loss = torch.sum((pred_h - target_h) ** 2 * mask_h) / denom
                point_loss = point_loss + cumret24_weight_used * cumret_loss
            if gate_entropy_weight > 0 and "gate_weights" in extras:
                w = torch.clamp(extras["gate_weights"], 1e-8, 1.0)
                ent = -torch.sum(w * torch.log(w), dim=-1).mean()
                point_loss = point_loss - gate_entropy_weight * ent
            if (moe_entropy_weight > 0 or moe_balance_weight > 0) and "moe_weights" in extras:
                w = torch.clamp(extras["moe_weights"], 1e-8, 1.0)
                if moe_entropy_weight > 0:
                    ent = -torch.sum(w * torch.log(w), dim=-1).mean()
                    point_loss = point_loss - moe_entropy_weight * ent
                if moe_balance_weight > 0:
                    mean_w = w.mean(dim=0)
                    target = torch.full_like(mean_w, 1.0 / mean_w.shape[0])
                    balance = torch.sum((mean_w - target) ** 2)
                    point_loss = point_loss + moe_balance_weight * balance
            rank_loss = None
            do_point = True
            do_rank = True
            if cs_rank_mode == "alternate" and cs_rank_enabled and cs_rank_weight > 0.0:
                phase = (global_step // cs_rank_alternate_every) % 2
                do_rank = phase == 0
                do_point = not do_rank
            if cs_rank_enabled and cs_rank_weight > 0.0 and (cs_rank_mode == "always" or do_rank):
                if not cs_batch_enabled:
                    if not warned_cs_label:
                        print("warning: cs_rank enabled but cs_batching disabled; skipping cs_rank.")
                        warned_cs_label = True
                else:
                    h_idx = max(0, min(cs_rank_h, y_true.shape[1]) - 1)
                    if "rank_pred" in extras:
                        pred_rank = extras["rank_pred"][:, h_idx]
                    else:
                        pred_rank = q_hat[..., q50_idx][:, h_idx]
                    if cs_rank_use_residual:
                        if y_true_cs is None:
                            denom_cs = torch.clamp(mask.sum(dim=0), min=1.0)
                            y_mean = torch.sum(y_true * mask, dim=0) / denom_cs
                            y_true_cs = y_true - y_mean
                        label_rank = y_true_cs[:, h_idx]
                        pred_rank = pred_rank - pred_rank.mean()
                    else:
                        label_rank = y_true[:, h_idx]
                    valid_rank = mask[:, h_idx] > 0
                    pred_rank = pred_rank[valid_rank]
                    label_rank = label_rank[valid_rank]
                    if pred_rank.numel() > 1:
                        idx = sample_pair_indices(
                            label_rank,
                            cs_rank_sample_pairs,
                            cs_rank_sample_strategy,
                            cs_rank_strat_bins,
                            device,
                        )
                        if idx.numel() == 0:
                            diff = None
                            label_diff = None
                        else:
                            diff = pred_rank[idx[0]] - pred_rank[idx[1]]
                            label_diff = label_rank[idx[0]] - label_rank[idx[1]]
                        if cs_rank_min_diff > 0:
                            keep = torch.abs(label_diff) >= cs_rank_min_diff if label_diff is not None else None
                        else:
                            keep = label_diff != 0 if label_diff is not None else None
                        if keep is not None and torch.any(keep):
                            sign = torch.sign(label_diff[keep])
                            rank_loss = F.softplus(-sign * diff[keep]).mean()
                            rank_loss_sum += float(rank_loss.item())
                            rank_batches += 1
            main_loss = None
            if do_point:
                main_loss = point_loss
            if do_rank and rank_loss is not None:
                main_loss = (
                    main_loss + cs_rank_weight * rank_loss if main_loss is not None else cs_rank_weight * rank_loss
                )
            if main_loss is None:
                continue
            if (
                grad_diag_enabled
                and rank_loss is not None
                and point_loss.requires_grad
                and (global_step % max(1, grad_diag_every) == 0)
            ):
                rank_loss_scaled = rank_loss * cs_rank_weight if grad_diag_use_weighted else rank_loss
                g_point = torch.autograd.grad(point_loss, diag_params, retain_graph=True, allow_unused=True)
                g_rank = torch.autograd.grad(rank_loss_scaled, diag_params, retain_graph=True, allow_unused=True)
                g_point_norm = grad_norm(g_point, diag_params_mask, device)
                g_rank_norm = grad_norm(g_rank, diag_params_mask, device)
                g_dot = grad_dot(g_point, g_rank, diag_params_mask, device)
                denom = g_point_norm * g_rank_norm + 1e-12
                g_cos = g_dot / denom
                ratio = g_rank_norm / (g_point_norm + 1e-12)
                print(
                    "grad_diag step="
                    f"{global_step} point_norm={g_point_norm.item():.4f} "
                    f"rank_norm={g_rank_norm.item():.4f} cos={g_cos.item():.4f} "
                    f"rank_over_point={ratio.item():.4f}"
                )
                if log_path is not None:
                    append_log(
                        log_path,
                        {
                            "step": global_step,
                            "grad_point_norm": float(g_point_norm.item()),
                            "grad_rank_norm": float(g_rank_norm.item()),
                            "grad_rank_point_cos": float(g_cos.item()),
                            "grad_rank_over_point": float(ratio.item()),
                        },
                    )
            dir_loss_total = None
            if dir_enabled:
                y_last = batch["y_past"][:, -1, 0]
                if target_mode != "level":
                    y_last = torch.zeros_like(y_last)
                if dir_delta_mode == "step":
                    ref = torch.cat([y_last[:, None], y_true[:, :-1]], dim=1)
                else:
                    ref = y_last[:, None].expand_as(y_true)
                delta = y_true - ref
                if dir_epsilon_mode == "vol":
                    sigma_t = compute_sigma_torch(batch["y_past"], batch["mask"], dir_epsilon_window)
                    eps = dir_epsilon_k * sigma_t * sqrt_h_t
                else:
                    eps = tau_h_t
                if dir_mag_weighting:
                    mag_w = compute_magnitude_weights(delta, mask_dir, dir_mag_power, dir_mag_min, dir_mag_max)
                else:
                    mag_w = torch.ones_like(delta)
                if dir_type == "three_class":
                    if "dir_logits3" not in extras:
                        raise RuntimeError("direction head type three_class but logits missing.")
                    logits3 = extras["dir_logits3"]
                    flat = torch.abs(delta) < eps
                    up = delta >= eps
                    down = delta <= -eps
                    labels = torch.where(flat, torch.tensor(1, device=device), torch.where(up, torch.tensor(2, device=device), torch.tensor(0, device=device)))
                    valid_mask = mask_dir > 0
                    weights = dir_weights_t * valid_mask
                    if dir_mag_weighting:
                        weights = weights * mag_w
                    loss_raw = torch.nn.functional.cross_entropy(logits3.view(-1, 3), labels.view(-1), reduction="none")
                    loss_raw = loss_raw.view_as(labels) * weights
                    denom = torch.clamp(weights.sum(), min=1.0)
                    dir_loss_total = loss_raw.sum() / denom
                else:
                    if "dir_move_logits" not in extras or "dir_dir_logits" not in extras:
                        raise RuntimeError("direction head enabled but logits missing.")
                    move_label = (torch.abs(delta) >= eps) & (mask_dir > 0)
                    dir_label = (delta > 0) & (mask_dir > 0)
                    move_logits = extras["dir_move_logits"]
                    dir_logits = extras["dir_dir_logits"]
                    bce = torch.nn.functional.binary_cross_entropy_with_logits
                    move_mask = mask_dir * dir_weights_t
                    if dir_mag_weighting:
                        move_mask = move_mask * mag_w
                    move_loss_raw = bce(move_logits, move_label.float(), reduction="none")
                    move_loss_raw = move_loss_raw * move_mask
                    move_denom = torch.clamp(move_mask.sum(), min=1.0)
                    move_loss = move_loss_raw.sum() / move_denom
                    dir_mask = move_label.float() * dir_weights_t
                    if dir_mag_weighting:
                        dir_mask = dir_mask * mag_w
                    if dir_mask.sum() > 0:
                        dir_loss_raw = bce(dir_logits, dir_label.float(), reduction="none")
                        dir_loss_raw = dir_loss_raw * dir_mask
                        dir_denom = torch.clamp(dir_mask.sum(), min=1.0)
                        dir_loss = dir_loss_raw.sum() / dir_denom
                    else:
                        dir_loss = torch.tensor(0.0, device=device)
                    dir_loss_total = dir_move_weight * move_loss + dir_dir_weight * dir_loss
                if epoch < dir_warmup_epochs:
                    dir_loss_total = torch.tensor(0.0, device=device)
            use_dir_loss = (
                dir_loss_total is not None
                and dir_loss_total.requires_grad
                and dir_loss_weight > 0.0
            )
            if mt_mode == "gradnorm" and use_dir_loss:
                if gradnorm_weights is None:
                    gradnorm_weights = torch.ones(2, device=device)
                if gradnorm_init is None:
                    gradnorm_init = (main_loss.detach(), dir_loss_total.detach())
                w_main = gradnorm_weights[0]
                w_dir = gradnorm_weights[1]
                w_dir_total = dir_loss_weight * w_dir
                g_main = torch.autograd.grad(main_loss * w_main, shared_params, retain_graph=True, allow_unused=True)
                g_dir = torch.autograd.grad(dir_loss_total * w_dir_total, shared_params, retain_graph=True, allow_unused=True)
                g_main_norm = grad_norm(g_main, shared_params_mask, device)
                g_dir_norm = grad_norm(g_dir, shared_params_mask, device)
                g_avg = (g_main_norm + g_dir_norm) / 2.0
                loss_main_ratio = main_loss.detach() / (gradnorm_init[0] + gradnorm_eps)
                loss_dir_ratio = dir_loss_total.detach() / (gradnorm_init[1] + gradnorm_eps)
                r_avg = (loss_main_ratio + loss_dir_ratio) / 2.0
                r_avg = torch.clamp(r_avg, min=gradnorm_eps)
                target_main = g_avg * torch.pow(loss_main_ratio / r_avg, gradnorm_alpha)
                target_dir = g_avg * torch.pow(loss_dir_ratio / r_avg, gradnorm_alpha)
                with torch.no_grad():
                    gradnorm_weights[0] = torch.clamp(
                        gradnorm_weights[0] * (target_main / (g_main_norm + gradnorm_eps)),
                        min=gradnorm_min,
                        max=gradnorm_max,
                    )
                    gradnorm_weights[1] = torch.clamp(
                        gradnorm_weights[1] * (target_dir / (g_dir_norm + gradnorm_eps)),
                        min=gradnorm_min,
                        max=gradnorm_max,
                    )
                    weight_sum = gradnorm_weights.sum()
                    if weight_sum > 0:
                        gradnorm_weights.mul_(2.0 / weight_sum)
                w_main = gradnorm_weights[0]
                w_dir_total = dir_loss_weight * gradnorm_weights[1]
                loss_total = w_main * main_loss + w_dir_total * dir_loss_total
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()
            elif mt_mode == "pcgrad" and use_dir_loss:
                loss_main_w = main_loss
                loss_dir_w = dir_loss_weight * dir_loss_total
                grads_main = torch.autograd.grad(loss_main_w, all_params, retain_graph=True, allow_unused=True)
                grads_dir = torch.autograd.grad(loss_dir_w, all_params, retain_graph=True, allow_unused=True)
                mask_for_pc = shared_mask if shared_mask is not None else [True] * len(all_params)
                grads_main_proj = project_conflicting(grads_main, grads_dir, mask_for_pc, device)
                grads_dir_proj = project_conflicting(grads_dir, grads_main, mask_for_pc, device)
                combined: List[torch.Tensor | None] = []
                for g_main, g_dir in zip(grads_main_proj, grads_dir_proj):
                    if g_main is None and g_dir is None:
                        combined.append(None)
                    elif g_main is None:
                        combined.append(g_dir)
                    elif g_dir is None:
                        combined.append(g_main)
                    else:
                        combined.append(g_main + g_dir)
                optimizer.zero_grad()
                for param, grad in zip(all_params, combined):
                    if grad is None:
                        continue
                    param.grad = grad
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()
                loss_total = loss_main_w + loss_dir_w
            else:
                loss_total = main_loss
                if use_dir_loss:
                    loss_total = loss_total + dir_loss_weight * dir_loss_total
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimizer.step()
            total_loss += loss_total.item()
            running_loss += loss_total.item()
            batch_size = target.shape[0]
            running_samples += batch_size
            global_step += 1

            if log_every and (batch_idx + 1) % log_every == 0:
                elapsed = time.time() - last_log_time
                avg_loss = running_loss / max(log_every, 1)
                samples_per_s = running_samples / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} step={global_step} train_loss={avg_loss:.4f} "
                    f"samples_per_s={samples_per_s:.1f} lr={lr:.2e}"
                )
                if log_path is not None:
                    append_log(
                        log_path,
                        {
                            "time": time.time(),
                            "epoch": epoch,
                            "step": global_step,
                            "split": "train",
                            "loss": avg_loss,
                            "samples_per_s": samples_per_s,
                            "lr": lr,
                            "mt_w_main": float(gradnorm_weights[0].item()) if gradnorm_weights is not None else None,
                            "mt_w_dir": float(gradnorm_weights[1].item()) if gradnorm_weights is not None else None,
                        },
                    )
                running_loss = 0.0
                running_samples = 0
                last_log_time = time.time()
        if rank_batches > 0:
            print(f"train_rank_loss={rank_loss_sum / max(rank_batches, 1):.4f}")
        model.eval()
        val_loss = 0.0
        val_dir_loss = 0.0
        val_cov = 0.0
        val_width = 0.0
        cov_batches = 0
        val_rank_sum = 0.0
        val_rank_batches = 0
        dir_batches = 0
        dir_tp = None
        dir_tn = None
        dir_fp = None
        dir_fn = None
        if dir_enabled and dir_type != "three_class":
            dir_tp = torch.zeros(cfg["data"]["H"], device=device)
            dir_tn = torch.zeros(cfg["data"]["H"], device=device)
            dir_fp = torch.zeros(cfg["data"]["H"], device=device)
            dir_fn = torch.zeros(cfg["data"]["H"], device=device)
        with torch.no_grad():
            for batch, target in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                target = target.to(device)
                q_hat, extras = model(batch)
                y_true_raw = target[..., 0]
                mask = torch.isfinite(y_true_raw).float()
                y_true = torch.nan_to_num(y_true_raw, nan=0.0)
                q_hat_main = q_hat
                y_true_main = y_true
                y_true_cs = None
                use_cs_val = (cs_label and cs_batch_val) or cs_label_val
                if use_cs_val:
                    if not cs_batch_val:
                        if not warned_cs_label:
                            print("warning: cs_label_val enabled but cs_batching.val_enabled is False; skipping cs_label_val.")
                            warned_cs_label = True
                    else:
                        denom_cs = torch.clamp(mask.sum(dim=0), min=1.0)
                        y_mean = torch.sum(y_true * mask, dim=0) / denom_cs
                        y_true_cs = y_true - y_mean
                        q_mean = torch.sum(q_hat * mask.unsqueeze(-1), dim=0) / denom_cs.unsqueeze(-1)
                        q_hat_main = q_hat - q_mean
                        y_true_main = y_true_cs
                val_main = pinball_weight * masked_pinball_loss_torch(y_true_main, q_hat_main, quantiles, mask, quantile_weights)
                if cumret24_weight_used > 0:
                    if "cumret24" not in extras:
                        raise RuntimeError("cumret24_weight>0 but model did not return cumret24. Enable cumret24_head.")
                    h_idx = min(y_true.shape[1] - 1, cfg["data"]["H"] - 1)
                    target_h = y_true[:, h_idx]
                    if target_mode == "level":
                        y_last = batch["y_past"][:, -1, 0]
                        target_h = target_h - y_last
                    pred_h = extras["cumret24"]
                    mask_h = mask[:, h_idx]
                    denom = torch.clamp(mask_h.sum(), min=1.0)
                    cumret_loss = torch.sum((pred_h - target_h) ** 2 * mask_h) / denom
                    val_main = val_main + cumret24_weight_used * cumret_loss
                if cs_rank_enabled and cs_rank_weight > 0.0 and cs_batch_val:
                    h_idx = max(0, min(cs_rank_h, y_true.shape[1]) - 1)
                    if "rank_pred" in extras:
                        pred_rank = extras["rank_pred"][:, h_idx]
                    else:
                        pred_rank = q_hat[..., q50_idx][:, h_idx]
                    if cs_rank_use_residual:
                        if y_true_cs is None:
                            denom_cs = torch.clamp(mask.sum(dim=0), min=1.0)
                            y_mean = torch.sum(y_true * mask, dim=0) / denom_cs
                            y_true_cs = y_true - y_mean
                        label_rank = y_true_cs[:, h_idx]
                        pred_rank = pred_rank - pred_rank.mean()
                    else:
                        label_rank = y_true[:, h_idx]
                    valid_rank = mask[:, h_idx] > 0
                    pred_rank = pred_rank[valid_rank]
                    label_rank = label_rank[valid_rank]
                    if pred_rank.numel() > 1:
                        idx = sample_pair_indices(
                            label_rank,
                            cs_rank_sample_pairs,
                            cs_rank_sample_strategy,
                            cs_rank_strat_bins,
                            device,
                        )
                        if idx.numel() == 0:
                            diff = None
                            label_diff = None
                        else:
                            diff = pred_rank[idx[0]] - pred_rank[idx[1]]
                            label_diff = label_rank[idx[0]] - label_rank[idx[1]]
                        if cs_rank_min_diff > 0:
                            keep = torch.abs(label_diff) >= cs_rank_min_diff if label_diff is not None else None
                        else:
                            keep = label_diff != 0 if label_diff is not None else None
                        if keep is not None and torch.any(keep):
                            sign = torch.sign(label_diff[keep])
                            rank_loss = F.softplus(-sign * diff[keep]).mean()
                            val_rank_sum += float(rank_loss.item())
                            val_rank_batches += 1
                val_loss += val_main.item()
                if dir_enabled:
                    y_last = batch["y_past"][:, -1, 0]
                    if target_mode != "level":
                        y_last = torch.zeros_like(y_last)
                    if dir_delta_mode == "step":
                        ref = torch.cat([y_last[:, None], y_true[:, :-1]], dim=1)
                    else:
                        ref = y_last[:, None].expand_as(y_true)
                    delta = y_true - ref
                    if dir_epsilon_mode == "vol":
                        sigma_t = compute_sigma_torch(batch["y_past"], batch["mask"], dir_epsilon_window)
                        eps = dir_epsilon_k * sigma_t * sqrt_h_t
                    else:
                        eps = tau_h_t
                    if dir_mag_weighting:
                        mag_w = compute_magnitude_weights(delta, mask, dir_mag_power, dir_mag_min, dir_mag_max)
                    else:
                        mag_w = torch.ones_like(delta)
                    if dir_type == "three_class":
                        if "dir_logits3" not in extras:
                            raise RuntimeError("direction head type three_class but logits missing.")
                        logits3 = extras["dir_logits3"]
                        flat = torch.abs(delta) < eps
                        up = delta >= eps
                        down = delta <= -eps
                        labels = torch.where(
                            flat,
                            torch.tensor(1, device=device),
                            torch.where(up, torch.tensor(2, device=device), torch.tensor(0, device=device)),
                        )
                        valid_mask = mask > 0
                        weights = dir_weights_t * valid_mask
                        if dir_mag_weighting:
                            weights = weights * mag_w
                        loss_raw = torch.nn.functional.cross_entropy(
                            logits3.view(-1, 3),
                            labels.view(-1),
                            reduction="none",
                        )
                        loss_raw = loss_raw.view_as(labels) * weights
                        denom = torch.clamp(weights.sum(), min=1.0)
                        dir_loss = loss_raw.sum() / denom
                    else:
                        if "dir_move_logits" not in extras or "dir_dir_logits" not in extras:
                            raise RuntimeError("direction head enabled but logits missing.")
                        move_label = (torch.abs(delta) >= eps) & (mask > 0)
                        dir_label = (delta > 0) & (mask > 0)
                        move_logits = extras["dir_move_logits"]
                        dir_logits = extras["dir_dir_logits"]
                        bce = torch.nn.functional.binary_cross_entropy_with_logits
                        move_mask = mask * dir_weights_t
                        if dir_mag_weighting:
                            move_mask = move_mask * mag_w
                        move_loss_raw = bce(move_logits, move_label.float(), reduction="none")
                        move_loss_raw = move_loss_raw * move_mask
                        move_denom = torch.clamp(move_mask.sum(), min=1.0)
                        move_loss = move_loss_raw.sum() / move_denom
                        dir_mask = move_label.float() * dir_weights_t
                        if dir_mag_weighting:
                            dir_mask = dir_mask * mag_w
                        if dir_mask.sum() > 0:
                            dir_loss_raw = bce(dir_logits, dir_label.float(), reduction="none")
                            dir_loss_raw = dir_loss_raw * dir_mask
                            dir_denom = torch.clamp(dir_mask.sum(), min=1.0)
                            dir_loss = dir_loss_raw.sum() / dir_denom
                        else:
                            dir_loss = torch.tensor(0.0, device=device)
                        dir_loss = dir_move_weight * move_loss + dir_dir_weight * dir_loss
                        if dir_tp is not None:
                            dir_pred = torch.sigmoid(dir_logits) >= 0.5
                            move_mask = move_label & (mask > 0)
                            dir_tp += torch.sum(move_mask & dir_label & dir_pred, dim=0)
                            dir_tn += torch.sum(move_mask & (~dir_label) & (~dir_pred), dim=0)
                            dir_fp += torch.sum(move_mask & (~dir_label) & dir_pred, dim=0)
                            dir_fn += torch.sum(move_mask & dir_label & (~dir_pred), dim=0)
                    val_dir_loss += dir_loss.item()
                    dir_batches += 1
                if q10_idx is not None and q90_idx is not None:
                    q10 = q_hat[..., q10_idx]
                    q90 = q_hat[..., q90_idx]
                    val_cov += masked_coverage_torch(y_true, q10, q90, mask).item()
                    val_width += masked_interval_width_torch(q10, q90, mask).item()
                    cov_batches += 1
        val_loss = val_loss / max(len(val_loader), 1)
        if dir_batches > 0:
            val_dir_loss = val_dir_loss / dir_batches
        val_dir_wmcc = None
        if dir_tp is not None:
            tp = dir_tp.detach().cpu().numpy()
            tn = dir_tn.detach().cpu().numpy()
            fp = dir_fp.detach().cpu().numpy()
            fn = dir_fn.detach().cpu().numpy()
            mcc = mcc_from_counts(tp, tn, fp, fn)
            weights = default_horizon_weights(cfg["data"]["H"])
            valid = np.isfinite(mcc)
            if np.any(valid):
                val_dir_wmcc = float(np.sum(mcc[valid] * weights[valid]) / max(np.sum(weights[valid]), 1e-6))
        epoch_time = time.time() - epoch_start
        if cov_batches > 0:
            val_cov /= cov_batches
            val_width /= cov_batches
            last_val_cov = val_cov
            print(
                f"epoch={epoch} train_loss={total_loss/len(train_loader):.4f} "
                f"val_loss={val_loss:.4f} val_cov80={val_cov:.4f} val_width80={val_width:.4f} "
                f"epoch_time_s={epoch_time:.1f}"
            )
        else:
            print(
                f"epoch={epoch} train_loss={total_loss/len(train_loader):.4f} "
                f"val_loss={val_loss:.4f} epoch_time_s={epoch_time:.1f}"
            )
        if dir_batches > 0:
            print(f"val_dir_loss={val_dir_loss:.4f}")
        if val_dir_wmcc is not None:
            print(f"val_dir_wmcc={val_dir_wmcc:.4f}")
        if val_rank_batches > 0:
            print(f"val_rank_loss={val_rank_sum / max(val_rank_batches, 1):.4f}")

        if log_path is not None:
            record = {
                "time": time.time(),
                "epoch": epoch,
                "step": global_step,
                "split": "val",
                "loss": val_loss,
                "epoch_time_s": epoch_time,
            }
            if val_rank_batches > 0:
                record["rank_loss"] = val_rank_sum / max(val_rank_batches, 1)
            if cov_batches > 0:
                record["coverage80"] = val_cov
                record["width80"] = val_width
            if dir_batches > 0:
                record["dir_loss"] = val_dir_loss
            if val_dir_wmcc is not None:
                record["dir_wmcc"] = val_dir_wmcc
            if cumret24_warmup_epochs > 0 or cumret24_weight > 0:
                record["cumret24_weight"] = cumret24_weight_used
            append_log(log_path, record)
        if save_every_epoch:
            epoch_path = output_path.with_name(f"{output_path.stem}_epoch{epoch}{output_path.suffix}")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": cfg,
                },
                epoch_path,
            )

        if early_metric == "auto":
            if pinball_weight > 0:
                early_score = val_loss
            elif dir_enabled:
                early_score = val_dir_loss
            else:
                early_score = val_loss
        elif early_metric == "direction":
            early_score = val_dir_loss
        elif early_metric == "direction_mcc":
            if val_dir_wmcc is not None and np.isfinite(val_dir_wmcc):
                early_score = -val_dir_wmcc
            else:
                early_score = val_dir_loss
        elif early_metric == "pinball":
            early_score = val_loss
        else:
            early_score = val_loss

        if early_enabled:
            if early_score < best_val - early_min_delta:
                best_val = early_score
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= early_patience:
                    print(f"early_stop at epoch={epoch} best_epoch={best_epoch} best_val={best_val:.4f}")
                    break

        if val_loss < best_val_loss - early_min_delta:
            best_val_loss = val_loss
            best_epoch_loss = epoch
            best_state_loss = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if dir_batches > 0:
            if val_dir_wmcc is not None and np.isfinite(val_dir_wmcc):
                if val_dir_wmcc > best_dir_wmcc + 1e-9:
                    best_dir_wmcc = val_dir_wmcc
                    best_epoch_dir = epoch
                    best_state_dir = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif val_dir_loss < best_val_dir - early_min_delta:
                best_val_dir = val_dir_loss
                best_epoch_dir = epoch
                best_state_dir = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    save_last = bool(cfg["training"].get("save_last", False))
    output_path_last = cfg["training"].get("output_path_last")
    output_path_best_loss = cfg["training"].get("output_path_best_loss")
    output_path_best_dir = cfg["training"].get("output_path_best_dir")
    save_best_loss = bool(cfg["training"].get("save_best_loss", False)) or output_path_best_loss is not None
    save_best_dir = bool(cfg["training"].get("save_best_dir", False)) or output_path_best_dir is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = best_state if best_state is not None else model.state_dict()
    torch.save(
        {
            "model_state": state,
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
        },
        output_path,
    )
    if save_last:
        if output_path_last:
            last_path = Path(output_path_last)
        else:
            last_path = output_path.with_name(output_path.stem + "_last" + output_path.suffix)
        last_path.parent.mkdir(parents=True, exist_ok=True)
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state": last_state,
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
            },
            last_path,
        )
    if save_best_loss and best_state_loss is not None:
        if output_path_best_loss:
            loss_path = Path(output_path_best_loss)
        else:
            loss_path = output_path.with_name(output_path.stem + "_best_loss" + output_path.suffix)
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state_loss, "config": cfg}, loss_path)
    if save_best_dir and best_state_dir is not None:
        if output_path_best_dir:
            dir_path = Path(output_path_best_dir)
        else:
            dir_path = output_path.with_name(output_path.stem + "_best_dir" + output_path.suffix)
        dir_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state_dir, "config": cfg}, dir_path)
    if cov_guard_min is not None or cov_guard_max is not None:
        if last_val_cov is None or not np.isfinite(last_val_cov):
            msg = "coverage_guard: val_cov unavailable"
            if cov_guard_fail:
                raise RuntimeError(msg)
            print(f"warning: {msg}")
        else:
            min_ok = cov_guard_min is None or last_val_cov >= float(cov_guard_min)
            max_ok = cov_guard_max is None or last_val_cov <= float(cov_guard_max)
            if not (min_ok and max_ok):
                msg = f"coverage_guard_failed: val_cov80={last_val_cov:.4f} not in [{cov_guard_min},{cov_guard_max}]"
                if cov_guard_fail:
                    raise RuntimeError(msg)
                print(f"warning: {msg}")


if __name__ == "__main__":
    main()
