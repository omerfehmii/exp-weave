from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from backtest.harness import generate_panel_origins, make_time_splits, select_indices_by_time
from data.features import append_direction_features
from data.loader import WindowedDataset, compress_series_observed, load_panel_npz
from eval import apply_scaling, build_model
from train import filter_indices_with_observed
from utils import load_config, set_seed


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _get_split_indices(
    series_list: list,
    cfg: Dict,
    split,
    indices: List[Tuple[int, int]],
    split_name: str,
    split_purge: int,
    split_embargo: int,
) -> List[Tuple[int, int]]:
    horizon = cfg["data"]["H"]
    split_idx = select_indices_by_time(
        indices,
        split,
        split_name,
        horizon=horizon,
        purge=split_purge,
        embargo=split_embargo,
    )
    split_idx = filter_indices_with_observed(
        series_list,
        split_idx,
        cfg["data"]["L"],
        cfg["data"]["H"],
        cfg["data"].get("min_past_obs", 1),
        cfg["data"].get("min_future_obs", 1),
    )
    return split_idx


def load_dir_head_arrays(
    config_path: str,
    checkpoint: str,
    split_name: str,
    split_purge: int,
    split_embargo: int,
    delta_mode: str = "origin",
) -> Dict[str, np.ndarray]:
    cfg = load_config(config_path)
    seed = cfg.get("training", {}).get("seed", 7)
    set_seed(seed)

    series_list = load_panel_npz(cfg["data"]["path"])
    if cfg.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()

    lengths = [len(s.y) for s in series_list]
    split = make_time_splits(
        min(lengths),
        cfg["data"].get("train_frac", 0.7),
        cfg["data"].get("val_frac", 0.15),
    )

    dir_feat_cfg = cfg.get("data", {}).get("direction_features", {})
    if dir_feat_cfg.get("enabled", False):
        window = int(dir_feat_cfg.get("window", 24))
        append_direction_features(series_list, window=window, split_end=split.train_end)

    indices = generate_panel_origins(
        lengths,
        cfg["data"]["L"],
        cfg["data"]["H"],
        cfg["data"].get("step", cfg["data"]["H"]),
    )
    split_idx = _get_split_indices(
        series_list,
        cfg,
        split,
        indices,
        split_name,
        split_purge,
        split_embargo,
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

    device = torch.device(cfg["training"].get("device", "cpu"))
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    quantiles = cfg["data"]["quantiles"]
    q50_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))

    move_logits_list = []
    dir_logits_list = []
    logits3_list = []
    y_future_list = []
    y_last_list = []
    mask_list = []
    origin_mask_list = []
    q50_list = []

    with torch.no_grad():
        for batch, target in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            q_hat, extras = model(batch)
            q50_list.append(q_hat[..., q50_idx].cpu().numpy())
            if "dir_logits3" in extras:
                logits3_list.append(extras["dir_logits3"].cpu().numpy())
            else:
                if "dir_move_logits" not in extras or "dir_dir_logits" not in extras:
                    raise RuntimeError("direction head logits not found in model output.")
                move_logits_list.append(extras["dir_move_logits"].cpu().numpy())
                dir_logits_list.append(extras["dir_dir_logits"].cpu().numpy())
            y_future_list.append(target[..., 0].cpu().numpy())
            y_last_list.append(batch["y_past"][:, -1, 0].cpu().numpy())
            mask_list.append(np.isfinite(target[..., 0].cpu().numpy()).astype(np.float32))
            origin_mask_list.append(batch["mask"][:, -1, 0].cpu().numpy())

    use_three_class = len(logits3_list) > 0
    if use_three_class:
        logits3 = np.concatenate(logits3_list, axis=0)
        move_prob = None
        dir_prob = None
    else:
        move_logits = np.concatenate(move_logits_list, axis=0)
        dir_logits = np.concatenate(dir_logits_list, axis=0)
        move_prob = _sigmoid(move_logits)
        dir_prob = _sigmoid(dir_logits)

    y_future = np.concatenate(y_future_list, axis=0)
    y_last = np.concatenate(y_last_list, axis=0)
    mask = np.concatenate(mask_list, axis=0) > 0
    origin_mask = np.concatenate(origin_mask_list, axis=0) > 0
    mask = mask & origin_mask[:, None]
    q50 = np.concatenate(q50_list, axis=0)

    H = y_future.shape[1]
    if delta_mode == "step":
        ref = np.concatenate([y_last[:, None], y_future[:, :-1]], axis=1)
    else:
        ref = y_last[:, None]
    delta = y_future - ref

    if use_three_class:
        probs = _softmax(logits3)
        p_down = probs[:, :, 0]
        p_flat = probs[:, :, 1]
        p_up = probs[:, :, 2]
        move_prob = 1.0 - p_flat
        dir_prob = p_up / np.maximum(p_up + p_down, 1e-6)
    else:
        move_prob = move_prob
        dir_prob = dir_prob

    return {
        "cfg": cfg,
        "series_list": series_list,
        "split": split,
        "indices": indices,
        "y_future": y_future,
        "y_last": y_last,
        "delta": delta,
        "mask": mask,
        "move_prob": move_prob,
        "dir_prob": dir_prob,
        "q50": q50,
        "use_three_class": use_three_class,
        "H": H,
    }


def get_train_indices(
    cfg: Dict,
    series_list: list,
    split,
    indices: List[Tuple[int, int]],
    split_purge: int,
    split_embargo: int,
) -> List[Tuple[int, int]]:
    return _get_split_indices(series_list, cfg, split, indices, "train", split_purge, split_embargo)
