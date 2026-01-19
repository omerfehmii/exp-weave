from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import compute_delta_t


@dataclass
class MissingnessConfig:
    pattern: str = "none"  # none, random, burst
    rate: float = 0.0
    burst_prob: float = 0.0
    burst_len: int = 24
    seed: int = 7


def apply_missingness_window(
    y_past: np.ndarray,
    mask: np.ndarray,
    cfg: MissingnessConfig,
    key: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.pattern == "none":
        return y_past, mask
    rng = np.random.default_rng(cfg.seed + int(key))
    mask_mod = mask.copy()
    if cfg.pattern == "random":
        drop = rng.random(mask_mod.shape) < cfg.rate
        drop &= mask_mod > 0
        mask_mod[drop] = 0.0
    elif cfg.pattern == "burst":
        if rng.random() < cfg.burst_prob:
            length = mask_mod.shape[0]
            burst_len = min(cfg.burst_len, length)
            start = int(rng.integers(0, length - burst_len + 1))
            mask_mod[start : start + burst_len] = 0.0
    else:
        raise ValueError(f"Unknown missingness pattern: {cfg.pattern}")
    if np.all(mask_mod <= 0):
        mask_mod[0] = 1.0
    y_mod = y_past.copy()
    y_mod[mask_mod <= 0] = 0.0
    return y_mod, mask_mod


class MissingnessDataset(Dataset):
    def __init__(self, base: Dataset, cfg: MissingnessConfig) -> None:
        self.base = base
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor]:
        batch, target = self.base[idx]
        y_past = batch["y_past"].numpy()
        mask = batch["mask"].numpy()
        y_mod, mask_mod = apply_missingness_window(y_past, mask, self.cfg, idx)
        delta_mod = compute_delta_t(mask_mod)
        batch["y_past"] = torch.from_numpy(y_mod.astype(np.float32))
        batch["mask"] = torch.from_numpy(mask_mod.astype(np.float32))
        batch["delta_t"] = torch.from_numpy(delta_mod.astype(np.float32))
        return batch, target
