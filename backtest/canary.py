from __future__ import annotations

from typing import Optional

import numpy as np


def feature_shift_canary(x_future_feats: np.ndarray, shift: int = 1) -> np.ndarray:
    if x_future_feats.ndim < 2:
        raise ValueError("x_future_feats must be at least 2D (T, F).")
    shifted = np.roll(x_future_feats, shift=shift, axis=-2)
    if shift > 0:
        shifted[..., :shift, :] = x_future_feats[..., :1, :]
    elif shift < 0:
        shifted[..., shift:, :] = x_future_feats[..., -1:, :]
    return shifted


def mask_randomization_canary(mask: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    flat = mask.reshape(mask.shape[0], -1)
    perm = rng.permutation(flat.shape[0])
    shuffled = flat[perm]
    return shuffled.reshape(mask.shape)


def id_shuffle_canary(series_ids: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    return rng.permutation(series_ids)
