from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np


def _nanquantile(values: np.ndarray, q: float, axis: Optional[int] = None) -> np.ndarray:
    try:
        return np.nanquantile(values, q, axis=axis, method="higher")
    except TypeError:  # numpy<1.22
        return np.nanquantile(values, q, axis=axis, interpolation="higher")


def _safe_quantile(values: np.ndarray, q: float, axis: Optional[int] = None) -> np.ndarray:
    if axis is None:
        if np.all(np.isnan(values)):
            return np.array(0.0, dtype=np.float32)
        return _nanquantile(values, q)
    if axis != 0:
        return _nanquantile(values, q, axis=axis)
    if values.ndim == 1:
        return _safe_quantile(values, q, axis=None)
    all_nan = np.all(np.isnan(values), axis=0)
    if np.all(all_nan):
        return np.zeros(values.shape[1], dtype=np.float32)
    global_q = _safe_quantile(values, q, axis=None)
    out = np.full(values.shape[1], global_q, dtype=np.float32)
    valid = ~all_nan
    if np.any(valid):
        out[valid] = _nanquantile(values[:, valid], q, axis=0)
    return out


class CQRCalibrator:
    def __init__(self, alpha: float = 0.1, per_horizon: bool = True) -> None:
        self.alpha = alpha
        self.per_horizon = per_horizon
        self.k: Optional[np.ndarray] = None

    def fit(self, q_low: np.ndarray, q_high: np.ndarray, y: np.ndarray) -> None:
        scores = np.maximum(q_low - y, y - q_high)
        if self.per_horizon:
            self.k = _safe_quantile(scores, 1 - self.alpha, axis=0)
        else:
            self.k = _safe_quantile(scores, 1 - self.alpha)

    def apply(self, q_low: np.ndarray, q_high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.k is None:
            raise RuntimeError("CQRCalibrator is not fit.")
        return q_low - self.k, q_high + self.k


class RollingCQR:
    def __init__(self, window_size: int, alpha: float = 0.1, per_horizon: bool = True) -> None:
        self.window_size = window_size
        self.alpha = alpha
        self.per_horizon = per_horizon
        self.buffer: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, q_low: np.ndarray, q_high: np.ndarray, y: np.ndarray) -> None:
        scores = np.maximum(q_low - y, y - q_high)
        self.buffer.append(scores)

    def get_k(self) -> np.ndarray:
        if not self.buffer:
            raise RuntimeError("RollingCQR has no data.")
        scores = np.concatenate(list(self.buffer), axis=0)
        if self.per_horizon:
            return _safe_quantile(scores, 1 - self.alpha, axis=0)
        return _safe_quantile(scores, 1 - self.alpha)

    def apply(self, q_low: np.ndarray, q_high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = self.get_k()
        return q_low - k, q_high + k
