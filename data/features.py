from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import numpy as np


@dataclass
class CalendarFeatureConfig:
    include_hour: bool = True
    include_dow: bool = True
    include_doy: bool = True


def _to_datetime_index(timestamps: Iterable) -> np.ndarray:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas optional
        raise ImportError("pandas is required for calendar features") from exc
    return pd.to_datetime(np.asarray(timestamps))


def _cyclical_features(values: np.ndarray, period: int) -> np.ndarray:
    angles = 2 * math.pi * values.astype(np.float32) / float(period)
    return np.stack([np.sin(angles), np.cos(angles)], axis=-1)


def make_calendar_features(
    timestamps: Iterable,
    config: Optional[CalendarFeatureConfig] = None,
) -> np.ndarray:
    cfg = config or CalendarFeatureConfig()
    ts = _to_datetime_index(timestamps)
    feats = []
    if cfg.include_hour:
        feats.append(_cyclical_features(ts.hour.values, 24))
    if cfg.include_dow:
        feats.append(_cyclical_features(ts.dayofweek.values, 7))
    if cfg.include_doy:
        feats.append(_cyclical_features(ts.dayofyear.values - 1, 365))
    if not feats:
        return np.zeros((len(ts), 0), dtype=np.float32)
    return np.concatenate(feats, axis=-1).astype(np.float32)


def make_observation_mask(y: np.ndarray) -> np.ndarray:
    if np.isnan(y).any():
        return (~np.isnan(y)).astype(np.float32)
    return np.ones_like(y, dtype=np.float32)


def compute_delta_t(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)
    if mask.ndim == 1:
        mask = mask[:, None]
    t_len, dim = mask.shape
    delta = np.zeros((t_len, dim), dtype=np.float32)
    last_seen = np.zeros((dim,), dtype=np.float32)
    for t in range(t_len):
        last_seen = np.where(mask[t] > 0.0, 0.0, last_seen + 1.0)
        delta[t] = last_seen
    return delta


def log1p_delta_t(delta_t: np.ndarray) -> np.ndarray:
    return np.log1p(delta_t.astype(np.float32))


def compute_direction_features(y: np.ndarray, window: int = 24) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    y = np.nan_to_num(y, nan=0.0)
    T, D = y.shape
    dy = np.diff(y, axis=0, prepend=y[:1])
    feats = []
    idx = np.arange(T, dtype=np.int64)
    start = np.maximum(0, idx - window + 1)
    for d in range(D):
        x = dy[:, d]
        csum = np.cumsum(x, dtype=np.float64)
        csum2 = np.cumsum(x * x, dtype=np.float64)
        csum = np.concatenate([[0.0], csum])
        csum2 = np.concatenate([[0.0], csum2])
        sum_ = csum[idx + 1] - csum[start]
        sum2 = csum2[idx + 1] - csum2[start]
        n = (idx - start + 1).astype(np.float64)
        mean = sum_ / n
        var = sum2 / n - mean * mean
        std = np.sqrt(np.maximum(var, 0.0))
        feats.append(np.stack([x, mean.astype(np.float32), std.astype(np.float32)], axis=-1))
    return np.concatenate(feats, axis=-1).astype(np.float32)


def append_direction_features(
    series_list: list,
    window: int = 24,
    split_end: int | None = None,
) -> None:
    for series in series_list:
        feats = compute_direction_features(series.y, window=window)
        if split_end is not None and series.x_past_feats is not None:
            feats = feats.copy()
            fill = feats[:split_end].mean(axis=0, keepdims=True)
            feats[split_end:] = fill
        if series.x_past_feats is None:
            series.x_past_feats = feats
        else:
            series.x_past_feats = np.concatenate([series.x_past_feats, feats], axis=-1)
