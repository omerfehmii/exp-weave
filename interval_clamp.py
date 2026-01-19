from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _half(q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(q90 - q10)


def fit_min_half_abs(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    quantile: float = 0.05,
    scale: float = 0.1,
) -> float:
    half = _half(q10, q90).reshape(-1)
    half = half[np.isfinite(half)]
    half = half[half > 0]
    if half.size == 0:
        return 0.0
    return float(np.quantile(half, quantile) * scale)


def fit_min_half_rel(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    quantile: float = 0.05,
    scale: float = 0.1,
    eps: float = 1e-6,
) -> float:
    half = _half(q10, q90)
    rel = half / (np.abs(q50) + eps)
    rel = rel.reshape(-1)
    rel = rel[np.isfinite(rel)]
    rel = rel[rel > 0]
    if rel.size == 0:
        return 0.0
    return float(np.quantile(rel, quantile) * scale)


def apply_min_half_clamp(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    min_half: float | None = None,
    min_half_rel: float | None = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10 = np.nan_to_num(q10, nan=0.0)
    q50 = np.nan_to_num(q50, nan=0.0)
    q90 = np.nan_to_num(q90, nan=0.0)
    half = _half(q10, q90)
    if min_half is not None and min_half > 0:
        half = np.maximum(half, min_half)
    if min_half_rel is not None and min_half_rel > 0:
        half = np.maximum(half, min_half_rel * (np.abs(q50) + eps))
    lo = q50 - half
    hi = q50 + half
    return lo, q50, hi


def half_stats(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    mask: np.ndarray | None = None,
    axis: int | None = None,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    half = _half(q10, q90)
    if mask is not None:
        half = np.where(mask > 0, half, np.nan)
    if axis is None:
        half_flat = half.reshape(-1)
        half_flat = half_flat[np.isfinite(half_flat)]
        if half_flat.size == 0:
            return {"p50": np.array(0.0), "p95": np.array(0.0), "zero_frac": np.array(1.0)}
        return {
            "p50": np.array(np.quantile(half_flat, 0.5)),
            "p95": np.array(np.quantile(half_flat, 0.95)),
            "zero_frac": np.array(np.mean(half_flat <= eps)),
        }
    zero_mask = np.where(np.isnan(half), np.nan, half <= eps)
    zero_frac = np.nanmean(zero_mask, axis=axis)
    p50 = np.nanquantile(half, 0.5, axis=axis)
    p95 = np.nanquantile(half, 0.95, axis=axis)
    return {"p50": p50, "p95": p95, "zero_frac": zero_frac}
