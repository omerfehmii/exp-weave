from __future__ import annotations

from typing import Tuple

import numpy as np


def _coverage_with_s(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    s: float,
    mask: np.ndarray,
) -> float:
    mid = q50
    half = 0.5 * np.abs(q90 - q10)
    lo = mid - s * half
    hi = mid + s * half
    inside = (y >= lo) & (y <= hi) & (mask > 0)
    denom = np.sum(mask)
    return float(np.sum(inside) / max(denom, 1.0))


def fit_s_global(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    target: float = 0.90,
    s_lo: float = 0.10,
    s_hi: float = 1.50,
    iters: int = 40,
    mask: np.ndarray | None = None,
) -> float:
    if mask is None:
        mask = np.isfinite(y).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    q10 = np.nan_to_num(q10, nan=0.0)
    q50 = np.nan_to_num(q50, nan=0.0)
    q90 = np.nan_to_num(q90, nan=0.0)

    cov_hi = _coverage_with_s(y, q10, q50, q90, s_hi, mask)
    if cov_hi < target:
        return s_hi
    cov_lo = _coverage_with_s(y, q10, q50, q90, s_lo, mask)
    if cov_lo >= target:
        return s_lo

    lo, hi = s_lo, s_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cov = _coverage_with_s(y, q10, q50, q90, mid, mask)
        if cov >= target:
            hi = mid
        else:
            lo = mid
    return hi


def fit_s_per_horizon(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    target: float = 0.90,
    s_lo: float = 0.10,
    s_hi: float = 1.50,
    iters: int = 40,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    _, H = y.shape
    s = np.zeros(H, dtype=np.float32)
    for h in range(H):
        mask_h = None if mask is None else mask[:, h : h + 1]
        s[h] = fit_s_global(
            y[:, h : h + 1],
            q10[:, h : h + 1],
            q50[:, h : h + 1],
            q90[:, h : h + 1],
            target=target,
            s_lo=s_lo,
            s_hi=s_hi,
            iters=iters,
            mask=mask_h,
        )
    return s


def apply_width_scaling(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    s: np.ndarray | float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10 = np.nan_to_num(q10, nan=0.0)
    q50 = np.nan_to_num(q50, nan=0.0)
    q90 = np.nan_to_num(q90, nan=0.0)
    mid = q50
    half = 0.5 * np.abs(q90 - q10)
    if np.isscalar(s):
        lo = mid - s * half
        hi = mid + s * half
    else:
        scale = np.asarray(s, dtype=np.float32).reshape(1, -1)
        lo = mid - half * scale
        hi = mid + half * scale
    return lo, mid, hi
