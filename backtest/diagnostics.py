from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def width_stats_per_horizon(
    w: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, np.ndarray]:
    valid = mask > 0
    w_masked = np.where(valid, w, np.nan)
    median = np.nanquantile(w_masked, 0.5, axis=0)
    p95 = np.nanquantile(w_masked, 0.95, axis=0)
    p99 = np.nanquantile(w_masked, 0.99, axis=0)
    zero_rate = np.nanmean((w_masked <= eps), axis=0)
    bimod_ratio = p95 / np.maximum(median, eps)
    return {
        "median": median,
        "p95": p95,
        "p99": p99,
        "zero_rate": zero_rate,
        "bimod_ratio": bimod_ratio,
    }


def band_summary(values: np.ndarray, bands: Tuple[Tuple[int, int], ...]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for start, end in bands:
        band = values[start - 1 : end]
        summary[f"{start}_{end}"] = float(np.nanmean(band))
    return summary


def mean_offdiag_corr(x: np.ndarray) -> float:
    if x.ndim == 3:
        x = x[..., 0]
    H = x.shape[1]
    corr_sum = 0.0
    count = 0
    for i in range(H):
        for j in range(i + 1, H):
            a = x[:, i]
            b = x[:, j]
            if np.std(a) == 0.0 or np.std(b) == 0.0:
                continue
            corr = np.corrcoef(a, b)[0, 1]
            corr_sum += corr
            count += 1
    return float(corr_sum / max(count, 1))


def mean_offdiag_cosine(x: np.ndarray, eps: float = 1e-8) -> float:
    # x: [B, H, D] or [H, D]
    if x.ndim == 2:
        x = x[None, ...]
    B, H, D = x.shape
    sims = []
    for b in range(B):
        vec = x[b]
        norm = np.linalg.norm(vec, axis=1, keepdims=True) + eps
        vec = vec / norm
        sim = vec @ vec.T
        sims.append(sim[np.triu_indices(H, k=1)])
    if not sims:
        return 0.0
    return float(np.mean(np.concatenate(sims, axis=0)))


def attention_similarity(attn: np.ndarray, eps: float = 1e-8) -> float:
    # attn: [B, heads, H, N]
    if attn.ndim != 4:
        raise ValueError("attn must be [B, heads, H, N]")
    B, heads, H, N = attn.shape
    sims = []
    for b in range(B):
        vec = attn[b].reshape(heads, H, N)
        vec = vec.mean(axis=0)  # [H, N]
        norm = np.linalg.norm(vec, axis=1, keepdims=True) + eps
        vec = vec / norm
        sim = vec @ vec.T
        sims.append(sim[np.triu_indices(H, k=1)])
    if not sims:
        return 0.0
    return float(np.mean(np.concatenate(sims, axis=0)))
