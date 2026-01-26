from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

REGIME_TREND = 0
REGIME_RANGE_MR = 1
REGIME_CHOP_VOL = 2
REGIME_JUMPY = 3
REGIME_NAMES = ["TREND", "RANGE_MR", "CHOP_VOL", "JUMPY"]


@dataclass
class RegimeThresholds:
    trend: float
    mr: float
    vol: float
    chop: float
    jump: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "trend": float(self.trend),
            "mr": float(self.mr),
            "vol": float(self.vol),
            "chop": float(self.chop),
            "jump": float(self.jump),
        }


def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[None, :]
    return arr


def _to_2d_torch(arr: torch.Tensor) -> torch.Tensor:
    if arr.ndim == 1:
        return arr.unsqueeze(0)
    return arr


def compute_regime_features_np(
    y_past: np.ndarray,
    mask: np.ndarray | None = None,
    window: int = 240,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    y = np.asarray(y_past, dtype=np.float32)
    if y.ndim == 3:
        y = y[..., 0]
    y = _to_2d(y)
    if mask is None:
        m = np.ones_like(y, dtype=np.float32)
    else:
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        m = _to_2d(m)
    if window > 0 and y.shape[1] > window:
        y = y[:, -window:]
        m = m[:, -window:]
    if y.shape[1] < 2:
        zeros = np.zeros((y.shape[0], 1), dtype=np.float32)
        return {
            "trend_t": zeros[:, 0],
            "autocorr": zeros[:, 0],
            "vol": zeros[:, 0],
            "chop": zeros[:, 0],
            "jump": zeros[:, 0],
        }
    dy = y[:, 1:] - y[:, :-1]
    m_dy = m[:, 1:] * m[:, :-1]
    count = np.maximum(np.sum(m_dy, axis=1, keepdims=True), 1.0)
    mean = np.sum(dy * m_dy, axis=1, keepdims=True) / count
    var = np.sum((dy - mean) ** 2 * m_dy, axis=1, keepdims=True) / count
    vol = np.sqrt(np.maximum(var, 0.0))
    trend_t = mean / (vol + eps) * np.sqrt(count)

    if dy.shape[1] >= 2:
        r1 = dy[:, 1:]
        r0 = dy[:, :-1]
        m_pair = m_dy[:, 1:] * m_dy[:, :-1]
        count_p = np.maximum(np.sum(m_pair, axis=1, keepdims=True), 1.0)
        mean1 = np.sum(r1 * m_pair, axis=1, keepdims=True) / count_p
        mean0 = np.sum(r0 * m_pair, axis=1, keepdims=True) / count_p
        cov = np.sum((r1 - mean1) * (r0 - mean0) * m_pair, axis=1, keepdims=True) / count_p
        var1 = np.sum((r1 - mean1) ** 2 * m_pair, axis=1, keepdims=True) / count_p
        var0 = np.sum((r0 - mean0) ** 2 * m_pair, axis=1, keepdims=True) / count_p
        autocorr = cov / (np.sqrt(var1 * var0) + eps)
    else:
        autocorr = np.zeros_like(mean)

    chop = vol / (np.abs(mean) + eps)
    with np.errstate(invalid="ignore"):
        max_abs = np.max(np.where(m_dy > 0, np.abs(dy), -np.inf), axis=1, keepdims=True)
    max_abs = np.where(np.isfinite(max_abs), max_abs, 0.0)
    jump = max_abs / (vol + eps)

    return {
        "trend_t": trend_t[:, 0],
        "autocorr": autocorr[:, 0],
        "vol": vol[:, 0],
        "chop": chop[:, 0],
        "jump": jump[:, 0],
    }


def compute_regime_features_torch(
    y_past: torch.Tensor,
    mask: torch.Tensor | None = None,
    window: int = 240,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    y = y_past
    if y.ndim == 3:
        y = y[..., 0]
    y = _to_2d_torch(y)
    if mask is None:
        m = torch.ones_like(y)
    else:
        m = mask
        if m.ndim == 3:
            m = m[..., 0]
        m = _to_2d_torch(m)
    if window > 0 and y.shape[1] > window:
        y = y[:, -window:]
        m = m[:, -window:]
    if y.shape[1] < 2:
        zeros = torch.zeros((y.shape[0], 1), device=y.device, dtype=y.dtype)
        return {
            "trend_t": zeros[:, 0],
            "autocorr": zeros[:, 0],
            "vol": zeros[:, 0],
            "chop": zeros[:, 0],
            "jump": zeros[:, 0],
        }
    dy = y[:, 1:] - y[:, :-1]
    m_dy = m[:, 1:] * m[:, :-1]
    count = torch.clamp(m_dy.sum(dim=1, keepdim=True), min=1.0)
    mean = (dy * m_dy).sum(dim=1, keepdim=True) / count
    var = ((dy - mean) ** 2 * m_dy).sum(dim=1, keepdim=True) / count
    vol = torch.sqrt(torch.clamp(var, min=0.0))
    trend_t = mean / (vol + eps) * torch.sqrt(count)

    if dy.shape[1] >= 2:
        r1 = dy[:, 1:]
        r0 = dy[:, :-1]
        m_pair = m_dy[:, 1:] * m_dy[:, :-1]
        count_p = torch.clamp(m_pair.sum(dim=1, keepdim=True), min=1.0)
        mean1 = (r1 * m_pair).sum(dim=1, keepdim=True) / count_p
        mean0 = (r0 * m_pair).sum(dim=1, keepdim=True) / count_p
        cov = ((r1 - mean1) * (r0 - mean0) * m_pair).sum(dim=1, keepdim=True) / count_p
        var1 = ((r1 - mean1) ** 2 * m_pair).sum(dim=1, keepdim=True) / count_p
        var0 = ((r0 - mean0) ** 2 * m_pair).sum(dim=1, keepdim=True) / count_p
        autocorr = cov / (torch.sqrt(var1 * var0) + eps)
    else:
        autocorr = torch.zeros_like(mean)

    chop = vol / (torch.abs(mean) + eps)
    dy_abs = torch.abs(dy)
    dy_abs = dy_abs.masked_fill(m_dy <= 0, float("-inf"))
    max_abs = dy_abs.max(dim=1, keepdim=True).values
    max_abs = torch.where(torch.isfinite(max_abs), max_abs, torch.zeros_like(max_abs))
    jump = max_abs / (vol + eps)

    return {
        "trend_t": trend_t[:, 0],
        "autocorr": autocorr[:, 0],
        "vol": vol[:, 0],
        "chop": chop[:, 0],
        "jump": jump[:, 0],
    }


def make_gate_features(
    y_past: torch.Tensor,
    mask: torch.Tensor | None,
    window: int = 240,
    eps: float = 1e-6,
) -> torch.Tensor:
    feats = compute_regime_features_torch(y_past, mask, window=window, eps=eps)
    trend = torch.tanh(feats["trend_t"] / 5.0)
    autocorr = torch.clamp(feats["autocorr"], -1.0, 1.0)
    vol = torch.log1p(torch.clamp(feats["vol"], min=0.0))
    chop = torch.log1p(torch.clamp(feats["chop"], min=0.0))
    return torch.stack([trend, autocorr, vol, chop], dim=-1)


def fit_regime_thresholds(
    features: Dict[str, np.ndarray],
    trend_q: float = 0.7,
    mr_q: float = 0.3,
    vol_q: float = 0.7,
    chop_q: float = 0.7,
    jump_q: float = 0.9,
) -> RegimeThresholds:
    trend_abs = np.abs(features["trend_t"])
    trend = float(np.nanquantile(trend_abs, trend_q)) if trend_abs.size else 0.0
    mr = float(np.nanquantile(features["autocorr"], mr_q)) if features["autocorr"].size else 0.0
    vol = float(np.nanquantile(features["vol"], vol_q)) if features["vol"].size else 0.0
    chop = float(np.nanquantile(features["chop"], chop_q)) if features["chop"].size else 0.0
    jump = float(np.nanquantile(features["jump"], jump_q)) if features["jump"].size else 0.0
    return RegimeThresholds(trend=trend, mr=mr, vol=vol, chop=chop, jump=jump)


def label_regimes(features: Dict[str, np.ndarray], thresholds: RegimeThresholds) -> np.ndarray:
    trend = features["trend_t"]
    autocorr = features["autocorr"]
    vol = features["vol"]
    chop = features["chop"]
    jump = features["jump"]
    n = trend.shape[0]
    labels = np.full(n, REGIME_CHOP_VOL, dtype=np.int64)
    jumpy = jump >= thresholds.jump
    trend_mask = (~jumpy) & (np.abs(trend) >= thresholds.trend) & (chop <= thresholds.chop)
    range_mask = (~jumpy) & (~trend_mask) & (autocorr <= thresholds.mr) & (np.abs(trend) < thresholds.trend)
    labels[jumpy] = REGIME_JUMPY
    labels[trend_mask] = REGIME_TREND
    labels[range_mask] = REGIME_RANGE_MR
    return labels


def regime_names(labels: Iterable[int]) -> List[str]:
    return [REGIME_NAMES[int(x)] if 0 <= int(x) < len(REGIME_NAMES) else "UNKNOWN" for x in labels]

