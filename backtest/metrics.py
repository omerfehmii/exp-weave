from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, quantiles: Iterable[float]) -> float:
    y = _flatten(y_true)
    q = q_pred.reshape(-1, q_pred.shape[-1])
    losses = []
    for i, tau in enumerate(quantiles):
        diff = y - q[:, i]
        losses.append(np.maximum(tau * diff, (tau - 1) * diff))
    return float(np.mean(np.stack(losses, axis=1)))


def coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    within = (y_true >= q_low) & (y_true <= q_high)
    return float(np.mean(within))


def interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.mean(q_high - q_low))


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    if y_true.ndim == 3:
        y_true = y_true[..., 0]
    if y_pred.ndim == 3:
        y_pred = y_pred[..., 0]
    mae_h = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse_h = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return {"mae": mae_h, "rmse": rmse_h}


def horizon_collapse(y_pred: np.ndarray) -> float:
    if y_pred.ndim == 3:
        y_pred = y_pred[..., 0]
    H = y_pred.shape[1]
    corr_sum = 0.0
    count = 0
    for i in range(H):
        for j in range(i + 1, H):
            a = y_pred[:, i]
            b = y_pred[:, j]
            if np.std(a) == 0.0 or np.std(b) == 0.0:
                continue
            corr = np.corrcoef(a, b)[0, 1]
            corr_sum += corr
            count += 1
    return float(corr_sum / max(count, 1))


def quantile_crossing_rate(q_pred: np.ndarray) -> float:
    if q_pred.shape[-1] < 2:
        return 0.0
    crossings = 0
    total = 0
    for i in range(1, q_pred.shape[-1]):
        crossings += np.sum(q_pred[..., i] < q_pred[..., i - 1])
        total += np.prod(q_pred.shape[:-1])
    return float(crossings / max(total, 1))


def gate_entropy(weights: np.ndarray) -> float:
    weights = np.clip(weights, 1e-8, 1.0)
    ent = -np.sum(weights * np.log(weights), axis=-1)
    return float(np.mean(ent))
