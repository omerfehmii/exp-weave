from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Tuple

import numpy as np


@dataclass
class ACIGuardrail:
    eps_width: float = 1e-6
    s_clip: Tuple[float, float] = (0.1, 5.0)


def normalized_residual(
    y: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    eps_width: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wL = np.maximum(q50 - q10, eps_width)
    wU = np.maximum(q90 - q50, eps_width)
    left = (q10 - y) / wL
    right = (y - q90) / wU
    r = np.maximum(left, right)
    return r, wL, wU


def apply_scale(
    q50: np.ndarray,
    wL: np.ndarray,
    wU: np.ndarray,
    s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lo = q50 - s * wL
    hi = q50 + s * wU
    return lo, hi


def quantile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


class ResidualBuffer:
    def __init__(
        self,
        window: int,
        min_count: int,
        shrinkage_tau: float,
        guardrail: ACIGuardrail,
    ) -> None:
        self.window = window
        self.min_count = min_count
        self.shrinkage_tau = shrinkage_tau
        self.guardrail = guardrail
        self._store: Dict[Tuple[int, Tuple[int | None, int | None]], Deque[Tuple[int, float]]] = {}

    def add(self, horizon: int, bucket: Tuple[int | None, int | None], time: int, value: float) -> None:
        key = (horizon, bucket)
        if key not in self._store:
            self._store[key] = deque()
        self._store[key].append((time, float(value)))

    def trim(self, current_time: int) -> None:
        if self.window <= 0:
            return
        cutoff = current_time - self.window
        for buf in self._store.values():
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def clear(self) -> None:
        self._store.clear()

    def _values(self, horizon: int, bucket: Tuple[int | None, int | None]) -> Tuple[np.ndarray, int]:
        key = (horizon, bucket)
        buf = self._store.get(key)
        if not buf:
            return np.array([], dtype=np.float32), 0
        vals = np.array([v for _, v in buf], dtype=np.float32)
        return vals, len(vals)

    def get_scale(
        self,
        horizon: int,
        bucket: Tuple[int | None, int | None],
        alpha: float,
        fallback_bucket: Tuple[int | None, int | None],
        global_bucket: Tuple[int | None, int | None],
    ) -> float:
        vals, n = self._values(horizon, bucket)
        if n < self.min_count:
            vals, n = self._values(horizon, fallback_bucket)
        if n < self.min_count:
            vals, n = self._values(horizon, global_bucket)
        if n == 0:
            return 1.0
        qhat = float(np.quantile(vals, 1.0 - alpha))
        s = 1.0 + qhat
        s = float(np.clip(s, self.guardrail.s_clip[0], self.guardrail.s_clip[1]))
        if self.shrinkage_tau > 0:
            global_vals, global_n = self._values(horizon, global_bucket)
            if global_n > 0:
                qhat_g = float(np.quantile(global_vals, 1.0 - alpha))
                s_g = float(np.clip(1.0 + qhat_g, self.guardrail.s_clip[0], self.guardrail.s_clip[1]))
                weight = n / (n + self.shrinkage_tau)
                s = weight * s + (1 - weight) * s_g
        return float(np.clip(s, self.guardrail.s_clip[0], self.guardrail.s_clip[1]))


def update_alpha(
    alpha: float,
    miss: float,
    alpha_target: float,
    gamma: float,
    alpha_clip: Tuple[float, float],
) -> float:
    updated = alpha + gamma * (alpha_target - miss)
    return float(np.clip(updated, alpha_clip[0], alpha_clip[1]))
