from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    quantiles: List[float]


class MonotoneQuantileHead(nn.Module):
    def __init__(self, d_model: int, quantiles: Iterable[float], delta_floor: float = 0.0) -> None:
        super().__init__()
        self.quantiles = list(quantiles)
        if len(self.quantiles) < 2:
            raise ValueError("Quantiles must include at least two values.")
        self.base = nn.Linear(d_model, 1)
        self.deltas = nn.Linear(d_model, len(self.quantiles) - 1)
        self.delta_floor = float(delta_floor)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        base = self.base(h)
        deltas = F.softplus(self.deltas(h)) + self.delta_floor
        q_list = [base]
        cumsum = base
        for i in range(deltas.shape[-1]):
            cumsum = cumsum + deltas[..., i : i + 1]
            q_list.append(cumsum)
        return torch.cat(q_list, dim=-1)


class DualPathQuantileHead(nn.Module):
    def __init__(self, d_model: int, quantiles: Iterable[float], delta_floor: float = 0.0) -> None:
        super().__init__()
        self.quantiles = list(quantiles)
        if self.quantiles != [0.1, 0.5, 0.9]:
            raise ValueError("DualPathQuantileHead requires quantiles [0.1, 0.5, 0.9].")
        self.base = nn.Linear(d_model, 1)
        self.deltas = nn.Linear(d_model, 2)
        self.delta_floor = float(delta_floor)

    def forward(self, base_h: torch.Tensor, delta_h: torch.Tensor) -> torch.Tensor:
        q50 = self.base(base_h)
        deltas = F.softplus(self.deltas(delta_h)) + self.delta_floor
        q10 = q50 - deltas[..., 0:1]
        q90 = q50 + deltas[..., 1:2]
        return torch.cat([q10, q50, q90], dim=-1)


class LSQQuantileHead(nn.Module):
    def __init__(self, d_model: int, quantiles: Iterable[float], s_min: float = 0.0) -> None:
        super().__init__()
        self.quantiles = list(quantiles)
        if self.quantiles != [0.1, 0.5, 0.9]:
            raise ValueError("LSQQuantileHead currently supports quantiles [0.1, 0.5, 0.9].")
        self.mu = nn.Linear(d_model, 1)
        self.scale = nn.Linear(d_model, 1)
        self.delta = nn.Linear(d_model, 2)
        self.s_min = float(s_min)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mu = self.mu(h)
        scale = F.softplus(self.scale(h)) + self.s_min
        raw = self.delta(h)
        d_low = -F.softplus(raw[..., 0:1])
        d_high = F.softplus(raw[..., 1:2])
        q10 = mu + scale * d_low
        q50 = mu
        q90 = mu + scale * d_high
        return torch.cat([q10, q50, q90], dim=-1)


class FreeQuantileHead(nn.Module):
    def __init__(self, d_model: int, quantiles: Iterable[float]) -> None:
        super().__init__()
        self.quantiles = list(quantiles)
        if len(self.quantiles) < 2:
            raise ValueError("Quantiles must include at least two values.")
        self.proj = nn.Linear(d_model, len(self.quantiles))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)
