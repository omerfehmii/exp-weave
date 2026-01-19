from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GateConfig:
    entropy_floor: Optional[float] = None
    temperature: float = 1.0


class GatedFusion(nn.Module):
    def __init__(self, d_model: int, hidden: int, config: Optional[GateConfig] = None) -> None:
        super().__init__()
        self.config = config or GateConfig()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, mem_fine: torch.Tensor, mem_coarse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.cat([mem_fine.mean(dim=1), mem_coarse.mean(dim=1)], dim=-1)
        logits = self.mlp(pooled) / self.config.temperature
        weights = torch.softmax(logits, dim=-1)
        if self.config.entropy_floor is not None:
            weights = self._apply_entropy_floor(weights, self.config.entropy_floor)
        mem = torch.cat([weights[:, :1, None] * mem_fine, weights[:, 1:, None] * mem_coarse], dim=1)
        return mem, weights

    @staticmethod
    def _apply_entropy_floor(weights: torch.Tensor, entropy_floor: float) -> torch.Tensor:
        eps = 1e-8
        weights = torch.clamp(weights, eps, 1.0)
        ent = -torch.sum(weights * torch.log(weights), dim=-1, keepdim=True)
        max_ent = math.log(weights.shape[-1])
        denom = torch.clamp(max_ent - ent, min=1e-6)
        blend = (entropy_floor - ent) / denom
        blend = torch.clamp(blend, 0.0, 1.0)
        uniform = torch.full_like(weights, 1.0 / weights.shape[-1])
        return (1 - blend) * weights + blend * uniform
