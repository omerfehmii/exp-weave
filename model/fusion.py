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
    extra_dim: int = 0
    fixed_weight: Optional[float] = None
    logit_clip: Optional[float] = None


class GatedFusion(nn.Module):
    def __init__(self, d_model: int, hidden: int, config: Optional[GateConfig] = None) -> None:
        super().__init__()
        self.config = config or GateConfig()
        in_dim = d_model * 2 + int(self.config.extra_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        mem_fine: torch.Tensor,
        mem_coarse: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = torch.cat([mem_fine.mean(dim=1), mem_coarse.mean(dim=1)], dim=-1)
        if self.config.extra_dim:
            if extra is None:
                extra = torch.zeros((pooled.shape[0], self.config.extra_dim), device=pooled.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, extra], dim=-1)
        logits = self.mlp(pooled) / self.config.temperature
        if self.config.logit_clip is not None:
            clip = float(self.config.logit_clip)
            logits = torch.clamp(logits, -clip, clip)
        if self.config.fixed_weight is not None:
            w = float(self.config.fixed_weight)
            w = max(0.0, min(1.0, w))
            weights = torch.tensor([1.0 - w, w], device=logits.device, dtype=logits.dtype).repeat(logits.shape[0], 1)
        else:
            weights = torch.softmax(logits, dim=-1)
        if self.config.entropy_floor is not None:
            weights = self._apply_entropy_floor(weights, self.config.entropy_floor)
        mem = torch.cat([weights[:, :1, None] * mem_fine, weights[:, 1:, None] * mem_coarse], dim=1)
        if return_logits:
            return mem, weights, logits
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
