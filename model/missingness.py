from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeltaTConfig:
    mode: str = "MLP_LOG1P"  # or BUCKET
    n_buckets: int = 16
    max_delta: float = 48.0


class DeltaTEncoder(nn.Module):
    def __init__(self, dim: int, config: Optional[DeltaTConfig] = None) -> None:
        super().__init__()
        self.config = config or DeltaTConfig()
        if self.config.mode == "MLP_LOG1P":
            self.proj = nn.Linear(dim, dim)
        elif self.config.mode == "BUCKET":
            self.embed = nn.Embedding(self.config.n_buckets, dim)
        else:
            raise ValueError(f"Unknown delta_t mode: {self.config.mode}")

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        if self.config.mode == "MLP_LOG1P":
            return self.proj(torch.log1p(delta_t))
        bucket = torch.clamp(delta_t, max=self.config.max_delta)
        bucket = (bucket / self.config.max_delta) * (self.config.n_buckets - 1)
        bucket = bucket.long()
        return self.embed(bucket)


def make_token_key_padding_mask(token_mask: torch.Tensor) -> torch.Tensor:
    return token_mask == 0
