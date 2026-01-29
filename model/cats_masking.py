from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class CATSConfig:
    enabled: bool = True
    p_min: float = 0.1
    p_max: float = 0.7
    scaling: str = "NONE"  # or INVERTED


def apply_cats_masking(
    attn_out: torch.Tensor,
    config: Optional[CATSConfig] = None,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cfg = config or CATSConfig()
    if not cfg.enabled or not training:
        return attn_out, torch.ones(attn_out.shape[:2], device=attn_out.device)
    B, H, _ = attn_out.shape
    probs = torch.linspace(cfg.p_min, cfg.p_max, H, device=attn_out.device)
    keep = torch.rand(B, H, 1, device=attn_out.device) > probs.view(1, H, 1)
    if cfg.scaling == "INVERTED":
        scale = 1.0 / (1.0 - probs).view(1, H, 1)
        masked = attn_out * keep * scale
    else:
        masked = attn_out * keep
    return masked, keep.squeeze(-1).float()
