from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .missingness import DeltaTConfig, DeltaTEncoder


@dataclass
class PatchScale:
    P: int
    S: int
    name: str


class PatchEmbed(nn.Module):
    def __init__(self, input_dim: int, d_model: int, scale_name: str) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.scale_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.scale_name = scale_name

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(patches)
        return tokens + self.scale_embedding


class MultiScalePatcher(nn.Module):
    def __init__(
        self,
        scales: List[PatchScale],
        d_model: int,
        in_dim: int,
        target_dim: int,
        delta_t_dim: int,
        mask_embedding: bool = True,
        delta_t_mode: str = "MLP_LOG1P",
        summary_enabled: bool = False,
        summary_window: int = 24,
        summary_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.mask_embedding = mask_embedding
        self.delta_t_mode = delta_t_mode
        self.summary_enabled = summary_enabled
        self.summary_window = summary_window
        self.summary_dropout = summary_dropout
        if delta_t_dim > 0 and delta_t_mode != "NONE":
            dt_cfg = DeltaTConfig(mode=delta_t_mode)
            self.delta_t_encoder = DeltaTEncoder(delta_t_dim, dt_cfg)
        else:
            self.delta_t_encoder = None
        self.embedders = nn.ModuleDict()
        delta_dim = delta_t_dim if self.delta_t_encoder is not None else 0
        for scale in scales:
            patch_dim = scale.P * (in_dim + delta_dim)
            self.embedders[scale.name] = PatchEmbed(patch_dim, d_model, scale.name)
        if summary_enabled:
            self.summary_proj = nn.Sequential(
                nn.Linear(2 * target_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.summary_scale = nn.ParameterDict(
                {scale.name: nn.Parameter(torch.zeros(1, 1, d_model)) for scale in scales}
            )

    def forward(
        self,
        y_past: torch.Tensor,
        x_past_feats: torch.Tensor,
        mask: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        B, L, D = y_past.shape
        if x_past_feats.ndim == 2:
            x_past_feats = x_past_feats.unsqueeze(0)
        parts = [y_past, x_past_feats]
        if self.mask_embedding:
            parts.append(mask)
        per_step = torch.cat(parts, dim=-1)
        if self.delta_t_encoder is not None:
            delta_enc = self.delta_t_encoder(delta_t)
            per_step = torch.cat([per_step, delta_enc], dim=-1)
        tokens: Dict[str, torch.Tensor] = {}
        token_masks: Dict[str, torch.Tensor] = {}
        for scale in self.scales:
            P, S = scale.P, scale.S
            patches = per_step.unfold(dimension=1, size=P, step=S)
            patches = patches.contiguous().view(B, -1, P * per_step.shape[-1])
            mask_patch = mask.unfold(dimension=1, size=P, step=S)
            mask_valid = (mask_patch.sum(dim=(-1, -2)) > 0).float()
            all_invalid = mask_valid.sum(dim=1) == 0
            if all_invalid.any():
                mask_valid = mask_valid.clone()
                mask_valid[all_invalid, 0] = 1.0
            tokens[scale.name] = self.embedders[scale.name](patches)
            token_masks[scale.name] = mask_valid
        if self.summary_enabled:
            dy = y_past[:, 1:, :] - y_past[:, :-1, :]
            if dy.shape[1] == 0:
                dy = torch.zeros_like(y_past[:, :1, :])
            window = self.summary_window if self.summary_window > 0 else dy.shape[1]
            dy_win = dy[:, -window:, :]
            mean = dy_win.mean(dim=1)
            std = dy_win.std(dim=1, unbiased=False)
            summary_feats = torch.cat([mean, std], dim=-1)
            summary = self.summary_proj(summary_feats).unsqueeze(1)
            summary_mask = torch.ones((B, 1), dtype=mask.dtype, device=mask.device)
            if self.training and self.summary_dropout > 0:
                drop = torch.rand((B, 1), device=mask.device) < self.summary_dropout
                summary_mask = summary_mask.masked_fill(drop, 0.0)
                summary = summary.masked_fill(drop.unsqueeze(-1), 0.0)
            for scale in self.scales:
                summary_tok = summary + self.summary_scale[scale.name]
                tokens[scale.name] = torch.cat([summary_tok, tokens[scale.name]], dim=1)
                token_masks[scale.name] = torch.cat([summary_mask, token_masks[scale.name]], dim=1)
        return tokens, token_masks
