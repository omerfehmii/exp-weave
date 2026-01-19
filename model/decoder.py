from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .cats_masking import CATSConfig, apply_cats_masking
from .encoders import MultiHeadAttention, RMSNorm, SwiGLU


@dataclass
class DecoderConfig:
    d_model: int = 256
    n_heads: int = 8
    mlp_hidden: int = 1024
    dropout: float = 0.1
    mode: str = "CA_ONLY"  # or HYBRID_QSA


class DecoderBlock(nn.Module):
    def __init__(self, cfg: DecoderConfig, cats_cfg: Optional[CATSConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.cats_cfg = cats_cfg or CATSConfig()
        self.norm_q = RMSNorm(cfg.d_model)
        self.norm_q2 = RMSNorm(cfg.d_model)
        self.norm_ff = RMSNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, rope=False)
        self.self_attn = (
            MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, rope=False)
            if cfg.mode == "HYBRID_QSA"
            else None
        )
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        q: torch.Tensor,
        mem: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        mem_key_padding_mask: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None,
        mem_weights: Optional[Tuple[float, float]] = None,
        cats_enabled_override: Optional[bool] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor | list[torch.Tensor]]]:
        if self.self_attn is not None:
            self_attn_out = self.self_attn(self.norm_q(q), self.norm_q(q), self.norm_q(q))
            q = q + self.dropout(self_attn_out)
        attn_weights = None
        if isinstance(mem, tuple):
            mem_list = list(mem)
            mask_list = list(mem_key_padding_mask) if isinstance(mem_key_padding_mask, tuple) else [None, None]
            weights = mem_weights or (0.5, 0.5)
            out_list = []
            attn_list = []
            for mem_i, mask_i, w in zip(mem_list, mask_list, weights):
                if return_attn:
                    out_i, attn_i = self.cross_attn(
                        self.norm_q2(q),
                        mem_i,
                        mem_i,
                        key_padding_mask=mask_i,
                        return_attn=True,
                    )
                    attn_list.append(attn_i)
                else:
                    out_i = self.cross_attn(self.norm_q2(q), mem_i, mem_i, key_padding_mask=mask_i)
                out_list.append(out_i * w)
            attn_out = sum(out_list)
            if return_attn:
                attn_weights = attn_list
        else:
            if return_attn:
                attn_out, attn_weights = self.cross_attn(
                    self.norm_q2(q),
                    mem,
                    mem,
                    key_padding_mask=mem_key_padding_mask,
                    return_attn=True,
                )
            else:
                attn_out = self.cross_attn(self.norm_q2(q), mem, mem, key_padding_mask=mem_key_padding_mask)
        cats_cfg = self.cats_cfg
        if cats_enabled_override is not None:
            cats_cfg = CATSConfig(
                enabled=cats_enabled_override,
                p_min=self.cats_cfg.p_min,
                p_max=self.cats_cfg.p_max,
                scaling=self.cats_cfg.scaling,
            )
        attn_out, cats_mask = apply_cats_masking(attn_out, cats_cfg, self.training)
        q = q + self.dropout(attn_out)
        q = q + self.dropout(self.mlp(self.norm_ff(q)))
        return q, cats_mask, attn_weights


class ForecastDecoder(nn.Module):
    def __init__(
        self,
        H: int,
        d_model: int,
        future_dim: int,
        cfg: DecoderConfig,
        cats_cfg: Optional[CATSConfig] = None,
    ) -> None:
        super().__init__()
        self.H = H
        self.q_learned = nn.Parameter(torch.zeros(1, H, d_model))
        self.horizon_emb = nn.Embedding(H, d_model)
        self.future_proj = nn.Linear(future_dim, d_model) if future_dim > 0 else None
        self.block = DecoderBlock(cfg, cats_cfg)

    def forward(
        self,
        mem: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        x_future_feats: torch.Tensor,
        mem_key_padding_mask: Optional[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]] = None,
        mem_weights: Optional[Tuple[float, float]] = None,
        cats_enabled_override: Optional[bool] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor | list[torch.Tensor]]]:
        B = mem[0].shape[0] if isinstance(mem, tuple) else mem.shape[0]
        device = mem[0].device if isinstance(mem, tuple) else mem.device
        horizon_ids = torch.arange(self.H, device=device)
        q = self.q_learned.expand(B, -1, -1) + self.horizon_emb(horizon_ids)[None, :, :]
        if self.future_proj is not None and x_future_feats.numel() > 0:
            q = q + self.future_proj(x_future_feats)
        q, cats_mask, attn = self.block(
            q,
            mem,
            mem_key_padding_mask,
            mem_weights=mem_weights,
            cats_enabled_override=cats_enabled_override,
            return_attn=return_attn,
        )
        return q, cats_mask, attn
