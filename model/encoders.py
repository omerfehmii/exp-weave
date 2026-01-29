from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.scale


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.fc1(x)
        x_gate, x_val = x_proj.chunk(2, dim=-1)
        return self.fc2(self.dropout(F.silu(x_gate) * x_val))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.stack((freqs.cos(), freqs.sin()), dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    cos = rope[..., 0]
    sin = rope[..., 1]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rope: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if rope and self.head_dim % 2 != 0:
            raise ValueError("RoPE requires even head dimension.")
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = rope
        self.rope = RotaryEmbedding(self.head_dim) if rope else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        B, Q, _ = query.shape
        K = key.shape[1]
        q = self.q_proj(query).view(B, Q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            rope_q = self.rope(Q, query.device)
            rope_k = self.rope(K, query.device)
            q = apply_rope(q, rope_q)
            k = apply_rope(k, rope_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask
            all_masked = mask.all(dim=1)
            if all_masked.any():
                mask = mask.clone()
                mask[all_masked, 0] = False
            mask = mask[:, None, None, :].to(dtype=scores.dtype)
            scores = scores.masked_fill(mask > 0.0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, Q, self.d_model)
        out = self.out_proj(out)
        if return_attn:
            return out, attn
        return out


@dataclass
class EncoderConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_hidden: int = 1024
    dropout: float = 0.1
    use_rope: bool = True


class TransformerEncoderBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, rope=cfg.use_rope)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x
