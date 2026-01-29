from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .cats_masking import CATSConfig
from .decoder import DecoderConfig, ForecastDecoder
from .encoders import EncoderConfig, TransformerEncoder
from .fusion import GateConfig, GatedFusion
from .head import DualPathQuantileHead, FreeQuantileHead, LSQQuantileHead, MonotoneQuantileHead
from .missingness import make_token_key_padding_mask
from .patching import MultiScalePatcher, PatchScale
from regime import make_gate_features


@dataclass
class ModelConfig:
    L: int
    H: int
    target_dim: int
    past_feat_dim: int
    future_feat_dim: int
    quantiles: List[float]
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_hidden: int = 1024
    dropout: float = 0.1
    fusion: str = "TWO_ENCODERS_GATED"  # or JOINT_TOKENS, DUAL_SUM, TWO_ENCODERS_CONCAT
    gate_entropy_floor: Optional[float] = None
    gate_temperature: float = 1.0
    gate_logit_clip: Optional[float] = None
    gate_fixed_weight: Optional[float] = None
    gate_use_regime_features: bool = False
    gate_regime_window: int = 240
    gate_regime_eps: float = 1e-6
    gate_disable_coarse: bool = False
    gate_disable_fine: bool = False
    scale_drop_coarse: float = 0.0
    scale_drop_fine: float = 0.0
    moe_enabled: bool = False
    moe_gate_hidden: int = 256
    moe_gate_temperature: float = 1.0
    moe_gate_logit_clip: Optional[float] = None
    moe_gate_use_regime_features: bool = False
    moe_gate_regime_window: int = 240
    moe_gate_regime_eps: float = 1e-6
    moe_expert_drop_trend: float = 0.0
    moe_expert_drop_mr: float = 0.0
    dual_sum_weight: str = "equal"  # or token
    decoder_mode: str = "CA_ONLY"
    cats_enabled: bool = True
    cats_p_min: float = 0.1
    cats_p_max: float = 0.7
    cats_scaling: str = "NONE"
    dual_path: bool = False
    dual_path_uncertainty_cats: bool = False
    head_type: str = "MONO"  # MONO, DUAL_PATH, LSQ, FREE
    head_delta_floor: float = 0.0
    head_lsq_s_min: float = 0.0
    head_detach: bool = False
    mask_embedding: bool = True
    delta_t_mode: str = "MLP_LOG1P"  # MLP_LOG1P, BUCKET, NONE
    attn_logit_bias: str = "HARD_NEG_INF"  # HARD_NEG_INF or NONE
    summary_enabled: bool = False
    summary_window: int = 24
    summary_dropout: float = 0.0
    dir_head_enabled: bool = False
    dir_head_type: str = "hierarchical"  # hierarchical or three_class
    dir_head_detach: bool = False
    dir_head_dropout: float = 0.0
    rank_head_enabled: bool = False
    rank_head_detach: bool = False
    rank_head_dropout: float = 0.0
    cumret24_head: bool = False
    use_series_id: bool = False
    series_id_vocab: Optional[int] = None


class MultiScaleForecastModel(nn.Module):
    def __init__(self, cfg: ModelConfig, scales: List[PatchScale]) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.target_dim + cfg.past_feat_dim
        if cfg.mask_embedding:
            in_dim += cfg.target_dim
        delta_t_dim = cfg.target_dim if cfg.delta_t_mode != "NONE" else 0
        self.patcher = MultiScalePatcher(
            scales,
            cfg.d_model,
            in_dim,
            cfg.target_dim,
            delta_t_dim,
            mask_embedding=cfg.mask_embedding,
            delta_t_mode=cfg.delta_t_mode,
            summary_enabled=cfg.summary_enabled,
            summary_window=cfg.summary_window,
            summary_dropout=cfg.summary_dropout,
        )
        enc_cfg = EncoderConfig(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mlp_hidden=cfg.mlp_hidden,
            dropout=cfg.dropout,
            use_rope=True,
        )
        self.fusion = cfg.fusion
        if cfg.fusion == "JOINT_TOKENS":
            self.encoder = TransformerEncoder(enc_cfg)
        elif cfg.fusion in {"TWO_ENCODERS_GATED", "DUAL_SUM", "TWO_ENCODERS_CONCAT"}:
            self.encoder_fine = TransformerEncoder(enc_cfg)
            self.encoder_coarse = TransformerEncoder(enc_cfg)
            if cfg.fusion == "TWO_ENCODERS_GATED":
                extra_dim = 4 if cfg.gate_use_regime_features else 0
                gate_cfg = GateConfig(
                    entropy_floor=cfg.gate_entropy_floor,
                    temperature=cfg.gate_temperature,
                    extra_dim=extra_dim,
                    fixed_weight=cfg.gate_fixed_weight,
                    logit_clip=cfg.gate_logit_clip,
                )
                self.gate = GatedFusion(cfg.d_model, cfg.d_model, gate_cfg)
            else:
                self.gate = None
        else:
            raise ValueError(f"Unknown fusion mode: {cfg.fusion}")
        cats_cfg = CATSConfig(
            enabled=cfg.cats_enabled,
            p_min=cfg.cats_p_min,
            p_max=cfg.cats_p_max,
            scaling=cfg.cats_scaling,
        )
        dec_cfg = DecoderConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            mlp_hidden=cfg.mlp_hidden,
            dropout=cfg.dropout,
            mode=cfg.decoder_mode,
        )
        self.decoder = ForecastDecoder(cfg.H, cfg.d_model, cfg.future_feat_dim, dec_cfg, cats_cfg)
        def _build_head() -> nn.Module:
            if cfg.head_type == "MONO":
                return MonotoneQuantileHead(cfg.d_model, cfg.quantiles, delta_floor=cfg.head_delta_floor)
            if cfg.head_type == "DUAL_PATH":
                return DualPathQuantileHead(cfg.d_model, cfg.quantiles, delta_floor=cfg.head_delta_floor)
            if cfg.head_type == "LSQ":
                return LSQQuantileHead(cfg.d_model, cfg.quantiles, s_min=cfg.head_lsq_s_min)
            if cfg.head_type == "FREE":
                return FreeQuantileHead(cfg.d_model, cfg.quantiles)
            raise ValueError(f"Unknown head type: {cfg.head_type}")

        if cfg.moe_enabled:
            self.expert_heads = nn.ModuleList([_build_head(), _build_head()])
            gate_in_dim = cfg.d_model + (4 if cfg.moe_gate_use_regime_features else 0)
            self.moe_gate = nn.Sequential(
                nn.Linear(gate_in_dim, cfg.moe_gate_hidden),
                nn.ReLU(),
                nn.Linear(cfg.moe_gate_hidden, 2),
            )
        else:
            self.expert_heads = None
            self.moe_gate = None
            self.head = _build_head()
        self.dir_head_enabled = cfg.dir_head_enabled
        if self.dir_head_enabled:
            self.dir_dropout = nn.Dropout(cfg.dir_head_dropout) if cfg.dir_head_dropout > 0 else None
            self.dir_head_type = cfg.dir_head_type
            if cfg.dir_head_type == "three_class":
                self.dir_logits3 = nn.Linear(cfg.d_model, 3)
                self.dir_move = None
                self.dir_dir = None
            else:
                self.dir_logits3 = None
                self.dir_move = nn.Linear(cfg.d_model, 1)
                self.dir_dir = nn.Linear(cfg.d_model, 1)
        else:
            self.dir_dropout = None
            self.dir_logits3 = None
            self.dir_move = None
            self.dir_dir = None
        self.rank_head_enabled = cfg.rank_head_enabled
        if self.rank_head_enabled:
            self.rank_dropout = nn.Dropout(cfg.rank_head_dropout) if cfg.rank_head_dropout > 0 else None
            self.rank_head = nn.Linear(cfg.d_model, 1)
        else:
            self.rank_dropout = None
            self.rank_head = None
        self.cumret24_head_enabled = cfg.cumret24_head
        if self.cumret24_head_enabled:
            self.cumret24_head = nn.Linear(cfg.d_model, 1)
        else:
            self.cumret24_head = None
        if cfg.use_series_id:
            if cfg.series_id_vocab is None:
                raise ValueError("series_id_vocab required when use_series_id is True.")
            self.series_embed = nn.Embedding(cfg.series_id_vocab, cfg.d_model)
        else:
            self.series_embed = None

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_diagnostics: bool = False,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_past = batch["y_past"]
        x_past_feats = batch["x_past_feats"]
        x_future_feats = batch["x_future_feats"]
        mask = batch["mask"]
        delta_t = batch["delta_t"]
        if y_past.ndim == 2:
            y_past = y_past.unsqueeze(-1)
            mask = mask.unsqueeze(-1)
            delta_t = delta_t.unsqueeze(-1)
        tokens, token_masks = self.patcher(y_past, x_past_feats, mask, delta_t)
        if self.series_embed is not None and "series_id" in batch:
            series_vec = self.series_embed(batch["series_id"]).unsqueeze(1)
            for key in tokens:
                tokens[key] = tokens[key] + series_vec
        extras: Dict[str, torch.Tensor] = {}
        use_attn_mask = self.cfg.attn_logit_bias != "NONE"

        def _maybe_mask(mask_tensor: torch.Tensor) -> Optional[torch.Tensor]:
            return make_token_key_padding_mask(mask_tensor) if use_attn_mask else None

        mem_weights = None
        if self.fusion == "JOINT_TOKENS":
            mem = torch.cat([tokens[key] for key in tokens], dim=1)
            mem_mask = torch.cat([token_masks[key] for key in token_masks], dim=1)
            mem = self.encoder(mem, key_padding_mask=_maybe_mask(mem_mask))
        elif self.fusion in {"TWO_ENCODERS_GATED", "DUAL_SUM", "TWO_ENCODERS_CONCAT"}:
            if "fine" not in tokens or "coarse" not in tokens:
                raise ValueError("Multi-scale fusion requires scales named 'fine' and 'coarse'.")
            mem_f = self.encoder_fine(tokens["fine"], key_padding_mask=_maybe_mask(token_masks["fine"]))
            mem_c = self.encoder_coarse(tokens["coarse"], key_padding_mask=_maybe_mask(token_masks["coarse"]))
            if self.cfg.gate_disable_coarse:
                mem_c = mem_c.clone()
                mem_c[:] = 0.0
                token_masks["coarse"] = token_masks["coarse"].clone()
                token_masks["coarse"][:] = 0.0
            if self.cfg.gate_disable_fine:
                mem_f = mem_f.clone()
                mem_f[:] = 0.0
                token_masks["fine"] = token_masks["fine"].clone()
                token_masks["fine"][:] = 0.0
            if self.training and (self.cfg.scale_drop_coarse > 0 or self.cfg.scale_drop_fine > 0):
                p_coarse = max(0.0, float(self.cfg.scale_drop_coarse))
                p_fine = max(0.0, float(self.cfg.scale_drop_fine))
                p_total = min(p_coarse + p_fine, 1.0)
                if p_total > 0:
                    draw = torch.rand((mem_f.shape[0], 1), device=mem_f.device)
                    drop_coarse = draw < p_coarse
                    drop_fine = (draw >= p_coarse) & (draw < p_total)
                    if drop_coarse.any():
                        mem_c = mem_c.clone()
                        mem_c[drop_coarse.squeeze(-1)] = 0.0
                        token_masks["coarse"] = token_masks["coarse"].clone()
                        token_masks["coarse"][drop_coarse.squeeze(-1)] = 0.0
                    if drop_fine.any():
                        mem_f = mem_f.clone()
                        mem_f[drop_fine.squeeze(-1)] = 0.0
                        token_masks["fine"] = token_masks["fine"].clone()
                        token_masks["fine"][drop_fine.squeeze(-1)] = 0.0
            if self.fusion == "TWO_ENCODERS_GATED":
                gate_extra = None
                if self.cfg.gate_use_regime_features:
                    gate_extra = make_gate_features(
                        y_past,
                        mask,
                        window=self.cfg.gate_regime_window,
                        eps=self.cfg.gate_regime_eps,
                    )
                if return_diagnostics:
                    mem, weights, logits = self.gate(mem_f, mem_c, extra=gate_extra, return_logits=True)
                    extras["gate_logits"] = logits
                else:
                    mem, weights = self.gate(mem_f, mem_c, extra=gate_extra, return_logits=False)
                mem_mask = torch.cat([token_masks["fine"], token_masks["coarse"]], dim=1)
                extras["gate_weights"] = weights
            elif self.fusion == "TWO_ENCODERS_CONCAT":
                mem = torch.cat([mem_f, mem_c], dim=1)
                mem_mask = torch.cat([token_masks["fine"], token_masks["coarse"]], dim=1)
            else:
                mem = (mem_f, mem_c)
                mem_mask = (token_masks["fine"], token_masks["coarse"])
                if self.cfg.dual_sum_weight == "token":
                    n_f = token_masks["fine"].shape[1]
                    n_c = token_masks["coarse"].shape[1]
                    total = float(n_f + n_c)
                    mem_weights = (n_f / total, n_c / total)
                else:
                    mem_weights = (0.5, 0.5)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion}")

        cats_mask = None
        attn = None
        if self.cfg.dual_path and self.cfg.head_type != "DUAL_PATH":
            raise ValueError("dual_path enabled requires head_type=DUAL_PATH.")
        dual_path = self.cfg.head_type == "DUAL_PATH"
        if dual_path:
            dec_masked, cats_mask, attn = self.decoder(
                mem,
                x_future_feats,
                _maybe_mask(mem_mask)
                if not isinstance(mem_mask, tuple)
                else tuple(_maybe_mask(m) for m in mem_mask),
                mem_weights=mem_weights,
                cats_enabled_override=self.cfg.cats_enabled,
                return_attn=return_attn,
            )
            dec_uncert, _, _ = self.decoder(
                mem,
                x_future_feats,
                _maybe_mask(mem_mask)
                if not isinstance(mem_mask, tuple)
                else tuple(_maybe_mask(m) for m in mem_mask),
                mem_weights=mem_weights,
                cats_enabled_override=self.cfg.dual_path_uncertainty_cats,
                return_attn=False,
            )
            head_in = dec_masked.detach() if self.cfg.head_detach else dec_masked
            uncert_in = dec_uncert.detach() if self.cfg.head_detach else dec_uncert
            if self.cfg.moe_enabled:
                q_list = []
                for head in self.expert_heads or []:
                    q_list.append(head(head_in, uncert_in))
                q_hat, moe_weights, moe_logits = self._apply_moe(head_in, q_list, y_past, mask)
                extras["moe_weights"] = moe_weights
                extras["moe_logits"] = moe_logits
            else:
                q_hat = self.head(head_in, uncert_in)
            if return_diagnostics:
                extras["dec_out"] = dec_masked
                extras["dec_out_uncert"] = dec_uncert
        else:
            dec_out, cats_mask, attn = self.decoder(
                mem,
                x_future_feats,
                _maybe_mask(mem_mask)
                if not isinstance(mem_mask, tuple)
                else tuple(_maybe_mask(m) for m in mem_mask),
                mem_weights=mem_weights,
                return_attn=return_attn,
            )
            head_in = dec_out.detach() if self.cfg.head_detach else dec_out
            if self.cfg.moe_enabled:
                q_list = []
                for head in self.expert_heads or []:
                    q_list.append(head(head_in))
                q_hat, moe_weights, moe_logits = self._apply_moe(head_in, q_list, y_past, mask)
                extras["moe_weights"] = moe_weights
                extras["moe_logits"] = moe_logits
            else:
                q_hat = self.head(head_in)
            if return_diagnostics:
                extras["dec_out"] = dec_out
        if self.rank_head is not None:
            rank_in = dec_masked if dual_path else dec_out
            if self.cfg.rank_head_detach:
                rank_in = rank_in.detach()
            if self.rank_dropout is not None:
                rank_in = self.rank_dropout(rank_in)
            extras["rank_pred"] = self.rank_head(rank_in).squeeze(-1)
        if self.dir_head_enabled:
            dir_in = dec_masked if dual_path else dec_out
            if self.cfg.dir_head_detach:
                dir_in = dir_in.detach()
            if self.dir_dropout is not None:
                dir_in = self.dir_dropout(dir_in)
            if self.dir_logits3 is not None:
                extras["dir_logits3"] = self.dir_logits3(dir_in)
            else:
                extras["dir_move_logits"] = self.dir_move(dir_in).squeeze(-1)
                extras["dir_dir_logits"] = self.dir_dir(dir_in).squeeze(-1)
        if self.cumret24_head_enabled:
            if dual_path:
                cumret_in = dec_masked
            else:
                cumret_in = dec_out
            h_idx = min(self.cfg.H - 1, cumret_in.shape[1] - 1)
            extras["cumret24"] = self.cumret24_head(cumret_in[:, h_idx, :]).squeeze(-1)
        if cats_mask is not None:
            extras["cats_keep"] = cats_mask
        if return_attn and attn is not None:
            extras["attn"] = attn
        return q_hat, extras

    def _apply_moe(
        self,
        gate_source: torch.Tensor,
        q_list: List[torch.Tensor],
        y_past: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.moe_gate is None:
            raise RuntimeError("moe_gate is not initialized.")
        if not q_list:
            raise RuntimeError("moe_enabled but no expert outputs.")
        pooled = gate_source.mean(dim=1)
        gate_in = pooled
        if self.cfg.moe_gate_use_regime_features:
            gate_feats = make_gate_features(
                y_past,
                mask,
                window=self.cfg.moe_gate_regime_window,
                eps=self.cfg.moe_gate_regime_eps,
            )
            gate_in = torch.cat([gate_in, gate_feats], dim=-1)
        logits = self.moe_gate(gate_in) / self.cfg.moe_gate_temperature
        if self.cfg.moe_gate_logit_clip is not None:
            clip = float(self.cfg.moe_gate_logit_clip)
            logits = torch.clamp(logits, -clip, clip)
        weights = torch.softmax(logits, dim=-1)
        if self.training and (self.cfg.moe_expert_drop_trend > 0 or self.cfg.moe_expert_drop_mr > 0):
            p_t = max(0.0, float(self.cfg.moe_expert_drop_trend))
            p_m = max(0.0, float(self.cfg.moe_expert_drop_mr))
            p_total = min(p_t + p_m, 1.0)
            if p_total > 0:
                draw = torch.rand((weights.shape[0], 1), device=weights.device)
                drop_trend = draw < p_t
                drop_mr = (draw >= p_t) & (draw < p_total)
                if drop_trend.any():
                    weights = weights.clone()
                    weights[drop_trend.squeeze(-1), 0] = 0.0
                    weights[drop_trend.squeeze(-1), 1] = 1.0
                if drop_mr.any():
                    weights = weights.clone()
                    weights[drop_mr.squeeze(-1), 0] = 1.0
                    weights[drop_mr.squeeze(-1), 1] = 0.0
        q_hat = torch.zeros_like(q_list[0])
        for idx, q in enumerate(q_list):
            w = weights[:, idx].view(-1, 1, 1)
            q_hat = q_hat + q * w
        return q_hat, weights, logits
