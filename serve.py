from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import torch

from model import ModelConfig, MultiScaleForecastModel, PatchScale
from utils import load_config
from width_scaling import apply_width_scaling


def build_model(cfg: Dict) -> MultiScaleForecastModel:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    patch_cfg = cfg["patching"]
    decoder_cfg = cfg.get("decoder", {})
    dual_path_cfg = decoder_cfg.get("dual_path", {})
    head_cfg = cfg.get("head", {})
    scales = [PatchScale(**scale) for scale in patch_cfg["scales"]]
    head_cfg = cfg.get("head", {})
    missing_cfg = cfg.get("missingness", {})
    delta_t_mode = missing_cfg.get("delta_t_mode", missing_cfg.get("dt_embedding", "MLP_LOG1P"))
    config = ModelConfig(
        L=data_cfg["L"],
        H=data_cfg["H"],
        target_dim=data_cfg.get("target_dim", 1),
        past_feat_dim=data_cfg.get("past_feat_dim", 0),
        future_feat_dim=data_cfg.get("future_feat_dim", 0),
        quantiles=data_cfg["quantiles"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        mlp_hidden=model_cfg["mlp_hidden"],
        dropout=model_cfg["dropout"],
        fusion=patch_cfg["fusion"],
        gate_entropy_floor=patch_cfg.get("gate", {}).get("entropy_floor"),
        gate_temperature=patch_cfg.get("gate", {}).get("temperature", 1.0),
        dual_sum_weight=patch_cfg.get("dual_sum_weight", "equal"),
        decoder_mode=decoder_cfg.get("mode", "CA_ONLY"),
        cats_enabled=decoder_cfg.get("cats_masking", {}).get("enabled", True),
        cats_p_min=decoder_cfg.get("cats_masking", {}).get("p_min", 0.1),
        cats_p_max=decoder_cfg.get("cats_masking", {}).get("p_max", 0.7),
        cats_scaling=decoder_cfg.get("cats_masking", {}).get("scaling", "NONE"),
        dual_path=dual_path_cfg.get("enabled", False),
        dual_path_uncertainty_cats=dual_path_cfg.get("uncertainty_cats", False),
        head_type=head_cfg.get("type", "MONO"),
        head_delta_floor=head_cfg.get("delta_floor", 0.0),
        head_lsq_s_min=head_cfg.get("lsq_s_min", 0.0),
        mask_embedding=missing_cfg.get("mask_embedding", True),
        delta_t_mode=delta_t_mode,
        attn_logit_bias=missing_cfg.get("attn_logit_bias", "HARD_NEG_INF"),
    )
    return MultiScaleForecastModel(config, scales)


def _to_tensor(arr: List) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to JSON input payload.")
    parser.add_argument("--width_scaler", default=None, help="Optional NPZ file with width scaling `s`.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    batch = {
        "y_past": _to_tensor(payload["y_past"]),
        "x_past_feats": _to_tensor(payload.get("x_past_feats", [])),
        "x_future_feats": _to_tensor(payload.get("x_future_feats", [])),
        "mask": _to_tensor(payload.get("mask", [])),
        "delta_t": _to_tensor(payload.get("delta_t", [])),
    }
    if "series_id" in payload:
        batch["series_id"] = torch.tensor([payload["series_id"]], dtype=torch.long)

    with torch.no_grad():
        q_hat, _ = model(batch)
    q_hat = q_hat.squeeze(0).numpy()
    quantiles = cfg["data"]["quantiles"]
    q10 = q_hat[:, quantiles.index(0.1)]
    q50 = q_hat[:, quantiles.index(0.5)]
    q90 = q_hat[:, quantiles.index(0.9)]
    if args.width_scaler is not None:
        data = np.load(args.width_scaler)
        s = data["s"]
        q10, q50, q90 = apply_width_scaling(q10, q50, q90, s)
    output = {"q10": q10.tolist(), "q50": q50.tolist(), "q90": q90.tolist()}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
