from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict

import yaml


def _deep_update(cfg: Dict, updates: Dict) -> Dict:
    for key, val in updates.items():
        if isinstance(val, dict):
            cfg.setdefault(key, {})
            _deep_update(cfg[key], val)
        else:
            cfg[key] = val
    return cfg


def _write_cfg(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/yahoo_big.yaml")
    parser.add_argument("--out_dir", default="configs/ablations")
    parser.add_argument("--only", default=None, help="Comma-separated list of variant names.")
    args = parser.parse_args()

    with open(args.base, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    base_name = Path(args.base).stem
    variants = {
        "cats_off_sa_off": {
            "decoder": {"mode": "CA_ONLY", "cats_masking": {"enabled": False}},
        },
        "cats_on_sa_off": {
            "decoder": {"mode": "CA_ONLY", "cats_masking": {"enabled": True}},
        },
        "cats_off_sa_on": {
            "decoder": {"mode": "HYBRID_QSA", "cats_masking": {"enabled": False}},
        },
        "cats_on_sa_on": {
            "decoder": {"mode": "HYBRID_QSA", "cats_masking": {"enabled": True}},
        },
        "fusion_concat": {
            "patching": {"fusion": "TWO_ENCODERS_CONCAT"},
        },
        "fusion_dual_sum_equal": {
            "patching": {"fusion": "DUAL_SUM", "dual_sum_weight": "equal"},
        },
        "fusion_dual_sum_token": {
            "patching": {"fusion": "DUAL_SUM", "dual_sum_weight": "token"},
        },
        "dual_path_uncert_off": {
            "decoder": {"dual_path": {"enabled": True, "uncertainty_cats": False}},
            "head": {"type": "DUAL_PATH"},
        },
        "dual_path_uncert_on": {
            "decoder": {"dual_path": {"enabled": True, "uncertainty_cats": True}},
            "head": {"type": "DUAL_PATH"},
        },
        "lsq_smin_0": {
            "head": {"type": "LSQ", "lsq_s_min": 0.0},
        },
        "lsq_smin_0p01": {
            "head": {"type": "LSQ", "lsq_s_min": 0.01},
        },
        "lsq_smin_0p03": {
            "head": {"type": "LSQ", "lsq_s_min": 0.03},
        },
        "tail_weight_2x": {
            "training": {"loss": {"quantile_weights": [2.0, 1.0, 2.0]}},
        },
        "width_penalty": {
            "training": {"loss": {"width_min": 0.01, "width_min_weight": 0.1}},
        },
        "repulsion_0p1": {
            "training": {"loss": {"repulsion_weight": 0.1, "repulsion_scale": 1.0}},
        },
        "missing_A1_no_mask": {
            "missingness": {"mask_embedding": False, "dt_embedding": "MLP_LOG1P", "attn_logit_bias": "HARD_NEG_INF"},
        },
        "missing_A2_no_dt": {
            "missingness": {"mask_embedding": True, "dt_embedding": "NONE", "attn_logit_bias": "HARD_NEG_INF"},
        },
        "missing_A3_no_attn": {
            "missingness": {"mask_embedding": True, "dt_embedding": "MLP_LOG1P", "attn_logit_bias": "NONE"},
        },
        "missing_A4_minimal": {
            "missingness": {"mask_embedding": False, "dt_embedding": "NONE", "attn_logit_bias": "NONE"},
        },
    }

    if args.only:
        keep = {name.strip() for name in args.only.split(",") if name.strip()}
        variants = {k: v for k, v in variants.items() if k in keep}

    out_dir = Path(args.out_dir)
    for name, updates in variants.items():
        cfg = copy.deepcopy(base_cfg)
        cfg = _deep_update(cfg, updates)
        if "training" in cfg:
            cfg["training"]["output_path"] = f"artifacts/{base_name}_{name}.pt"
            cfg["training"]["log_path"] = f"artifacts/{base_name}_{name}_train_log.jsonl"
        out_path = out_dir / f"{base_name}_{name}.yaml"
        _write_cfg(out_path, cfg)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
