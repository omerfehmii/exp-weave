from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from data.loader import WindowedDataset, load_panel_npz, compress_series_observed
from eval import apply_scaling, build_model
from utils import load_config


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_cls", required=True)
    parser.add_argument("--checkpoint_cls", required=True)
    parser.add_argument("--preds_base", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9")
    parser.add_argument("--h_filter", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg_cls = load_config(args.config_cls)
    preds = np.load(args.preds_base)
    for key in ("y", "q50", "mask", "origin_t", "series_idx"):
        if key not in preds:
            raise ValueError(f"preds npz missing required key: {key}")
    y = preds["y"]
    q50 = preds["q50"]
    mask = preds["mask"].astype(np.float32)
    origin_t = preds["origin_t"].astype(np.int64)
    series_idx = preds["series_idx"].astype(np.int64)

    series_list = load_panel_npz(cfg_cls["data"]["path"])
    if cfg_cls.get("data", {}).get("observed_only", False):
        series_list = compress_series_observed(series_list)
    for s in series_list:
        s.ensure_features()
    # apply same scaling as classifier config
    lengths = [len(s.y) for s in series_list]
    train_end = int(min(lengths) * cfg_cls["data"].get("train_frac", 0.7))
    apply_scaling(
        series_list,
        train_end,
        scale_x=cfg_cls["data"].get("scale_x", True),
        scale_y=cfg_cls["data"].get("scale_y", True),
    )

    indices = list(zip(series_idx.tolist(), origin_t.tolist()))
    target_mode = cfg_cls["data"].get("target_mode", "level")
    target_log_eps = float(cfg_cls["data"].get("target_log_eps", 1e-6))
    ds = WindowedDataset(
        series_list,
        indices,
        cfg_cls["data"]["L"],
        cfg_cls["data"]["H"],
        target_mode=target_mode,
        target_log_eps=target_log_eps,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device(cfg_cls.get("training", {}).get("device", "cpu"))
    model = build_model(cfg_cls)
    state = torch.load(args.checkpoint_cls, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    logits_list = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _, extras = model(batch)
            if "dir_logits3" not in extras:
                raise RuntimeError("Classifier head missing dir_logits3.")
            logits_list.append(extras["dir_logits3"].cpu().numpy())
    logits3 = np.concatenate(logits_list, axis=0)

    h_idx = args.h_filter - 1
    if h_idx < 0 or h_idx >= y.shape[1]:
        raise ValueError("h_filter out of range.")
    probs = _softmax(logits3[:, h_idx, :])
    # class order: 0=down, 1=flat, 2=up
    p_down = probs[:, 0]
    p_flat = probs[:, 1]
    p_up = probs[:, 2]
    cls_pred = np.argmax(probs, axis=1)

    y_t = np.array([series_list[s_idx].y[t] for s_idx, t in indices])
    if y_t.ndim == 2 and y_t.shape[1] > 1:
        y_t = y_t[:, 0]
    label_1h = (y[:, 0] - y_t > 0).astype(np.int32)
    label_4h = (y[:, h_idx] - y_t > 0).astype(np.int32)
    pred_1h = (q50[:, 0] - y_t > 0).astype(np.int32)

    valid = (mask[:, 0] > 0) & (mask[:, h_idx] > 0) & np.isfinite(y_t)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    rows = []
    base_pred1_acc = _acc(label_1h[valid], pred_1h[valid])
    for t in thresholds:
        accept = valid & (np.maximum(p_up, p_down) >= t) & (cls_pred != 1)
        if not np.any(accept):
            rows.append([t, 0.0, "nan", "nan", base_pred1_acc, "nan", "nan", 0.0])
            continue
        filt_dir = np.where(cls_pred == 2, 1, 0)
        filt_dir = filt_dir[accept]
        lab1 = label_1h[accept]
        lab4 = label_4h[accept]
        pred1 = pred_1h[accept]
        cov = float(np.mean(accept))
        acc_1h = _acc(lab1, filt_dir)
        acc_4h = _acc(lab4, filt_dir)
        pred1_acc = _acc(lab1, pred1)
        agree = filt_dir == pred1
        agree_rate = float(np.mean(agree)) if agree.size else float("nan")
        pred1_acc_agree = _acc(lab1[agree], pred1[agree]) if np.any(agree) else float("nan")
        rows.append([t, cov, acc_1h, acc_4h, base_pred1_acc, pred1_acc, pred1_acc_agree, agree_rate])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "coverage",
                "filter_acc_1h",
                "filter_acc_4h",
                "pred1_acc_all",
                "pred1_acc_given_filter",
                "pred1_acc_when_agree",
                "filter_pred1_agree_rate",
            ]
        )
        writer.writerows(rows)
    print("pred1_acc_all", base_pred1_acc)


if __name__ == "__main__":
    main()
