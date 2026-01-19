from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .aci import ACIGuardrail, ResidualBuffer


def _key_to_str(h: int, bucket: Tuple[int | None, int | None]) -> str:
    hod, vol = bucket
    hod_s = "-1" if hod is None else str(hod)
    vol_s = "-1" if vol is None else str(vol)
    return f"{h}|{hod_s}|{vol_s}"


def _str_to_key(key: str) -> Tuple[int, Tuple[int | None, int | None]]:
    h_s, hod_s, vol_s = key.split("|")
    h = int(h_s)
    hod = None if hod_s == "-1" else int(hod_s)
    vol = None if vol_s == "-1" else int(vol_s)
    return h, (hod, vol)


def save_state(
    path: str,
    alpha: list[float],
    buffer: ResidualBuffer,
    vol_thresholds: list[float],
    last_t: int | None,
    config: Dict[str, float],
) -> None:
    state: Dict[str, object] = {
        "alpha": alpha,
        "vol_thresholds": vol_thresholds,
        "last_t": last_t,
        "config": config,
        "buffer": {},
    }
    buf_out: Dict[str, Dict[str, list[float]]] = {}
    for (h, bucket), entries in buffer._store.items():
        key = _key_to_str(h, bucket)
        times = [float(t) for t, _ in entries]
        values = [float(v) for _, v in entries]
        buf_out[key] = {"times": times, "values": values}
    state["buffer"] = buf_out
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(state, f)


def load_state(path: str) -> Tuple[list[float], ResidualBuffer, list[float], int | None, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    config = state.get("config", {})
    guard = ACIGuardrail(
        eps_width=float(config.get("eps_width", 1e-6)),
        s_clip=(float(config.get("s_clip_min", 0.1)), float(config.get("s_clip_max", 5.0))),
    )
    buffer = ResidualBuffer(
        int(config.get("window", 240)),
        int(config.get("min_count", 100)),
        float(config.get("shrinkage_tau", 1000.0)),
        guard,
    )
    for key, payload in state.get("buffer", {}).items():
        h, bucket = _str_to_key(key)
        times = payload.get("times", [])
        values = payload.get("values", [])
        for t, v in zip(times, values):
            buffer.add(h, bucket, int(t), float(v))
    vol_thresholds = state.get("vol_thresholds", [])
    if not isinstance(vol_thresholds, list):
        vol_thresholds = []
    last_t = state.get("last_t")
    return state.get("alpha", []), buffer, vol_thresholds, last_t, config
