from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def _parse_weights(weights: str | None, n: int) -> List[float]:
    if not weights:
        return [1.0 / n] * n
    parts = [float(x.strip()) for x in weights.split(",") if x.strip()]
    if len(parts) != n:
        raise ValueError("weights length must match number of preds files")
    total = sum(parts)
    if total <= 0:
        raise ValueError("weights must sum to > 0")
    return [w / total for w in parts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()

    weights = _parse_weights(args.weights, len(args.preds))

    base = np.load(args.preds[0])
    y = base["y"]
    mask = base["mask"] if "mask" in base else None
    origin_t = base["origin_t"] if "origin_t" in base else None
    series_idx = base["series_idx"] if "series_idx" in base else None
    regime = base["regime"] if "regime" in base else None

    q10 = np.zeros_like(base["q10"], dtype=np.float32)
    q50 = np.zeros_like(base["q50"], dtype=np.float32)
    q90 = np.zeros_like(base["q90"], dtype=np.float32)
    q50_stack = []

    for w, path in zip(weights, args.preds):
        preds = np.load(path)
        if origin_t is not None and not np.array_equal(preds["origin_t"], origin_t):
            raise ValueError("origin_t mismatch between preds files.")
        if series_idx is not None and not np.array_equal(preds["series_idx"], series_idx):
            raise ValueError("series_idx mismatch between preds files.")
        q10 += w * preds["q10"]
        q50 += w * preds["q50"]
        q90 += w * preds["q90"]
        q50_stack.append(preds["q50"].astype(np.float32))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q50_std = None
    if q50_stack:
        q50_std = np.std(np.stack(q50_stack, axis=0), axis=0)
    npz_kwargs = {"y": y, "q10": q10, "q50": q50, "q90": q90}
    if q50_std is not None:
        npz_kwargs["q50_std"] = q50_std
    if mask is not None:
        npz_kwargs["mask"] = mask
    if origin_t is not None:
        npz_kwargs["origin_t"] = origin_t
    if series_idx is not None:
        npz_kwargs["series_idx"] = series_idx
    if regime is not None:
        npz_kwargs["regime"] = regime
    np.savez(out_path, **npz_kwargs)
    print(f"saved ensemble preds -> {out_path}")


if __name__ == "__main__":
    main()
