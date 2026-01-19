from __future__ import annotations

import argparse
from typing import List

import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from width_scaling import apply_width_scaling


def _parse_s_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _masked_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask: np.ndarray) -> float:
    inside = (y >= lo) & (y <= hi) & (mask > 0)
    denom = np.sum(mask)
    return float(np.sum(inside) / max(denom, 1.0))


def _masked_width(lo: np.ndarray, hi: np.ndarray, mask: np.ndarray) -> float:
    denom = np.sum(mask)
    return float(np.sum((hi - lo) * mask) / max(denom, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_npz", required=True)
    parser.add_argument("--s_list", default="0.1,1.0,1.5")
    args = parser.parse_args()

    data = np.load(args.calib_npz)
    y = data["y"]
    q10 = data["q10"]
    q50 = data["q50"]
    q90 = data["q90"]
    if "mask" in data:
        mask = data["mask"].astype(np.float32)
    else:
        mask = np.isfinite(y).astype(np.float32)
    y = np.nan_to_num(y, nan=0.0)
    q10 = np.nan_to_num(q10, nan=0.0)
    q50 = np.nan_to_num(q50, nan=0.0)
    q90 = np.nan_to_num(q90, nan=0.0)

    half = 0.5 * np.abs(q90 - q10)
    half_min = float(np.min(half))
    half_med = float(np.median(half))
    half_max = float(np.max(half))
    half_zero_frac = float(np.mean(half <= 1e-12))

    print("half_stats", f"min={half_min:.6f}", f"median={half_med:.6f}", f"max={half_max:.6f}", f"zero_frac={half_zero_frac:.6f}")

    s_values = _parse_s_list(args.s_list)
    covs = []
    widths = []
    q50_max_delta = 0.0
    for s in s_values:
        lo, mid, hi = apply_width_scaling(q10, q50, q90, s)
        q50_max_delta = max(q50_max_delta, float(np.max(np.abs(mid - q50))))
        cov = _masked_coverage(y, lo, hi, mask)
        width = _masked_width(lo, hi, mask)
        covs.append(cov)
        widths.append(width)
        print(f"s={s:.3f} coverage={cov:.6f} width={width:.6f}")

    monotonic_cov = all(covs[i] <= covs[i + 1] + 1e-8 for i in range(len(covs) - 1))
    monotonic_width = all(widths[i] <= widths[i + 1] + 1e-8 for i in range(len(widths) - 1))
    print("cov_monotonic", monotonic_cov)
    print("width_monotonic", monotonic_width)
    print("q50_max_delta", f"{q50_max_delta:.6f}")

    y_test = np.array([[0.0], [0.0]], dtype=np.float32)
    lo_test = np.array([[-1.0], [-1.0]], dtype=np.float32)
    hi_test = np.array([[1.0], [1.0]], dtype=np.float32)
    mask_test = np.ones_like(y_test)
    cov_test = _masked_coverage(y_test, lo_test, hi_test, mask_test)
    print("synthetic_coverage", f"{cov_test:.6f}")


if __name__ == "__main__":
    main()
