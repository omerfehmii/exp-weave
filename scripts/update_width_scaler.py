from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[1]))

from width_scaling import fit_s_global, fit_s_per_horizon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--mode", default="global", choices=["global", "per_horizon"])
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--window", type=int, default=720, help="Hours for rolling window.")
    parser.add_argument("--s_lo", type=float, default=0.1)
    parser.add_argument("--s_hi", type=float, default=1.5)
    parser.add_argument("--iters", type=int, default=40)
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

    if "origin_t" in data:
        origin_t = data["origin_t"].astype(np.int64)
        max_t = int(np.max(origin_t))
        keep = origin_t >= (max_t - args.window)
        y = y[keep]
        q10 = q10[keep]
        q50 = q50[keep]
        q90 = q90[keep]
        mask = mask[keep]

    if args.mode == "per_horizon":
        s = fit_s_per_horizon(
            y,
            q10,
            q50,
            q90,
            target=args.target,
            s_lo=args.s_lo,
            s_hi=args.s_hi,
            iters=args.iters,
            mask=mask,
        )
    else:
        s = fit_s_global(
            y,
            q10,
            q50,
            q90,
            target=args.target,
            s_lo=args.s_lo,
            s_hi=args.s_hi,
            iters=args.iters,
            mask=mask,
        )

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, s=s, mode=args.mode, target=args.target, window=args.window)
    print(f"saved width_scaler to {out}")


if __name__ == "__main__":
    main()
