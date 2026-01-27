import argparse
from pathlib import Path

import numpy as np


def load_pnl(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 0:
        data = np.array([float(data)])
    return data.astype(np.float64)


def bootstrap_mean_ci(pnl: np.ndarray, n: int, alpha: float) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n):
        sample = rng.choice(pnl, size=pnl.size, replace=True)
        means.append(np.mean(sample))
    means = np.asarray(means, dtype=np.float64)
    lo = np.quantile(means, alpha / 2.0)
    hi = np.quantile(means, 1.0 - alpha / 2.0)
    return float(np.mean(pnl)), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnl_csv", required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    pnl = load_pnl(Path(args.pnl_csv))
    mean, lo, hi = bootstrap_mean_ci(pnl, args.n, args.alpha)
    print(f"n={pnl.size} mean={mean:.6f} ci{int((1-args.alpha)*100)}=[{lo:.6f}, {hi:.6f}]")


if __name__ == "__main__":
    main()
