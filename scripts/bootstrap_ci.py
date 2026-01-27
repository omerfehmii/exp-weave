import argparse
from pathlib import Path

import numpy as np


def load_pnl(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 0:
        data = np.array([float(data)])
    return data.astype(np.float64)


def lag1_autocorr(pnl: np.ndarray) -> float:
    if pnl.size < 2:
        return float("nan")
    x = pnl[:-1]
    y = pnl[1:]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y)) + 1e-12
    return float(np.sum(x * y) / denom)


def n_eff_from_rho1(n: int, rho1: float) -> float:
    if not np.isfinite(rho1):
        return float(n)
    neff = n * (1.0 - rho1) / (1.0 + rho1 + 1e-12)
    return float(max(1.0, neff))


def block_bootstrap_sample(pnl: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = pnl.size
    if block_len <= 1:
        return rng.choice(pnl, size=n, replace=True)
    block_len = min(block_len, n)
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    chunks = [pnl[s : s + block_len] for s in starts]
    sample = np.concatenate(chunks, axis=0)[:n]
    return sample


def bootstrap_mean_ci(
    pnl: np.ndarray, n: int, alpha: float, method: str, block_len: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n):
        if method == "block":
            sample = block_bootstrap_sample(pnl, block_len, rng)
        else:
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
    parser.add_argument("--method", choices=["iid", "block"], default="iid")
    parser.add_argument("--block_len", type=int, default=0)
    args = parser.parse_args()

    pnl = load_pnl(Path(args.pnl_csv))
    rho1 = lag1_autocorr(pnl)
    neff = n_eff_from_rho1(pnl.size, rho1)
    block_len = int(args.block_len)
    if args.method == "block" and block_len <= 0:
        block_len = max(2, int(round(np.sqrt(pnl.size))))
    mean, lo, hi = bootstrap_mean_ci(pnl, args.n, args.alpha, args.method, block_len)
    print(
        f"n={pnl.size} rho1={rho1:.4f} n_eff={neff:.1f} "
        f"mean={mean:.6f} ci{int((1-args.alpha)*100)}=[{lo:.6f}, {hi:.6f}]"
    )
    if args.method == "block":
        print(f"block_len={block_len}")


if __name__ == "__main__":
    main()
