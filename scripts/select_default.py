from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np


TARGET_COVERAGE = 0.80
MIN_COVERAGE = 0.78


def _load_metrics(path: str) -> Dict[str, float]:
    data = np.load(path, allow_pickle=True)
    metrics = data["metrics"].item()
    return metrics


def _metric(metrics: Dict[str, float], key: str, legacy_key: str) -> float:
    if key in metrics:
        return float(metrics[key])
    if legacy_key in metrics:
        return float(metrics[legacy_key])
    return float("nan")


def _score(metrics: Dict[str, float]) -> tuple[float, float, float, float]:
    coverage = _metric(metrics, "coverage80", "coverage90")
    width = _metric(metrics, "width80", "width90")
    return (
        abs(coverage - TARGET_COVERAGE),
        width,
        metrics["pinball"],
        metrics["collapse"],
    )


def _s_stats(path: Optional[str]) -> Optional[Dict[str, float]]:
    if not path:
        return None
    data = np.load(path)
    s = data["s"]
    if np.isscalar(s):
        return {"min": float(s), "median": float(s), "max": float(s)}
    return {"min": float(np.min(s)), "median": float(np.median(s)), "max": float(np.max(s))}


def _eligible(metrics: Dict[str, float]) -> bool:
    if metrics.get("crossing", 0.0) > 0.0:
        return False
    if _metric(metrics, "coverage80", "coverage90") < MIN_COVERAGE:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--global_ws", required=True)
    parser.add_argument("--perh_ws", required=True)
    parser.add_argument("--perh_s", default=None)
    parser.add_argument("--global_s", default=None)
    args = parser.parse_args()

    candidates = {
        "A_base": {"metrics": _load_metrics(args.base), "s_stats": None},
        "B_global": {"metrics": _load_metrics(args.global_ws), "s_stats": _s_stats(args.global_s)},
        "C_per_h": {"metrics": _load_metrics(args.perh_ws), "s_stats": _s_stats(args.perh_s)},
    }

    perh_stats = candidates["C_per_h"]["s_stats"]
    perh_unstable = False
    if perh_stats is not None:
        perh_unstable = perh_stats["min"] < 0.2 or perh_stats["max"] > 2.0

    base_metrics = candidates["A_base"]["metrics"]
    if _eligible(base_metrics) and abs(_metric(base_metrics, "coverage80", "coverage90") - TARGET_COVERAGE) <= 0.01:
        print("prod_default=A_base (coverage near target, simplest)")
        print("metrics", base_metrics)
        return

    eligible = {k: v for k, v in candidates.items() if _eligible(v["metrics"])}
    if perh_unstable and "C_per_h" in eligible:
        print("warning: per-horizon s unstable; deprioritizing C_per_h.")
        eligible.pop("C_per_h", None)

    if not eligible:
        print("warning: no eligible candidates, falling back to lowest score among all.")
        eligible = candidates

    best = min(eligible.items(), key=lambda kv: _score(kv[1]["metrics"]))
    print(f"prod_default={best[0]}")
    print("metrics", best[1]["metrics"])
    if best[1]["s_stats"] is not None:
        print("s_stats", best[1]["s_stats"])


if __name__ == "__main__":
    main()
