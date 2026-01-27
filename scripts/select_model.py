#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _to_float(val: str | None) -> float:
    if val is None:
        return float("nan")
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


def _row_float(row: dict, key: str, legacy_key: str | None = None) -> float:
    val = row.get(key)
    if (val is None or str(val).strip() == "") and legacy_key:
        val = row.get(legacy_key)
    return _to_float(val)


def load_candidates(summary_path: Path) -> list[dict]:
    with summary_path.open() as f:
        rows = list(csv.DictReader(f))
    candidates: list[dict] = []
    for row in rows:
        for tag in ("best_loss", "best_dir", "last"):
            prefix = f"{tag}_"
            if prefix + "mae" not in row:
                continue
            cand = {
                "run_name": row.get("run_name"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "config": row.get("config"),
                "checkpoint": row.get(f"checkpoint_{tag}") if tag != "last" else row.get("checkpoint_last"),
                "tag": tag,
                "mae": _to_float(row.get(prefix + "mae")),
                "pinball": _to_float(row.get(prefix + "pinball")),
                "coverage80": _row_float(row, prefix + "coverage80", prefix + "coverage90"),
                "width80": _row_float(row, prefix + "width80", prefix + "width90"),
                "aci_coverage80": _row_float(row, prefix + "aci_coverage80", prefix + "aci_coverage90"),
                "aci_width80": _row_float(row, prefix + "aci_width80", prefix + "aci_width90"),
                "dir_wMCC": _to_float(row.get(prefix + "dir_wMCC")),
                "dir_wAUC": _to_float(row.get(prefix + "dir_wAUC")),
                "dir_acc_mean": _to_float(row.get(prefix + "dir_acc_mean")),
            }
            candidates.append(cand)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary.csv (seed3)")
    parser.add_argument("--baseline_model", default="detach")
    parser.add_argument("--budget_mae_mult", type=float, default=1.7)
    parser.add_argument("--budget_pinball_mult", type=float, default=2.0)
    parser.add_argument("--aci_min", type=float, default=0.78)
    parser.add_argument("--aci_max", type=float, default=0.82)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    candidates = load_candidates(summary_path)
    if not candidates:
        raise SystemExit("no candidates found in summary")

    baseline = [c for c in candidates if c["model"] == args.baseline_model and c["tag"] == "best_loss"]
    if not baseline:
        raise SystemExit(f"no baseline candidates found for model={args.baseline_model}")
    baseline_mae = min(c["mae"] for c in baseline if _is_finite(c["mae"]))
    baseline_pinball = min(c["pinball"] for c in baseline if _is_finite(c["pinball"]))
    if not _is_finite(baseline_mae) or not _is_finite(baseline_pinball):
        raise SystemExit("baseline metrics missing")

    filtered = []
    for c in candidates:
        if not (_is_finite(c["mae"]) and _is_finite(c["pinball"]) and _is_finite(c["aci_coverage80"]) and _is_finite(c["dir_wMCC"])):
            continue
        if c["mae"] > args.budget_mae_mult * baseline_mae:
            continue
        if c["pinball"] > args.budget_pinball_mult * baseline_pinball:
            continue
        if not (args.aci_min <= c["aci_coverage80"] <= args.aci_max):
            continue
        filtered.append(c)

    if not filtered:
        print("no candidates satisfy budget constraints")
        return

    filtered.sort(key=lambda x: (x["dir_wMCC"], x["dir_wAUC"], -x["mae"]), reverse=True)
    best = filtered[0]

    print("selected", best["run_name"], best["tag"])
    print("checkpoint", best["checkpoint"])
    print("mae", best["mae"], "pinball", best["pinball"], "aci_cov", best["aci_coverage80"])
    print("dir_wMCC", best["dir_wMCC"], "dir_wAUC", best["dir_wAUC"], "dir_acc", best["dir_acc_mean"])

    if args.output_json:
        out = {
            "baseline": {
                "model": args.baseline_model,
                "mae": baseline_mae,
                "pinball": baseline_pinball,
            },
            "constraints": {
                "budget_mae_mult": args.budget_mae_mult,
                "budget_pinball_mult": args.budget_pinball_mult,
                "aci_min": args.aci_min,
                "aci_max": args.aci_max,
            },
            "selected": best,
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_json).open("w") as f:
            json.dump(out, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
