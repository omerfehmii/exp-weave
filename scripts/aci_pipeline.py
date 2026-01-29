from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--preds_in", default=None, help="If set, skip eval and use this preds npz.")
    parser.add_argument("--preds_out", default="runs/preds.npz")
    parser.add_argument("--aci_out", default="runs/aci_preds.npz")
    parser.add_argument("--aci_metrics_out", default="runs/aci_metrics.npz")
    parser.add_argument("--state_in", default=None)
    parser.add_argument("--state_out", default=None)
    parser.add_argument("--alpha_target", type=float, default=0.20)
    parser.add_argument("--coverage_mode", default="per_horizon", choices=["per_horizon", "trajectory"])
    parser.add_argument("--coverage_target", type=float, default=0.80)
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--recent_window", type=int, default=0)
    parser.add_argument("--gamma_base", type=float, default=0.02)
    parser.add_argument("--hod_bins", type=int, default=6)
    parser.add_argument("--vol_bins", type=int, default=3)
    parser.add_argument("--min_count", type=int, default=100)
    parser.add_argument("--shrinkage_tau", type=float, default=1000.0)
    parser.add_argument("--s_clip_min", type=float, default=0.1)
    parser.add_argument("--s_clip_max", type=float, default=5.0)
    parser.add_argument("--retro_refresh", default="false")
    parser.add_argument("--retro_every", type=int, default=1)
    parser.add_argument("--switch_prob", default="false")
    parser.add_argument("--ema_fast", type=float, default=0.2)
    parser.add_argument("--ema_slow", type=float, default=0.02)
    parser.add_argument("--switch_gamma", type=float, default=1.0)
    parser.add_argument("--lambda_min", type=float, default=0.15)
    parser.add_argument("--lambda_max", type=float, default=0.5)
    parser.add_argument("--switch_eps", type=float, default=1e-6)
    args = parser.parse_args()

    preds = args.preds_in or args.preds_out
    if args.preds_in is None:
        out_path = Path(args.preds_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                "eval.py",
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--split",
                args.split,
                "--use_cqr",
                "false",
                "--out_npz",
                str(out_path),
            ]
        )

    state_out = args.state_out
    if args.state_in and state_out is None:
        state_out = args.state_in

    cmd = [
        sys.executable,
        "scripts/aci_backtest.py",
        "--config",
        args.config,
        "--preds",
        preds,
        "--out_npz",
        args.aci_out,
        "--out_metrics",
        args.aci_metrics_out,
        "--alpha_target",
        str(args.alpha_target),
        "--coverage_mode",
        args.coverage_mode,
        "--coverage_target",
        str(args.coverage_target),
        "--window",
        str(args.window),
        "--gamma_base",
        str(args.gamma_base),
        "--hod_bins",
        str(args.hod_bins),
        "--vol_bins",
        str(args.vol_bins),
        "--min_count",
        str(args.min_count),
        "--shrinkage_tau",
        str(args.shrinkage_tau),
        "--s_clip_min",
        str(args.s_clip_min),
        "--s_clip_max",
        str(args.s_clip_max),
    ]
    if args.retro_refresh:
        cmd.extend(["--retro_refresh", args.retro_refresh, "--retro_every", str(args.retro_every)])
    if args.switch_prob:
        cmd.extend(
            [
                "--switch_prob",
                args.switch_prob,
                "--ema_fast",
                str(args.ema_fast),
                "--ema_slow",
                str(args.ema_slow),
                "--switch_gamma",
                str(args.switch_gamma),
                "--lambda_min",
                str(args.lambda_min),
                "--lambda_max",
                str(args.lambda_max),
                "--switch_eps",
                str(args.switch_eps),
            ]
        )
    if args.recent_window:
        cmd.extend(["--recent_window", str(args.recent_window)])
    if args.state_in:
        cmd.extend(["--state_in", args.state_in])
    if state_out:
        cmd.extend(["--state_out", state_out])
    _run(cmd)


if __name__ == "__main__":
    main()
