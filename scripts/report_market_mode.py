#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for val in np.unique(inv):
            idx = np.where(inv == val)[0]
            if idx.size > 1:
                ranks[idx] = ranks[idx].mean()
    return ranks


def _load_preds(path: Path, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as d:
        y = d["y"]
        q50 = d["q50"]
        mask = d["mask"] if "mask" in d else np.isfinite(y).astype(np.float32)
        origin_t = d["origin_t"]
    h_idx = h - 1
    valid = mask[:, h_idx] > 0
    return y[valid, h_idx], q50[valid, h_idx], origin_t[valid]


def _market_returns(y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(t)
    mkt = np.zeros_like(uniq, dtype=np.float64)
    for i, tt in enumerate(uniq):
        idx = t == tt
        mkt[i] = float(np.mean(y[idx])) if np.any(idx) else float("nan")
    return uniq, mkt


def _score_cs(score: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = score.copy().astype(np.float64)
    for tt in np.unique(t):
        idx = t == tt
        if np.any(idx):
            out[idx] = out[idx] - np.mean(out[idx])
    return out


def _ret_cs(ret: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = ret.copy().astype(np.float64)
    for tt in np.unique(t):
        idx = t == tt
        if np.any(idx):
            out[idx] = out[idx] - np.mean(out[idx])
    return out


def _ic_stats(score: np.ndarray, ret: np.ndarray, t: np.ndarray) -> Tuple[float, float]:
    ics = []
    ics_cs = []
    for tt in np.unique(t):
        idx = t == tt
        if np.sum(idx) < 5:
            continue
        s = score[idx]
        r = ret[idx]
        if np.std(s) < 1e-12 or np.std(r) < 1e-12:
            continue
        ics.append(_corr(s, r))
        s_cs = s - np.mean(s)
        r_cs = r - np.mean(r)
        if np.std(s_cs) < 1e-12 or np.std(r_cs) < 1e-12:
            ics_cs.append(float("nan"))
        else:
            ics_cs.append(_corr(s_cs, r_cs))
    ic_raw = float(np.nanmean(ics)) if ics else float("nan")
    ic_cs = float(np.nanmean(ics_cs)) if ics_cs else float("nan")
    return ic_raw, ic_cs


def _summarize_one(path: Path, h: int) -> dict:
    y, score, t = _load_preds(path, h)
    uniq, mkt = _market_returns(y, t)
    # sample-level correlation with market return (market return per sample)
    mkt_per_sample = np.zeros_like(y, dtype=np.float64)
    for tt, mv in zip(uniq, mkt):
        mkt_per_sample[t == tt] = mv
    score_cs = _score_cs(score, t)
    ret_cs = _ret_cs(y, t)

    corr_raw = _corr(score, mkt_per_sample)
    corr_cs = _corr(score_cs, mkt_per_sample)

    # time-level correlation using mean score per time
    mean_score = np.zeros_like(mkt, dtype=np.float64)
    mean_score_cs = np.zeros_like(mkt, dtype=np.float64)
    for i, tt in enumerate(uniq):
        idx = t == tt
        if np.any(idx):
            mean_score[i] = float(np.mean(score[idx]))
            mean_score_cs[i] = float(np.mean(score_cs[idx]))
        else:
            mean_score[i] = float("nan")
            mean_score_cs[i] = float("nan")

    corr_raw_time = _corr(mean_score, mkt)
    corr_cs_time = _corr(mean_score_cs, mkt)

    # IC raw vs CS
    ic_raw, ic_cs = _ic_stats(score, y, t)
    pooled_raw = _corr(score, y)
    pooled_cs = _corr(score_cs, ret_cs)
    pooled_raw_s = _corr(_rankdata(score), _rankdata(y))
    pooled_cs_s = _corr(_rankdata(score_cs), _rankdata(ret_cs))

    return {
        "path": str(path),
        "n_samples": int(y.size),
        "n_times": int(uniq.size),
        "corr_raw_mkt_sample": corr_raw,
        "corr_cs_mkt_sample": corr_cs,
        "corr_raw_mkt_time": corr_raw_time,
        "corr_cs_mkt_time": corr_cs_time,
        "ic_raw_mean": ic_raw,
        "ic_cs_mean": ic_cs,
        "pooled_ic_raw_pearson": pooled_raw,
        "pooled_ic_cs_pearson": pooled_cs,
        "pooled_ic_raw_spearman": pooled_raw_s,
        "pooled_ic_cs_spearman": pooled_cs_s,
    }


def _expand_paths(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.glob("fold_*/ens_preds.npz")))
        else:
            out.append(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", nargs="+", required=True, help="ens_preds.npz or run dir(s)")
    parser.add_argument("--h", type=int, default=6)
    args = parser.parse_args()

    paths = _expand_paths(args.preds)
    if not paths:
        raise SystemExit("No preds found.")

    rows = [_summarize_one(p, args.h) for p in paths]
    # pretty print
    for row in rows:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    import json

    main()
