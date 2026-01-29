from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


def _read_tickers_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _get_close_frame(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.levels[0]
        level1 = df.columns.levels[1]
        if "Close" in level0:
            close = df["Close"]
        elif "Close" in level1:
            close = df.xs("Close", axis=1, level=1)
        else:
            raise KeyError("Close field not found in MultiIndex columns.")
    else:
        close = df[["Close"]].copy()
        close.columns = tickers[:1]
    close = close.sort_index()
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)
    close = close[~close.index.duplicated(keep="first")]
    return close


def _download_chunk(tickers: List[str], period: str, interval: str, retries: int) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for _ in range(retries):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
            )
            if df.empty:
                raise RuntimeError("empty response")
            return df
        except Exception as exc:  # pragma: no cover - network dependent
            last_err = exc
            time.sleep(1.0)
    raise RuntimeError(f"Yahoo download failed after retries: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default="AAPL,MSFT,GOOGL,AMZN,TSLA,SPY")
    parser.add_argument("--tickers-file", default=None)
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--output", default="data/yahoo_panel.npz")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Re-download even if output exists.")
    parser.add_argument("--no-full-hours", action="store_true", help="Do not expand to full hourly grid.")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"{args.output} already exists; use --force to re-download.")
        return

    tickers = _read_tickers_file(args.tickers_file) if args.tickers_file else _parse_tickers(args.tickers)
    if not tickers:
        raise ValueError("No tickers provided.")

    close_frames = []
    for chunk in _chunked(tickers, args.chunk_size):
        df = _download_chunk(chunk, args.period, args.interval, args.retries)
        close_frames.append(_get_close_frame(df, chunk))
    close = pd.concat(close_frames, axis=1)
    close = close.reindex(columns=tickers)
    if not args.no_full_hours:
        full_index = pd.date_range(start=close.index.min(), end=close.index.max(), freq="1h")
        close = close.reindex(full_index)

    y = close.to_numpy(dtype=np.float32).T
    timestamps = close.index.values.astype("datetime64[ns]")
    timestamps = np.tile(timestamps[None, :], (y.shape[0], 1))
    series_id = np.arange(y.shape[0], dtype=np.int64)
    series_names = np.asarray(tickers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        y=y,
        timestamps=timestamps,
        series_id=series_id,
        series_names=series_names,
    )
    print(f"saved {args.output} with shape={y.shape}")


if __name__ == "__main__":
    main()
