from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class TimeSplit:
    train_end: int
    val_end: int
    test_end: int


def make_time_splits(T: int, train_frac: float, val_frac: float) -> TimeSplit:
    train_end = int(T * train_frac)
    val_end = train_end + int(T * val_frac)
    test_end = T
    return TimeSplit(train_end=train_end, val_end=val_end, test_end=test_end)


def generate_origins(T: int, L: int, H: int, step: int) -> List[int]:
    origins = []
    t = L - 1
    last = T - H - 1
    while t <= last:
        origins.append(t)
        t += step
    return origins


def generate_panel_origins(
    series_lengths: Sequence[int],
    L: int,
    H: int,
    step: int,
) -> List[Tuple[int, int]]:
    indices: List[Tuple[int, int]] = []
    for series_idx, T in enumerate(series_lengths):
        for t in generate_origins(T, L, H, step):
            indices.append((series_idx, t))
    return indices


def select_indices_by_time(
    indices: Iterable[Tuple[int, int]],
    split: TimeSplit,
    split_name: str,
    horizon: int = 0,
    purge: int = 0,
    embargo: int = 0,
) -> List[Tuple[int, int]]:
    if split_name == "train":
        min_t = 0
        max_t = split.train_end - 1
    elif split_name == "val":
        min_t = split.train_end
        max_t = split.val_end - 1
    elif split_name == "test":
        min_t = split.val_end
        max_t = split.test_end - 1
    else:
        raise ValueError(f"Unknown split name: {split_name}")
    if split_name in {"val", "test"}:
        min_t += max(int(embargo), 0)
    if split_name in {"train", "val"}:
        max_t -= max(int(purge), 0)
    if max_t < min_t:
        return []
    max_t = max_t - horizon
    return [(s, t) for s, t in indices if min_t <= t <= max_t]


def assert_origin_contract(t: int, L: int, H: int, T: int) -> None:
    if t - L + 1 < 0:
        raise ValueError("Origin violates past window requirement.")
    if t + H >= T:
        raise ValueError("Origin violates forecast horizon requirement.")
