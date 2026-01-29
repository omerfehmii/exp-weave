from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import compute_delta_t, make_calendar_features, make_observation_mask


@dataclass
class SeriesData:
    y: np.ndarray
    y_raw: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    x_past_feats: Optional[np.ndarray] = None
    x_future_feats: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    delta_t: Optional[np.ndarray] = None
    series_id: Optional[int] = None

    def ensure_features(self) -> None:
        if self.mask is None:
            self.mask = make_observation_mask(self.y)
        if self.delta_t is None:
            self.delta_t = compute_delta_t(self.mask)
        if self.timestamps is not None and self.x_past_feats is None:
            self.x_past_feats = make_calendar_features(self.timestamps)
        if self.timestamps is not None and self.x_future_feats is None:
            self.x_future_feats = make_calendar_features(self.timestamps)


class WindowedDataset(Dataset):
    def __init__(
        self,
        series_list: Sequence[SeriesData],
        indices: Sequence[Tuple[int, int]],
        L: int,
        H: int,
        target_mode: str = "level",
        target_log_eps: float = 1e-6,
    ) -> None:
        self.series_list = series_list
        self.indices = list(indices)
        self.L = L
        self.H = H
        self.target_mode = target_mode
        self.target_log_eps = target_log_eps
        for series in self.series_list:
            series.ensure_features()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor]:
        series_idx, t = self.indices[idx]
        series = self.series_list[series_idx]
        y = series.y
        if y.ndim == 1:
            y = y[:, None]
        x_past = series.x_past_feats
        x_future = series.x_future_feats
        mask = series.mask
        delta_t = series.delta_t
        L = self.L
        H = self.H
        past_slice = slice(t - L + 1, t + 1)
        future_slice = slice(t + 1, t + H + 1)
        y_past = y[past_slice]
        y_future = y[future_slice]
        y_last = y_past[-1]
        if np.isnan(y_last).any():
            # Use last finite value in the past window per feature for return targets.
            y_last = y_last.copy()
            for d in range(y_past.shape[1]):
                col = y_past[:, d]
                idx = np.where(np.isfinite(col))[0]
                if idx.size:
                    y_last[d] = col[idx[-1]]
        if np.isnan(y_past).any():
            y_past = np.nan_to_num(y_past, nan=0.0)
        if x_past is None:
            x_past = np.zeros((y.shape[0], 0), dtype=np.float32)
        if x_future is None:
            x_future = np.zeros((y.shape[0], 0), dtype=np.float32)
        x_past_feats = x_past[past_slice]
        x_future_feats = x_future[future_slice]
        if mask is None:
            mask = np.ones_like(y, dtype=np.float32)
        if mask.ndim == 1:
            mask = mask[:, None]
        if delta_t is None:
            delta_t = np.zeros_like(y, dtype=np.float32)
        if delta_t.ndim == 1:
            delta_t = delta_t[:, None]
        mask_past = mask[past_slice]
        delta_past = delta_t[past_slice]

        batch = {
            "y_past": torch.from_numpy(y_past.astype(np.float32)),
            "x_past_feats": torch.from_numpy(x_past_feats.astype(np.float32)),
            "x_future_feats": torch.from_numpy(x_future_feats.astype(np.float32)),
            "mask": torch.from_numpy(mask_past.astype(np.float32)),
            "delta_t": torch.from_numpy(delta_past.astype(np.float32)),
        }
        if series.series_id is not None:
            batch["series_id"] = torch.tensor(series.series_id, dtype=torch.long)
        if self.target_mode == "return":
            target_arr = y_future - y_last[None, :]
        elif self.target_mode == "log_return":
            if series.y_raw is None:
                y_raw = y
            else:
                y_raw = series.y_raw
                if y_raw.ndim == 1:
                    y_raw = y_raw[:, None]
            y_past_raw = y_raw[past_slice]
            y_future_raw = y_raw[future_slice]
            y_last_raw = y_past_raw[-1]
            eps = self.target_log_eps
            target_arr = np.log(np.maximum(y_future_raw, eps) / np.maximum(y_last_raw[None, :], eps))
        else:
            target_arr = y_future
        target = torch.from_numpy(target_arr.astype(np.float32))
        return batch, target


def load_panel_npz(path: str) -> List[SeriesData]:
    data = np.load(path, allow_pickle=True)
    y = data["y"]
    timestamps = data.get("timestamps")
    x_past = data.get("x_past_feats")
    x_future = data.get("x_future_feats")
    mask = data.get("mask")
    delta_t = data.get("delta_t")
    series_ids = data.get("series_id")

    series_list: List[SeriesData] = []
    for i in range(y.shape[0]):
        series_list.append(
            SeriesData(
                y=y[i],
                timestamps=None if timestamps is None else timestamps[i],
                x_past_feats=None if x_past is None else x_past[i],
                x_future_feats=None if x_future is None else x_future[i],
                mask=None if mask is None else mask[i],
                delta_t=None if delta_t is None else delta_t[i],
                series_id=None if series_ids is None else int(series_ids[i]),
            )
        )
    return series_list


def compress_series_observed(series_list: List[SeriesData]) -> List[SeriesData]:
    compressed: List[SeriesData] = []
    for series in series_list:
        y = series.y
        y2 = y[:, None] if y.ndim == 1 else y
        mask = series.mask
        if mask is None:
            mask = make_observation_mask(y2)
        if mask.ndim == 1:
            mask = mask[:, None]
        obs = mask[:, 0] > 0
        if y2.shape[1] > 1:
            obs = np.all(mask > 0, axis=1)
        idx = np.where(obs)[0]
        if idx.size == 0:
            continue
        y_obs = y[idx] if y.ndim == 1 else y2[idx]
        ts_obs = None if series.timestamps is None else series.timestamps[idx]
        x_past_obs = None if series.x_past_feats is None else series.x_past_feats[idx]
        x_future_obs = None if series.x_future_feats is None else series.x_future_feats[idx]
        if ts_obs is not None and ts_obs.size > 0:
            dt = np.diff(ts_obs).astype("timedelta64[h]").astype(np.float32)
            dt = np.concatenate([[0.0], dt])
        else:
            dt = np.ones(len(idx), dtype=np.float32)
            dt[0] = 0.0
        if y_obs.ndim == 2:
            dt = np.repeat(dt[:, None], y_obs.shape[1], axis=1)
            mask_obs = np.ones_like(y_obs, dtype=np.float32)
        else:
            mask_obs = np.ones_like(y_obs, dtype=np.float32)
        compressed.append(
            SeriesData(
                y=y_obs,
                y_raw=None if series.y_raw is None else series.y_raw[idx],
                timestamps=ts_obs,
                x_past_feats=x_past_obs,
                x_future_feats=x_future_obs,
                mask=mask_obs,
                delta_t=dt,
                series_id=series.series_id,
            )
        )
    return compressed
