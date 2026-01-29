from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class StandardScaler:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        if mask is None:
            mean = np.nanmean(x, axis=0)
            std = np.nanstd(x, axis=0)
        else:
            mask = mask.astype(bool)
            masked = np.where(mask, x, np.nan)
            mean = np.nanmean(masked, axis=0)
            std = np.nanstd(masked, axis=0)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std, self.eps).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fit.")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fit.")
        return x * self.std + self.mean


@dataclass
class FoldFitPreprocessor:
    scale_y: bool = True
    scale_x: bool = True

    def __post_init__(self) -> None:
        self.y_scaler = StandardScaler()
        self.x_scaler = StandardScaler()

    def fit(
        self,
        y_train: np.ndarray,
        x_train: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        if self.scale_y:
            self.y_scaler.fit(y_train, mask=mask)
        if self.scale_x and x_train is not None:
            self.x_scaler.fit(x_train)

    def transform(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        y_out = self.y_scaler.transform(y) if self.scale_y else y
        if x is None:
            return y_out, None
        x_out = self.x_scaler.transform(x) if self.scale_x else x
        return y_out, x_out

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(y) if self.scale_y else y

    def inverse_return(self, r: np.ndarray) -> np.ndarray:
        if not self.scale_y:
            return r
        if self.y_scaler.std is None:
            raise RuntimeError("Scaler has not been fit.")
        return r * self.y_scaler.std
