"""Base interfaces for pose estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..types import PoseResult


class PoseEstimatorBase(ABC):
    """Abstract base class for pose estimators."""

    @abstractmethod
    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        """Estimate pose landmarks for ``image_bgr``."""

    def close(self) -> None:
        """Release resources (optional override)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        self.close()
        return None


__all__ = ["PoseEstimatorBase"]
