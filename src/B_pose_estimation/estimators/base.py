"""Interfaces base para los estimadores de pose."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..types import PoseResult


class PoseEstimatorBase(ABC):
    """Clase base abstracta para implementar estimadores de pose."""

    @abstractmethod
    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        """Estima los marcadores corporales presentes en ``image_bgr``."""

    def close(self) -> None:
        """Libera recursos asociados (puede sobrescribirse según la implementación)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        self.close()
        return None


__all__ = ["PoseEstimatorBase"]
