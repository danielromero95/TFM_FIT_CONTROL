"""Interfaces base que comparten todos los estimadores de pose."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..types import PoseResult


class PoseEstimatorBase(ABC):
    """Clase abstracta que define el contrato mínimo de un estimador de pose."""

    @abstractmethod
    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        """Estima los *landmarks* de pose presentes en ``image_bgr``."""

    def close(self) -> None:
        """Libera recursos asociados al estimador (sobrescribible)."""

    def __enter__(self):
        """Permite usar el estimador como *context manager* estándar."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        """Cierra el estimador al salir del contexto gestionado."""
        self.close()
        return None


__all__ = ["PoseEstimatorBase"]
