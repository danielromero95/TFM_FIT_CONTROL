"""Utilidades de series temporales para métricas de pose."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def angular_velocity(series: Iterable[float], fps: float, method: str = "forward") -> List[float]:
    """Devuelve la velocidad angular conservando la longitud de la serie original."""

    values = np.asarray(list(series), dtype=float)
    n = int(values.size)
    if n == 0:
        return []
    if fps == 0:
        return [0.0] * n

    dt = 1.0 / float(fps)

    if method == "central":
        if n < 2:
            return [0.0] * n

        mask = np.isfinite(values)
        if mask.sum() < 2:
            return [0.0] * n

        indices = np.arange(n, dtype=float)
        filled = values.copy()
        filled[~mask] = np.interp(indices[~mask], indices[mask], values[mask])

        vel = np.gradient(filled, dt)
        vel = np.where(mask, vel, 0.0)
        return vel.astype(float).tolist()

    if method != "forward":
        raise ValueError(f"Unsupported angular velocity method: {method}")

    if n == 1:
        return [0.0]

    diffs = np.diff(values)
    valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    step_vel = np.where(valid, np.abs(diffs) / dt, 0.0)

    vel = np.empty(n, dtype=float)
    vel[0] = 0.0
    vel[1:] = step_vel
    return vel.tolist()


def calculate_angular_velocity(angle_sequence: Iterable[float], fps: float) -> List[float]:
    """Envoltura de compatibilidad que usa diferencias hacia delante como en la versión legada."""

    return angular_velocity(angle_sequence, fps, method="forward")


__all__ = ["angular_velocity", "calculate_angular_velocity"]
