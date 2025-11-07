"""Time-series helpers for pose metrics."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


def angular_velocity(series: Iterable[float], fps: float, method: str = "forward") -> List[float]:
    """Return angular velocity with same length as the input series."""

    values = list(series)
    n = len(values)
    if n == 0:
        return []
    if fps == 0:
        return [0.0] * n

    dt = 1.0 / float(fps)

    if method == "central":
        if n < 2:
            return [0.0] * n
        s = pd.Series(values).interpolate(method="linear", limit_direction="both")
        arr = s.to_numpy(dtype=float)
        vel = np.gradient(arr, dt)
        return list(np.asarray(vel, dtype=float))

    if method != "forward":
        raise ValueError(f"Unsupported angular velocity method: {method}")

    arr = np.asarray(values, dtype=float)
    vel = np.empty(n, dtype=float)
    vel[0] = 0.0
    if n == 1:
        return vel.tolist()

    diffs = np.diff(arr)
    step_vel = np.abs(diffs) / dt
    valid = np.isfinite(arr[:-1]) & np.isfinite(arr[1:])
    step_vel = np.where(valid, step_vel, 0.0)

    vel[1:] = step_vel
    return vel.tolist()


def calculate_angular_velocity(angle_sequence: Iterable[float], fps: float) -> List[float]:
    """Backward compatibility wrapper for legacy forward-difference velocity."""

    return angular_velocity(angle_sequence, fps, method="forward")


__all__ = ["angular_velocity", "calculate_angular_velocity"]
