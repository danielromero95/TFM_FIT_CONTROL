"""Pure functions for computing biomechanical metrics."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _points_finite(landmarks, indices) -> bool:
    try:
        return all(
            np.isfinite(landmarks[i]["x"]) and np.isfinite(landmarks[i]["y"])
            for i in indices
        )
    except Exception:
        return False


def normalize_landmarks(landmarks: Iterable[Dict[str, float]]) -> list[Dict[str, float]]:
    """Centre landmarks around the midpoint of the hips."""
    hip_left, hip_right = landmarks[23], landmarks[24]
    cx = (hip_left["x"] + hip_right["x"]) / 2.0
    cy = (hip_left["y"] + hip_right["y"]) / 2.0
    return [
        {
            "x": lm["x"] - cx,
            "y": lm["y"] - cy,
            "z": lm["z"],
            "visibility": lm["visibility"],
        }
        for lm in landmarks
    ]


def calculate_angle(p1: Dict[str, float], p2: Dict[str, float], p3: Dict[str, float]) -> float:
    """Compute the angle in degrees formed by three points."""
    v1 = (p1["x"] - p2["x"], p1["y"] - p2["y"])
    v2 = (p3["x"] - p2["x"], p3["y"] - p2["y"])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 * mag2 == 0:
        return 0.0
    cosine = max(min(dot_product / (mag1 * mag2), 1.0), -1.0)
    return math.degrees(math.acos(cosine))


def extract_joint_angles(landmarks: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Return a dictionary with key joint angles."""

    joint_indices = {
        "left_knee": [23, 25, 27],
        "right_knee": [24, 26, 28],
        "left_elbow": [11, 13, 15],
        "right_elbow": [12, 14, 16],
    }

    angles: Dict[str, float] = {}
    for name, indices in joint_indices.items():
        if _points_finite(landmarks, indices):
            p1, p2, p3 = (landmarks[i] for i in indices)
            angles[name] = calculate_angle(p1, p2, p3)
        else:
            angles[name] = np.nan
    return angles


def calculate_distances(landmarks: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Return key distance measurements."""
    metrics: Dict[str, float] = {}
    if _points_finite(landmarks, [11, 12]):
        metrics["shoulder_width"] = abs(landmarks[12]["x"] - landmarks[11]["x"])
    else:
        metrics["shoulder_width"] = np.nan

    if _points_finite(landmarks, [27, 28]):
        metrics["foot_separation"] = abs(landmarks[28]["x"] - landmarks[27]["x"])
    else:
        metrics["foot_separation"] = np.nan

    return metrics


def angular_velocity(series: Iterable[float], fps: float, method: str = "forward") -> list[float]:
    """Return angular velocity with same length as the input series.

    - 'forward': vectorized forward difference using np.diff, padded with 0.0 at index 0.
                 Non-finite neighbor pairs yield 0.0 at that step (matches legacy semantics).
    - 'central': unchanged behavior (interpolation + np.gradient).
    """

    vals_list = list(series)
    n = len(vals_list)
    if n == 0:
        return []
    if fps == 0:
        return [0.0] * n

    dt = 1.0 / float(fps)

    if method == "central":
        if n < 2:
            return [0.0] * n
        s = pd.Series(vals_list).interpolate(method="linear", limit_direction="both")
        arr = s.to_numpy(dtype=float)
        vel = np.gradient(arr, dt)
        return list(np.asarray(vel, dtype=float))

    if method != "forward":
        raise ValueError(f"Unsupported angular velocity method: {method}")

    # Vectorized forward difference with NaN handling (pairwise finite check)
    arr = np.asarray(vals_list, dtype=float)
    vel = np.empty(n, dtype=float)
    vel[0] = 0.0
    if n == 1:
        return vel.tolist()

    diffs = np.diff(arr)  # length n-1
    step_vel = np.abs(diffs) / dt

    # Valid only when both neighbors are finite
    valid = np.isfinite(arr[:-1]) & np.isfinite(arr[1:])
    step_vel = np.where(valid, step_vel, 0.0)

    vel[1:] = step_vel
    return vel.tolist()


def calculate_angular_velocity(angle_sequence: Iterable[float], fps: float) -> list[float]:
    """Backward compatibility wrapper for legacy forward-difference velocity."""

    return angular_velocity(angle_sequence, fps, method="forward")


def calculate_symmetry(angle_left: float, angle_right: float) -> float:
    """Compute symmetry between two angles."""
    if pd.isna(angle_left) or pd.isna(angle_right):
        return np.nan
    max_angle = max(abs(angle_left), abs(angle_right))
    if max_angle == 0:
        return 1.0
    return 1.0 - (abs(angle_left - angle_right) / max_angle)
