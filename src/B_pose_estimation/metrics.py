"""Pure functions for computing biomechanical metrics."""

from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd


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
    return {
        "left_knee": calculate_angle(landmarks[23], landmarks[25], landmarks[27]),
        "right_knee": calculate_angle(landmarks[24], landmarks[26], landmarks[28]),
        "left_elbow": calculate_angle(landmarks[11], landmarks[13], landmarks[15]),
        "right_elbow": calculate_angle(landmarks[12], landmarks[14], landmarks[16]),
    }


def calculate_distances(landmarks: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Return key distance measurements."""
    return {
        "shoulder_width": abs(landmarks[12]["x"] - landmarks[11]["x"]),
        "foot_separation": abs(landmarks[28]["x"] - landmarks[27]["x"]),
    }


def calculate_angular_velocity(angle_sequence: Iterable[float], fps: float) -> list[float]:
    """Calculate the angular velocity for a sequence of angles."""
    angle_sequence = list(angle_sequence)
    if not angle_sequence or fps == 0:
        return [0.0] * len(angle_sequence)
    velocities = [0.0]
    dt = 1.0 / fps
    for i in range(1, len(angle_sequence)):
        velocities.append(abs(angle_sequence[i] - angle_sequence[i - 1]) / dt)
    return velocities


def calculate_symmetry(angle_left: float, angle_right: float) -> float:
    """Compute symmetry between two angles."""
    if pd.isna(angle_left) or pd.isna(angle_right):
        return np.nan
    max_angle = max(abs(angle_left), abs(angle_right))
    if max_angle == 0:
        return 1.0
    return 1.0 - (abs(angle_left - angle_right) / max_angle)
