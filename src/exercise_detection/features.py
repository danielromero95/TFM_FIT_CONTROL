"""Conversión de *landmarks* de MediaPipe a características numéricas.

Estos helpers encapsulan la lógica geométrica para mantenerla separada de la
lectura de vídeo y del clasificador.  Las funciones devuelven siempre diccionarios
planos para facilitar su serialización durante las pruebas.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

# Índices constantes de MediaPipe Pose; mantener aquí evita literales mágicos.
_POSE_LANDMARK_INDEX = {
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
}


def build_features_from_landmark_array(
    landmarks_arr: "np.ndarray",
    *,
    world_landmarks_arr: "np.ndarray | None" = None,
) -> Dict[str, float]:
    """Camino rápido: aceptar arrays (33,4)/(33,3) y calcular rasgos geométricos."""

    arr = np.asarray(landmarks_arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 33 or arr.shape[1] < 3:
        raise ValueError(f"Expected landmark array with shape (33,3/4), got {arr.shape}")
    coords = arr[:33, :3].copy()
    coords[~np.isfinite(coords)] = float("nan")
    world_coords: "np.ndarray | None" = None
    if world_landmarks_arr is not None:
        world_coords = np.asarray(world_landmarks_arr, dtype=float)
        if world_coords.ndim != 2 or world_coords.shape[0] < 33 or world_coords.shape[1] < 3:
            world_coords = None

    return _features_from_coords(coords, world_coords=world_coords)


def build_features_from_landmarks(
    landmarks: list[dict[str, float]],
    *,
    world_landmarks: Optional[list[dict[str, float]]] = None,
) -> Dict[str, float]:
    """Generar las características de detección a partir de un listado de landmarks."""

    if len(landmarks) < 33:
        raise ValueError(f"Expected 33 pose landmarks, received {len(landmarks)}")

    coords = np.empty((33, 3), dtype=float)

    def _safe_float(value: Any, default: float = float("nan")) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    for i in range(33):
        entry = landmarks[i] if 0 <= i < len(landmarks) else None
        coords[i, 0] = _safe_float(entry.get("x") if entry else float("nan"))
        coords[i, 1] = _safe_float(entry.get("y") if entry else float("nan"))
        coords[i, 2] = _safe_float(entry.get("z") if entry else float("nan"))

    world_coords: "np.ndarray | None" = None
    if world_landmarks:
        world_coords = np.empty((33, 3), dtype=float)

        for i in range(33):
            entry = world_landmarks[i] if 0 <= i < len(world_landmarks) else None
            world_coords[i, 0] = _safe_float(entry.get("x") if entry else float("nan"))
            world_coords[i, 1] = _safe_float(entry.get("y") if entry else float("nan"))
            world_coords[i, 2] = _safe_float(entry.get("z") if entry else float("nan"))

    return _features_from_coords(coords, world_coords=world_coords)


def _features_from_coords(
    coords: "np.ndarray",
    *,
    world_coords: "np.ndarray | None" = None,
) -> Dict[str, float]:
    """Calcular características geométricas a partir de landmarks 2D/3D."""

    if coords is None or not isinstance(coords, np.ndarray) or coords.shape[0] < 33 or coords.shape[1] < 3:
        raise ValueError("Expected coords with shape (33,3)")

    coords = np.asarray(coords, dtype=float)

    world = None
    if world_coords is not None:
        world = np.asarray(world_coords, dtype=float)
        if world.ndim != 2 or world.shape[0] < 33 or world.shape[1] < 3:
            world = None

    def _coord(name: str) -> np.ndarray:
        return coords[_POSE_LANDMARK_INDEX[name]]

    def _world_coord(name: str) -> Optional[np.ndarray]:
        if world is None:
            return None
        return world[_POSE_LANDMARK_INDEX[name]]

    left_hip = _coord("LEFT_HIP")
    right_hip = _coord("RIGHT_HIP")
    left_knee = _coord("LEFT_KNEE")
    right_knee = _coord("RIGHT_KNEE")
    left_ankle = _coord("LEFT_ANKLE")
    right_ankle = _coord("RIGHT_ANKLE")
    left_shoulder = _coord("LEFT_SHOULDER")
    right_shoulder = _coord("RIGHT_SHOULDER")
    left_elbow = _coord("LEFT_ELBOW")
    right_elbow = _coord("RIGHT_ELBOW")
    left_wrist = _coord("LEFT_WRIST")
    right_wrist = _coord("RIGHT_WRIST")

    hip_mid = (left_hip + right_hip) / 2.0
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0

    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    torso_left = float(np.linalg.norm(left_shoulder[:2] - left_hip[:2]))
    torso_right = float(np.linalg.norm(right_shoulder[:2] - right_hip[:2]))
    torso_length = (torso_left + torso_right) / 2.0

    norm_width = shoulder_width / (torso_length + 1e-6)

    ankle_width = float(np.linalg.norm(left_ankle[:2] - right_ankle[:2]))
    ankle_width_norm = ankle_width / (torso_length + 1e-6)

    dx = abs(float(left_shoulder[0] - right_shoulder[0]))
    dz = abs(float(left_shoulder[2] - right_shoulder[2]))
    yaw_deg = math.degrees(math.atan2(dz, dx + 1e-6))

    torso_vec = shoulder_mid - hip_mid
    tilt = math.degrees(math.atan2(abs(torso_vec[0]), abs(torso_vec[1]) + 1e-6))

    def _torso_length_world() -> float:
        left_shoulder_world = _world_coord("LEFT_SHOULDER")
        right_shoulder_world = _world_coord("RIGHT_SHOULDER")
        left_hip_world = _world_coord("LEFT_HIP")
        right_hip_world = _world_coord("RIGHT_HIP")

        if (
            left_shoulder_world is None
            or right_shoulder_world is None
            or left_hip_world is None
            or right_hip_world is None
        ):
            return float("nan")

        if (
            np.any(~np.isfinite(left_shoulder_world))
            or np.any(~np.isfinite(right_shoulder_world))
            or np.any(~np.isfinite(left_hip_world))
            or np.any(~np.isfinite(right_hip_world))
        ):
            return float("nan")

        left_len = float(np.linalg.norm(left_shoulder_world - left_hip_world))
        right_len = float(np.linalg.norm(right_shoulder_world - right_hip_world))
        return (left_len + right_len) / 2.0

    torso_length_world = _torso_length_world()

    features = {
        "knee_angle_left": _angle_degrees(left_hip, left_knee, left_ankle),
        "knee_angle_right": _angle_degrees(right_hip, right_knee, right_ankle),
        "hip_angle_left": _angle_degrees(left_shoulder, left_hip, left_knee),
        "hip_angle_right": _angle_degrees(right_shoulder, right_hip, right_knee),
        "elbow_angle_left": _angle_degrees(left_shoulder, left_elbow, left_wrist),
        "elbow_angle_right": _angle_degrees(right_shoulder, right_elbow, right_wrist),
        "shoulder_angle_left": _angle_degrees(left_hip, left_shoulder, left_elbow),
        "shoulder_angle_right": _angle_degrees(right_hip, right_shoulder, right_elbow),
        "pelvis_y": float((left_hip[1] + right_hip[1]) / 2.0),
        "torso_length": float(torso_length),
        "torso_length_world": float(torso_length_world),
        "wrist_left_x": float(left_wrist[0]),
        "wrist_left_y": float(left_wrist[1]),
        "wrist_right_x": float(right_wrist[0]),
        "wrist_right_y": float(right_wrist[1]),
        "shoulder_width_norm": float(norm_width),
        "shoulder_yaw_deg": float(yaw_deg),
        "shoulder_z_delta_abs": float(dz),
        "torso_tilt_deg": float(tilt),
        "ankle_width_norm": float(ankle_width_norm),
        "shoulder_left_x": float(left_shoulder[0]),
        "shoulder_left_y": float(left_shoulder[1]),
        "shoulder_right_x": float(right_shoulder[0]),
        "shoulder_right_y": float(right_shoulder[1]),
        "hip_left_x": float(left_hip[0]),
        "hip_left_y": float(left_hip[1]),
        "hip_right_x": float(right_hip[0]),
        "hip_right_y": float(right_hip[1]),
        "knee_left_x": float(left_knee[0]),
        "knee_left_y": float(left_knee[1]),
        "knee_right_x": float(right_knee[0]),
        "knee_right_y": float(right_knee[1]),
        "ankle_left_x": float(left_ankle[0]),
        "ankle_left_y": float(left_ankle[1]),
        "ankle_right_x": float(right_ankle[0]),
        "ankle_right_y": float(right_ankle[1]),
        "elbow_left_x": float(left_elbow[0]),
        "elbow_left_y": float(left_elbow[1]),
        "elbow_right_x": float(right_elbow[0]),
        "elbow_right_y": float(right_elbow[1]),
    }

    return features


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return _vector_angle(a - b, c - b)


def _vector_angle(u: np.ndarray, v: np.ndarray) -> float:
    if u is None or v is None:
        return float("nan")
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape[-1] < 3 or v.shape[-1] < 3:
        return float("nan")
    if np.any(~np.isfinite(u)) or np.any(~np.isfinite(v)):
        return float("nan")
    dot = float(np.dot(u, v))
    norm_product = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    cos_angle = np.clip(dot / norm_product, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))
