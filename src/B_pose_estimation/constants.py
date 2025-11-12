"""Constantes compartidas de *landmarks* empleadas por la estimación de pose."""

from __future__ import annotations

from typing import Dict, Tuple

LANDMARK_COUNT: int = 33

# Atajos de índice que replican el orden de landmarks de Mediapipe.
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

HIP_CENTER = (LEFT_HIP, RIGHT_HIP)
SHOULDER_CENTER = (LEFT_SHOULDER, RIGHT_SHOULDER)

JOINT_INDEX_MAP: Dict[str, Tuple[int, int, int]] = {
    "left_knee": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right_knee": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "left_elbow": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right_elbow": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "left_hip": (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "right_hip": (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
}

DISTANCE_PAIRS: Dict[str, Tuple[int, int]] = {
    "shoulder_width": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "foot_separation": (LEFT_ANKLE, RIGHT_ANKLE),
}

__all__ = [
    "LANDMARK_COUNT",
    "JOINT_INDEX_MAP",
    "DISTANCE_PAIRS",
    "HIP_CENTER",
    "SHOULDER_CENTER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
]
