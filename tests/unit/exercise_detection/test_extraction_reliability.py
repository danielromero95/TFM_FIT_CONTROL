from __future__ import annotations

from src.exercise_detection.constants import RELIABILITY_VIS_THRESHOLD
from src.exercise_detection.extraction import _evaluate_view_reliability


def _lm(vis: float) -> dict[str, float]:
    return {"x": 0.1, "y": 0.2, "z": 0.05, "visibility": vis}


def test_reliability_accepts_three_visible_joints() -> None:
    """Frames with a single occluded hip/shoulder still count as reliable."""

    landmarks = [{} for _ in range(33)]
    landmarks[11] = _lm(RELIABILITY_VIS_THRESHOLD + 0.05)  # left shoulder
    landmarks[12] = _lm(RELIABILITY_VIS_THRESHOLD + 0.10)  # right shoulder
    landmarks[23] = _lm(RELIABILITY_VIS_THRESHOLD + 0.08)  # left hip
    landmarks[24] = _lm(RELIABILITY_VIS_THRESHOLD - 0.20)  # right hip occluded

    reliable, _ = _evaluate_view_reliability(landmarks)

    assert reliable is True
