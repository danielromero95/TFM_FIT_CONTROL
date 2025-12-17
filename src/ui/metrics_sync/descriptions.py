"""Helper text for explaining metrics and primary counting angles."""

from __future__ import annotations

from typing import Dict


_PRIMARY_ANGLE_EXPLANATIONS: Dict[str, str] = {
    "left_knee": "Left knee flexion: angle between the left femur and tibia at the knee joint.",
    "right_knee": "Right knee flexion: angle between the right femur and tibia at the knee joint.",
    "left_hip": "Left hip hinge: angle formed by the torso, hip, and knee on the left side.",
    "right_hip": "Right hip hinge: angle formed by the torso, hip, and knee on the right side.",
    "hips_mean": "Average of left and right hip hinge angles to capture bilateral deadlift motion.",
    "left_elbow": "Left elbow flexion: angle between the upper arm (humerus) and forearm.",
    "right_elbow": "Right elbow flexion: angle between the upper arm (humerus) and forearm.",
}


def describe_primary_angle(metric: str | None) -> str | None:
    """Return a plain-language explanation of the primary counting metric."""

    if metric is None:
        return None
    return _PRIMARY_ANGLE_EXPLANATIONS.get(metric)

