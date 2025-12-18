import numpy as np

from src.exercise_detection.classification import classify_features
from src.exercise_detection.types import FeatureSeries


def _squat_like_series(length: int) -> dict[str, np.ndarray]:
    base = {
        "torso_length": np.full(length, 1.0),
        "torso_length_world": np.full(length, 1.0),
        "shoulder_width_norm": np.full(length, 0.62),
        "shoulder_yaw_deg": np.full(length, 8.0),
        "shoulder_z_delta_abs": np.full(length, 0.01),
        "ankle_width_norm": np.full(length, 0.55),
        "shoulder_left_y": np.full(length, 0.32),
        "shoulder_right_y": np.full(length, 0.32),
        "hip_left_y": np.linspace(0.48, 0.30, length),
        "hip_right_y": np.linspace(0.48, 0.30, length),
        "knee_left_x": np.full(length, 0.16),
        "knee_right_x": np.full(length, 0.16),
        "knee_left_y": np.linspace(0.38, 0.12, length),
        "knee_right_y": np.linspace(0.38, 0.12, length),
        "ankle_left_x": np.full(length, 0.05),
        "ankle_right_x": np.full(length, 0.05),
        "ankle_left_y": np.linspace(0.10, -0.04, length),
        "ankle_right_y": np.linspace(0.10, -0.04, length),
    }

    knee_pattern = np.concatenate([
        np.linspace(172, 92, 12),
        np.linspace(92, 175, 12),
    ])
    knee = np.tile(knee_pattern, max(1, length // knee_pattern.size + 1))[:length]
    hip = 158 - 0.5 * (knee - 115)
    torso = np.linspace(14, 38, length)

    base.update(
        {
            "knee_angle_left": knee,
            "knee_angle_right": knee,
            "hip_angle_left": hip,
            "hip_angle_right": hip,
            "elbow_angle_left": np.full(length, 132.0),
            "elbow_angle_right": np.full(length, 132.0),
            "torso_tilt_deg": torso,
            "wrist_left_y": np.linspace(0.34, 0.29, length),
            "wrist_right_y": np.linspace(0.34, 0.29, length),
            "wrist_left_x": np.zeros(length),
            "wrist_right_x": np.zeros(length),
            "pelvis_y": np.linspace(0.46, 0.30, length),
        }
    )

    return base


def test_reliability_mask_keeps_pose_when_marked_unreliable():
    length = 72
    coords = _squat_like_series(length)

    features = FeatureSeries(
        data=coords,
        sampling_rate=30.0,
        valid_frames=length,
        total_frames=length,
        view_reliability={"frame_reliability": np.zeros(length, dtype=bool)},
    )

    label, view, confidence, metadata = classify_features(features, return_metadata=True)

    assert metadata.get("rep_count", 0) > 0
    assert label == "squat"
    assert view in {"front", "unknown"}
    assert confidence <= 1.0
