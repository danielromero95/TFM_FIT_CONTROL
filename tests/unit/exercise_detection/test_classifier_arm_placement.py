import numpy as np

from src.exercise_detection.classification import classify_features
from src.exercise_detection.types import FeatureSeries


def _base_front_series(length: int) -> dict:
    return {
        "torso_length": np.full(length, 1.0),
        "torso_length_world": np.full(length, 1.0),
        "shoulder_width_norm": np.full(length, 0.62),
        "shoulder_yaw_deg": np.full(length, 8.0),
        "shoulder_z_delta_abs": np.full(length, 0.01),
        "ankle_width_norm": np.full(length, 0.55),
        "shoulder_left_y": np.full(length, 0.3),
        "shoulder_right_y": np.full(length, 0.3),
        "hip_left_y": np.linspace(0.45, 0.30, length),
        "hip_right_y": np.linspace(0.45, 0.30, length),
        "knee_left_x": np.full(length, 0.18),
        "knee_right_x": np.full(length, 0.18),
        "knee_left_y": np.linspace(0.35, 0.10, length),
        "knee_right_y": np.linspace(0.35, 0.10, length),
        "ankle_left_x": np.full(length, 0.05),
        "ankle_right_x": np.full(length, 0.05),
        "ankle_left_y": np.linspace(0.1, -0.05, length),
        "ankle_right_y": np.linspace(0.1, -0.05, length),
    }


def _feature_series_from(coords: dict) -> FeatureSeries:
    length = len(next(iter(coords.values())))
    return FeatureSeries(
        data=coords,
        sampling_rate=30.0,
        valid_frames=length,
        total_frames=length,
        view_reliability={"frame_reliability": np.ones(length, dtype=bool)},
    )


def test_front_squat_bar_high_blocks_deadlift():
    length = 80
    coords = _base_front_series(length)

    knee_pattern = np.concatenate([
        np.linspace(170, 90, 12),
        np.linspace(90, 175, 12),
    ])
    knee = np.tile(knee_pattern, max(1, length // knee_pattern.size + 1))[:length]
    hip = 155 - 0.5 * (knee - 115)
    torso = np.linspace(15, 45, length)

    coords.update(
        {
            "knee_angle_left": knee,
            "knee_angle_right": knee,
            "hip_angle_left": hip,
            "hip_angle_right": hip,
            "elbow_angle_left": np.full(length, 125.0),
            "elbow_angle_right": np.full(length, 125.0),
            "torso_tilt_deg": torso,
            "wrist_left_y": np.linspace(0.32, 0.28, length),
            "wrist_right_y": np.linspace(0.32, 0.28, length),
            "wrist_left_x": np.zeros(length),
            "wrist_right_x": np.zeros(length),
            "pelvis_y": np.linspace(0.42, 0.28, length),
        }
    )

    features = _feature_series_from(coords)
    label, view, _, metadata = classify_features(features, return_metadata=True)

    assert view == "front"
    assert label == "squat"
    scores = metadata.get("classification_scores", {})
    assert scores.get("deadlift_veto") is True


def test_front_deadlift_requires_bar_low_and_extension():
    length = 80
    coords = _base_front_series(length)

    knee = np.linspace(165, 135, length)
    hip = np.linspace(175, 120, length)
    torso = np.linspace(45, 80, length)

    coords.update(
        {
            "knee_angle_left": knee,
            "knee_angle_right": knee,
            "hip_angle_left": hip,
            "hip_angle_right": hip,
            "elbow_angle_left": np.full(length, 176.0),
            "elbow_angle_right": np.full(length, 176.0),
            "torso_tilt_deg": torso,
            "wrist_left_y": np.linspace(0.58, 0.52, length),
            "wrist_right_y": np.linspace(0.58, 0.52, length),
            "wrist_left_x": np.zeros(length),
            "wrist_right_x": np.zeros(length),
            "pelvis_y": np.linspace(0.52, 0.40, length),
        }
    )

    features = _feature_series_from(coords)
    label, view, _, metadata = classify_features(features, return_metadata=True)

    assert view in {"front", "unknown"}
    assert label == "deadlift"
    scores = metadata.get("classification_scores", {})
    assert scores.get("deadlift_veto") is False
