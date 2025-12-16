import numpy as np

from src.exercise_detection.classification import classify_features
from src.exercise_detection.types import FeatureSeries


def _squat_like_series(length: int) -> FeatureSeries:
    frames = np.arange(length)
    # Two clear knee flexion cycles
    single_rep = np.concatenate(
        [
            np.linspace(170, 90, 10),
            np.linspace(90, 170, 10),
        ]
    )
    knee = np.tile(single_rep, max(1, length // single_rep.size + 1))[:length]
    hip = 150 - 0.4 * (knee - 120)  # hip hinges but stays softer than knee
    torso = np.linspace(10, 50, length)  # noisy torso tilt trying to emulate hinge

    shoulder_width = np.full(length, 0.3)
    yaw = np.full(length, 5.0)

    coords = {
        "knee_angle_left": knee,
        "knee_angle_right": knee,
        "hip_angle_left": hip,
        "hip_angle_right": hip,
        "elbow_angle_left": np.full(length, 165.0),
        "elbow_angle_right": np.full(length, 165.0),
        "torso_tilt_deg": torso,
        "torso_length": np.full(length, 1.0),
        "torso_length_world": np.full(length, 1.0),
        "wrist_left_x": np.zeros(length),
        "wrist_left_y": np.linspace(0.2, 0.05, length),
        "wrist_right_x": np.zeros(length),
        "wrist_right_y": np.linspace(0.2, 0.05, length),
        "pelvis_y": np.linspace(0.4, 0.1, length),
        "shoulder_width_norm": shoulder_width,
        "shoulder_yaw_deg": yaw,
        "shoulder_z_delta_abs": np.zeros(length),
        "ankle_width_norm": np.full(length, 0.25),
        "shoulder_left_y": np.zeros(length),
        "shoulder_right_y": np.zeros(length),
        "hip_left_y": np.linspace(0.3, 0.1, length),
        "hip_right_y": np.linspace(0.3, 0.1, length),
        "knee_left_x": np.full(length, 0.2),
        "knee_left_y": np.linspace(0.2, -0.1, length),
        "knee_right_x": np.full(length, 0.2),
        "knee_right_y": np.linspace(0.2, -0.1, length),
        "ankle_left_x": np.full(length, 0.05),
        "ankle_left_y": np.linspace(0.0, -0.2, length),
        "ankle_right_x": np.full(length, 0.05),
        "ankle_right_y": np.linspace(0.0, -0.2, length),
    }

    return FeatureSeries(
        data=coords,
        sampling_rate=30.0,
        valid_frames=length,
        total_frames=length,
        view_reliability={"frame_reliability": np.ones(length, dtype=bool)},
    )


def test_front_view_squat_overrides_deadlift_noise():
    features = _squat_like_series(60)

    label, view, confidence, metadata = classify_features(features, return_metadata=True)

    assert view == "front"
    assert label == "squat"
    scores = metadata.get("classification_scores", {})
    assert scores.get("deadlift_veto") is True
    assert confidence > 0.0
