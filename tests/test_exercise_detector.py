# tests/test_exercise_detector.py
"""Unit tests for the heuristic exercise classifier."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from exercise_detection import FeatureSeries, classify_features


def _base_feature_data(length: int) -> dict[str, np.ndarray]:
    const_angle = np.full(length, 90.0)
    const_coord = np.full(length, 0.5)
    left_x = np.full(length, 0.4)
    right_x = np.full(length, 0.6)
    hip_y = np.full(length, 0.65)
    knee_y = np.full(length, 0.8)
    ankle_y = np.full(length, 0.92)
    return {
        "knee_angle_left": const_angle.copy(),
        "knee_angle_right": const_angle.copy(),
        "hip_angle_left": const_angle.copy(),
        "hip_angle_right": const_angle.copy(),
        "elbow_angle_left": const_angle.copy(),
        "elbow_angle_right": const_angle.copy(),
        "shoulder_angle_left": const_angle.copy(),
        "shoulder_angle_right": const_angle.copy(),
        "pelvis_y": const_coord.copy(),
        "torso_length": np.full(length, 0.52),
        "torso_length_world": np.full(length, 0.52),
        "wrist_left_x": const_coord.copy(),
        "wrist_left_y": const_coord.copy(),
        "wrist_right_x": const_coord.copy(),
        "wrist_right_y": const_coord.copy(),
        "shoulder_width_norm": np.full(length, 0.6),
        "shoulder_yaw_deg": np.full(length, np.nan),
        "shoulder_z_delta_abs": np.full(length, np.nan),
        "torso_tilt_deg": np.full(length, 15.0),
        "shoulder_left_x": left_x.copy(),
        "shoulder_left_y": np.full(length, 0.45),
        "shoulder_right_x": right_x.copy(),
        "shoulder_right_y": np.full(length, 0.45),
        "hip_left_x": np.full(length, 0.42),
        "hip_left_y": hip_y.copy(),
        "hip_right_x": np.full(length, 0.58),
        "hip_right_y": hip_y.copy(),
        "knee_left_x": np.full(length, 0.44),
        "knee_left_y": knee_y.copy(),
        "knee_right_x": np.full(length, 0.56),
        "knee_right_y": knee_y.copy(),
        "ankle_left_x": np.full(length, 0.46),
        "ankle_left_y": ankle_y.copy(),
        "ankle_right_x": np.full(length, 0.54),
        "ankle_right_y": ankle_y.copy(),
        "elbow_left_x": np.full(length, 0.38),
        "elbow_left_y": np.full(length, 0.52),
        "elbow_right_x": np.full(length, 0.62),
        "elbow_right_y": np.full(length, 0.52),
    }


def _make_feature_series(data: dict[str, np.ndarray]) -> FeatureSeries:
    length = len(next(iter(data.values())))
    return FeatureSeries(
        data=data,
        sampling_rate=30.0,
        valid_frames=length,
        total_frames=length,
    )


def test_classify_squat_like_features() -> None:
    length = 40
    data = _base_feature_data(length)
    data["knee_angle_left"] = np.linspace(80.0, 150.0, length)
    data["knee_angle_right"] = np.linspace(82.0, 148.0, length)
    data["hip_angle_left"] = np.linspace(60.0, 110.0, length)
    data["hip_angle_right"] = np.linspace(58.0, 108.0, length)
    data["pelvis_y"] = np.linspace(0.45, 0.60, length)
    data["shoulder_width_norm"] = np.full(length, 0.65)
    data["shoulder_yaw_deg"] = np.full(length, 8.0)
    data["shoulder_z_delta_abs"] = np.full(length, 0.01)
    data["torso_tilt_deg"] = np.full(length, 12.0)
    data["knee_left_x"] = np.full(length, 0.56)
    data["knee_right_x"] = np.full(length, 0.44)
    data["ankle_left_x"] = np.full(length, 0.46)
    data["ankle_right_x"] = np.full(length, 0.54)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "squat"
    assert view == "front"
    assert confidence >= 0.6


def test_classify_squat_like_features_side_view() -> None:
    length = 48
    data = _base_feature_data(length)
    data["knee_angle_left"] = np.linspace(85.0, 155.0, length)
    data["knee_angle_right"] = np.linspace(87.0, 153.0, length)
    data["hip_angle_left"] = np.linspace(65.0, 120.0, length)
    data["hip_angle_right"] = np.linspace(63.0, 118.0, length)
    data["pelvis_y"] = np.linspace(0.46, 0.64, length)
    data["shoulder_width_norm"] = np.full(length, 0.38)
    data["shoulder_yaw_deg"] = np.full(length, 38.0)
    data["shoulder_z_delta_abs"] = np.full(length, 0.16)
    data["torso_tilt_deg"] = np.full(length, 18.0)
    data["knee_left_x"] = np.full(length, 0.58)
    data["ankle_left_x"] = np.full(length, 0.46)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "squat"
    assert view == "side"
    assert confidence >= 0.55


def test_classify_bench_like_features() -> None:
    length = 40
    data = _base_feature_data(length)
    data["elbow_angle_left"] = np.linspace(40.0, 150.0, length)
    data["elbow_angle_right"] = np.linspace(38.0, 148.0, length)
    data["wrist_left_x"] = np.linspace(0.2, 0.5, length)
    data["wrist_right_x"] = np.linspace(0.25, 0.55, length)
    data["torso_tilt_deg"] = np.full(length, 75.0)
    data["shoulder_width_norm"] = np.full(length, 0.4)
    data["shoulder_yaw_deg"] = np.full(length, 35.0)
    data["shoulder_z_delta_abs"] = np.full(length, 0.2)
    data["wrist_left_y"] = np.linspace(0.35, 0.45, length)
    data["wrist_right_y"] = np.linspace(0.35, 0.45, length)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "bench_press"
    assert view == "side"
    assert confidence >= 0.6


def test_classify_deadlift_like_features() -> None:
    length = 40
    data = _base_feature_data(length)
    data["knee_angle_left"] = np.full(length, 165.0)
    data["knee_angle_right"] = np.full(length, 164.0)
    data["hip_angle_left"] = np.linspace(30.0, 115.0, length)
    data["hip_angle_right"] = np.linspace(32.0, 118.0, length)
    data["elbow_angle_left"] = np.full(length, 175.0)
    data["elbow_angle_right"] = np.full(length, 174.0)
    data["wrist_left_y"] = np.linspace(0.75, 0.92, length)
    data["wrist_right_y"] = np.linspace(0.78, 0.95, length)
    data["torso_tilt_deg"] = np.full(length, 42.0)
    data["shoulder_width_norm"] = np.full(length, 0.45)
    data["shoulder_yaw_deg"] = np.full(length, 35.0)
    data["shoulder_z_delta_abs"] = np.full(length, 0.2)
    data["knee_left_x"] = np.full(length, 0.47)
    data["ankle_left_x"] = np.full(length, 0.46)
    data["knee_left_y"] = np.linspace(0.78, 0.82, length)
    data["ankle_left_y"] = np.linspace(0.9, 0.94, length)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "deadlift"
    assert view == "side"
    assert confidence >= 0.6


def test_classify_deadlift_front_view_features() -> None:
    length = 48
    data = _base_feature_data(length)
    data["knee_angle_left"] = np.linspace(150.0, 175.0, length)
    data["knee_angle_right"] = np.linspace(149.0, 174.0, length)
    data["hip_angle_left"] = np.linspace(35.0, 115.0, length)
    data["hip_angle_right"] = np.linspace(37.0, 118.0, length)
    data["elbow_angle_left"] = np.full(length, 176.0)
    data["elbow_angle_right"] = np.full(length, 175.0)
    data["wrist_left_y"] = np.linspace(0.72, 0.92, length)
    data["wrist_right_y"] = np.linspace(0.73, 0.93, length)
    data["torso_tilt_deg"] = np.linspace(28.0, 46.0, length)
    data["shoulder_width_norm"] = np.full(length, 0.58)
    data["shoulder_yaw_deg"] = np.full(length, 6.0)
    data["shoulder_z_delta_abs"] = np.full(length, 0.012)
    data["ankle_left_x"] = np.linspace(0.47, 0.48, length)
    data["ankle_right_x"] = np.linspace(0.52, 0.53, length)
    data["knee_left_x"] = np.linspace(0.46, 0.47, length)
    data["knee_right_x"] = np.linspace(0.53, 0.54, length)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "deadlift"
    assert view == "front"
    assert confidence >= 0.5


def test_classify_low_signal_returns_unknown() -> None:
    length = 30
    data = _base_feature_data(length)
    data["shoulder_width_norm"] = np.full(length, np.nan)
    data["shoulder_yaw_deg"] = np.full(length, np.nan)
    data["shoulder_z_delta_abs"] = np.full(length, np.nan)
    data["knee_angle_left"] = np.full(length, 118.0)
    data["knee_angle_right"] = np.full(length, 118.0)
    data["hip_angle_left"] = np.full(length, 178.0)
    data["hip_angle_right"] = np.full(length, 178.0)
    data["elbow_angle_left"] = np.full(length, 170.0)
    data["elbow_angle_right"] = np.full(length, 170.0)
    data["wrist_left_y"] = np.full(length, 0.45)
    data["wrist_right_y"] = np.full(length, 0.45)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "unknown"
    assert view == "unknown"
    assert confidence == 0.0
