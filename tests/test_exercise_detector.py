"""Unit tests for the heuristic exercise classifier."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.detect.exercise_detector import FeatureSeries, classify_features


def _base_feature_data(length: int) -> dict[str, np.ndarray]:
    const_angle = np.full(length, 90.0)
    const_coord = np.full(length, 0.5)
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
        "wrist_left_x": const_coord.copy(),
        "wrist_left_y": const_coord.copy(),
        "wrist_right_x": const_coord.copy(),
        "wrist_right_y": const_coord.copy(),
        "shoulder_width_norm": np.full(length, 0.6),
        "shoulder_yaw_deg": np.full(length, np.nan),
        "shoulder_z_delta_abs": np.full(length, np.nan),
        "torso_tilt_deg": np.full(length, 15.0),
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

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "squat"
    assert view == "front"
    assert confidence >= 0.6


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

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "bench"
    assert view == "side"
    assert confidence >= 0.6


def test_classify_deadlift_like_features() -> None:
    length = 40
    data = _base_feature_data(length)
    data["hip_angle_left"] = np.linspace(30.0, 115.0, length)
    data["hip_angle_right"] = np.linspace(32.0, 118.0, length)
    data["wrist_left_y"] = np.linspace(0.2, 0.55, length)
    data["wrist_right_y"] = np.linspace(0.25, 0.58, length)
    data["torso_tilt_deg"] = np.full(length, 20.0)
    data["shoulder_width_norm"] = np.full(length, 0.45)
    data["shoulder_yaw_deg"] = np.full(length, 35.0)
    data["shoulder_z_delta_abs"] = np.full(length, 0.2)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "deadlift"
    assert view == "side"
    assert confidence >= 0.6


def test_classify_low_signal_returns_unknown() -> None:
    length = 30
    data = _base_feature_data(length)
    data["shoulder_width_norm"] = np.full(length, np.nan)
    data["shoulder_yaw_deg"] = np.full(length, np.nan)
    data["shoulder_z_delta_abs"] = np.full(length, np.nan)

    features = _make_feature_series(data)
    label, view, confidence = classify_features(features)

    assert label == "unknown"
    assert view == "unknown"
    assert confidence == 0.0
