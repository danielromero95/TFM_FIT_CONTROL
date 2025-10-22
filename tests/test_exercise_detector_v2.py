from __future__ import annotations

import types
import sys

import numpy as np


def _install_cv2_stub() -> None:
    if "cv2" in sys.modules:
        return

    def _unsupported(*args, **kwargs):  # pragma: no cover - defensive
        raise RuntimeError("OpenCV operations are not available in synthetic tests")

    cv2_stub = types.SimpleNamespace(
        COLOR_BGR2RGB=0,
        CAP_PROP_FPS=0,
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_POS_FRAMES=1,
        VideoCapture=_unsupported,
        cvtColor=_unsupported,
    )
    sys.modules["cv2"] = cv2_stub


_install_cv2_stub()

import src.exercise_detection.exercise_detector as detector
from src.exercise_detection.exercise_detector import FeatureSeries, _segment_reps, classify_features


def _make_feature_series(data: dict[str, np.ndarray], sampling_rate: float) -> FeatureSeries:
    length = 0
    for values in data.values():
        if values is not None:
            arr = np.asarray(values)
            if arr.size > length:
                length = arr.size
    if length == 0:
        raise ValueError("Synthetic feature data must be non-empty")
    normalized = {key: np.asarray(value, dtype=float) for key, value in data.items()}
    return FeatureSeries(
        data=normalized,
        sampling_rate=sampling_rate,
        valid_frames=length,
        total_frames=length,
    )


def _base_template(length: int, value: float) -> np.ndarray:
    return np.full(length, value, dtype=float)


def _timebase(sampling_rate: float, duration_s: float) -> tuple[np.ndarray, int]:
    count = int(sampling_rate * duration_s)
    t = np.arange(count, dtype=float) / sampling_rate
    return t, count


def _squat_series() -> FeatureSeries:
    sr = 30.0
    t, n = _timebase(sr, 12.0)
    freq = 0.45
    phase_shift = 0.1
    knee_left = 150 - 55 * np.sin(2 * np.pi * freq * t)
    knee_right = 148 - 50 * np.sin(2 * np.pi * freq * t + phase_shift)
    hip_left = 170 - 45 * np.sin(2 * np.pi * freq * t)
    hip_right = 172 - 42 * np.sin(2 * np.pi * freq * t + phase_shift / 2)
    elbow_left = 155 + 4 * np.sin(2 * np.pi * freq * t)
    elbow_right = 154 + 4 * np.sin(2 * np.pi * freq * t + phase_shift)
    pelvis_y = 0.5 + 0.06 * (1 - np.cos(2 * np.pi * freq * t))
    torso_tilt = 15 + 4 * np.sin(2 * np.pi * freq * t)

    data = {
        "knee_angle_left": knee_left,
        "knee_angle_right": knee_right,
        "hip_angle_left": hip_left,
        "hip_angle_right": hip_right,
        "elbow_angle_left": elbow_left,
        "elbow_angle_right": elbow_right,
        "shoulder_angle_left": _base_template(n, 130.0),
        "shoulder_angle_right": _base_template(n, 132.0),
        "pelvis_y": pelvis_y,
        "torso_length": _base_template(n, 0.52),
        "wrist_left_x": 0.3 + 0.01 * np.sin(2 * np.pi * freq * t),
        "wrist_right_x": 0.3 + 0.01 * np.sin(2 * np.pi * freq * t + phase_shift),
        "wrist_left_y": 0.4 + 0.02 * np.sin(2 * np.pi * freq * t),
        "wrist_right_y": 0.39 + 0.02 * np.sin(2 * np.pi * freq * t + phase_shift),
        "shoulder_width_norm": _base_template(n, 0.62),
        "ankle_width_norm": _base_template(n, 0.60),
        "shoulder_yaw_deg": _base_template(n, 12.0),
        "shoulder_z_delta_abs": _base_template(n, 0.03),
        "torso_tilt_deg": torso_tilt,
    }
    return _make_feature_series(data, sr)


def _bench_series() -> FeatureSeries:
    sr = 30.0
    t, n = _timebase(sr, 12.0)
    freq = 0.7
    elbow_left = 160 - 75 * np.sin(2 * np.pi * freq * t)
    elbow_right = 158 - 72 * np.sin(2 * np.pi * freq * t + 0.08)
    wrist_x_left = 0.5 + 0.11 * np.sin(2 * np.pi * freq * t)
    wrist_x_right = 0.5 + 0.11 * np.sin(2 * np.pi * freq * t + 0.05)
    wrist_y_left = 0.2 + 0.02 * np.sin(2 * np.pi * freq * t)
    wrist_y_right = 0.2 + 0.02 * np.sin(2 * np.pi * freq * t + 0.05)
    torso_tilt = _base_template(n, 72.0)

    data = {
        "knee_angle_left": _base_template(n, 178.0),
        "knee_angle_right": _base_template(n, 177.0),
        "hip_angle_left": _base_template(n, 175.0),
        "hip_angle_right": _base_template(n, 174.0),
        "elbow_angle_left": elbow_left,
        "elbow_angle_right": elbow_right,
        "shoulder_angle_left": _base_template(n, 120.0),
        "shoulder_angle_right": _base_template(n, 121.0),
        "pelvis_y": _base_template(n, 0.5),
        "torso_length": _base_template(n, 0.50),
        "wrist_left_x": wrist_x_left,
        "wrist_right_x": wrist_x_right,
        "wrist_left_y": wrist_y_left,
        "wrist_right_y": wrist_y_right,
        "shoulder_width_norm": _base_template(n, 0.58),
        "ankle_width_norm": _base_template(n, 0.42),
        "shoulder_yaw_deg": _base_template(n, 18.0),
        "shoulder_z_delta_abs": _base_template(n, 0.02),
        "torso_tilt_deg": torso_tilt,
    }
    return _make_feature_series(data, sr)


def _front_squat_series() -> FeatureSeries:
    sr = 30.0
    t, n = _timebase(sr, 12.0)
    freq = 0.5
    knee_left = 155 - 60 * np.sin(2 * np.pi * freq * t)
    knee_right = 152 - 55 * np.sin(2 * np.pi * freq * t + 0.1)
    hip_left = 168 - 55 * np.sin(2 * np.pi * freq * t)
    hip_right = 170 - 52 * np.sin(2 * np.pi * freq * t + 0.05)
    pelvis_y = 0.52 + 0.085 * (1 - np.cos(2 * np.pi * freq * t))
    torso_tilt = _base_template(n, 20.0)
    wrist_x_left = 0.42 + 0.035 * np.sin(2 * np.pi * freq * t)
    wrist_x_right = 0.41 + 0.035 * np.sin(2 * np.pi * freq * t + 0.08)

    data = {
        "knee_angle_left": knee_left,
        "knee_angle_right": knee_right,
        "hip_angle_left": hip_left,
        "hip_angle_right": hip_right,
        "elbow_angle_left": _base_template(n, 150.0),
        "elbow_angle_right": _base_template(n, 149.0),
        "shoulder_angle_left": _base_template(n, 128.0),
        "shoulder_angle_right": _base_template(n, 129.0),
        "pelvis_y": pelvis_y,
        "torso_length": _base_template(n, 0.50),
        "wrist_left_x": wrist_x_left,
        "wrist_right_x": wrist_x_right,
        "wrist_left_y": 0.44 + 0.02 * np.sin(2 * np.pi * freq * t),
        "wrist_right_y": 0.43 + 0.02 * np.sin(2 * np.pi * freq * t + 0.08),
        "shoulder_width_norm": _base_template(n, 0.64),
        "ankle_width_norm": _base_template(n, 0.59),
        "shoulder_yaw_deg": _base_template(n, 10.0),
        "shoulder_z_delta_abs": _base_template(n, 0.02),
        "torso_tilt_deg": torso_tilt,
    }
    return _make_feature_series(data, sr)


def _deadlift_series() -> FeatureSeries:
    sr = 30.0
    t, n = _timebase(sr, 14.0)
    freq = 0.4
    knee_left = 170 - 12 * np.sin(2 * np.pi * freq * t)
    knee_right = 169 - 11 * np.sin(2 * np.pi * freq * t + 0.06)
    hip_left = 170 - 60 * np.sin(2 * np.pi * freq * t)
    hip_right = 171 - 58 * np.sin(2 * np.pi * freq * t + 0.06)
    elbow_left = _base_template(n, 160.0)
    elbow_right = _base_template(n, 159.0)
    torso_tilt = 32 + 8 * np.sin(2 * np.pi * freq * t)
    pelvis_y = 0.5 + 0.018 * (1 - np.cos(2 * np.pi * freq * t))
    wrist_y_left = 0.5 + 0.09 * np.sin(2 * np.pi * freq * t)
    wrist_y_right = 0.5 + 0.09 * np.sin(2 * np.pi * freq * t + 0.04)

    data = {
        "knee_angle_left": knee_left,
        "knee_angle_right": knee_right,
        "hip_angle_left": hip_left,
        "hip_angle_right": hip_right,
        "elbow_angle_left": elbow_left,
        "elbow_angle_right": elbow_right,
        "shoulder_angle_left": _base_template(n, 128.0),
        "shoulder_angle_right": _base_template(n, 129.0),
        "pelvis_y": pelvis_y,
        "torso_length": _base_template(n, 0.54),
        "wrist_left_x": _base_template(n, 0.32),
        "wrist_right_x": _base_template(n, 0.31),
        "wrist_left_y": wrist_y_left,
        "wrist_right_y": wrist_y_right,
        "shoulder_width_norm": _base_template(n, 0.44),
        "ankle_width_norm": _base_template(n, 0.36),
        "shoulder_yaw_deg": _base_template(n, 32.0),
        "shoulder_z_delta_abs": _base_template(n, 0.07),
        "torso_tilt_deg": torso_tilt,
    }
    return _make_feature_series(data, sr)


def test_synthetic_squat_detection():
    features = _squat_series()
    label, _, confidence = classify_features(features)
    assert label == "squat"
    assert confidence >= 0.70


def test_synthetic_bench_detection():
    features = _bench_series()
    label, _, confidence = classify_features(features)
    assert label == "bench_press"
    assert confidence >= 0.70


def test_synthetic_deadlift_detection():
    features = _deadlift_series()
    label, _, confidence = classify_features(features)
    assert label == "deadlift"
    assert confidence >= 0.70


def test_front_squat_not_misclassified_as_bench_press():
    features = _front_squat_series()
    label, _, confidence = classify_features(features)
    assert label == "squat"
    assert confidence >= 0.60


def test_ambiguous_patterns_yield_unknown():
    squat = _squat_series()
    bench = _bench_series()
    data = {}
    for key in squat.data.keys():
        if key in bench.data:
            min_len = min(squat.data[key].size, bench.data[key].size)
            combined = 0.5 * (squat.data[key][:min_len] + bench.data[key][:min_len])
            data[key] = combined

    length = next(iter(data.values())).size

    for key in (
        "elbow_angle_left",
        "elbow_angle_right",
        "knee_angle_left",
        "knee_angle_right",
        "hip_angle_left",
        "hip_angle_right",
    ):
        series = data[key]
        mean = float(np.mean(series))
        data[key] = 0.5 * (series + mean)

    data["torso_tilt_deg"] = np.full(length, 38.0)
    data["pelvis_y"] = np.full(length, 0.52)
    data["wrist_left_x"] = np.full(length, 0.35)
    data["wrist_right_x"] = np.full(length, 0.34)
    data["wrist_left_y"] = np.full(length, 0.38)
    data["wrist_right_y"] = np.full(length, 0.38)
    data["torso_length"] = np.full(length, 0.51)
    data["ankle_width_norm"] = np.full(length, 0.48)

    sr = squat.sampling_rate
    features = _make_feature_series(data, sr)
    label, _, confidence = classify_features(features)
    assert label == "unknown"
    assert confidence <= 0.55


def test_invalid_torso_length_prevents_false_squat():
    sr = 30.0
    t, n = _timebase(sr, 10.0)
    freq = 0.5
    pelvis_y = 0.45 + 0.14 * (1 - np.cos(2 * np.pi * freq * t))
    static_series = _base_template(n, 170.0)
    elbow_series = _base_template(n, 155.0)
    base_data = {
        "knee_angle_left": static_series,
        "knee_angle_right": static_series,
        "hip_angle_left": static_series,
        "hip_angle_right": static_series,
        "elbow_angle_left": elbow_series,
        "elbow_angle_right": elbow_series,
        "pelvis_y": pelvis_y,
        "torso_tilt_deg": _base_template(n, 10.0),
        "shoulder_width_norm": _base_template(n, 0.6),
        "shoulder_yaw_deg": _base_template(n, 12.0),
        "shoulder_z_delta_abs": _base_template(n, 0.03),
        "wrist_left_x": _base_template(n, 0.3),
        "wrist_right_x": _base_template(n, 0.3),
        "wrist_left_y": _base_template(n, 0.4),
        "wrist_right_y": _base_template(n, 0.4),
        "ankle_width_norm": _base_template(n, 0.55),
    }

    for torso_values in (np.zeros(n, dtype=float), np.full(n, np.nan)):
        data = dict(base_data)
        data["torso_length"] = torso_values
        features = _make_feature_series(data, sr)
        label, _, confidence = classify_features(features)
        assert label == "unknown"
        assert confidence <= 0.55


def test_ankle_fallback_guides_view(monkeypatch):
    nan_series = np.full(12, np.nan)

    monkeypatch.setattr(detector, "_pick_view_label", lambda scores, evidence: "unknown")

    ankle_front = np.full(12, 0.65)
    view_front = detector._classify_view(nan_series, nan_series, nan_series, ankle_front)
    assert view_front == "front"

    ankle_side = np.full(12, 0.38)
    view_side = detector._classify_view(nan_series, nan_series, nan_series, ankle_side)
    assert view_side == "side"


def test_segment_reps_flat_activity_returns_empty():
    sr = 30.0
    length = 300
    constant = _base_template(length, 170.0)
    data = {
        "knee_angle_left": constant,
        "knee_angle_right": constant,
        "hip_angle_left": constant,
        "hip_angle_right": constant,
        "elbow_angle_left": constant,
        "elbow_angle_right": constant,
        "pelvis_y": _base_template(length, 0.5),
        "torso_length": _base_template(length, 0.5),
    }
    features = _make_feature_series(data, sr)
    reps = _segment_reps(features)
    assert reps == []
