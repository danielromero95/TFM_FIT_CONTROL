"""Synthetic tests for the refactored exercise classifier."""

from __future__ import annotations

import numpy as np

from exercise_detection import classify_features
from exercise_detection.classifier import classify_exercise
from exercise_detection.classification import (
    SMOOTH_KEYS,
    _nanmean_pair,
    _prepare_series,
    _resolve_torso_scale,
    _select_visible_side,
)
from exercise_detection.constants import DEFAULT_SAMPLING_RATE
from exercise_detection.metrics import compute_metrics
from exercise_detection.segmentation import segment_reps
from exercise_detection.smoothing import smooth
from exercise_detection.types import FeatureSeries
from exercise_detection.view import classify_view


FRAMES = 90
SAMPLING_RATE = 30.0


def test_deadlift_clear_hinge():
    features = make_deadlift_features()

    label, view, confidence = classify_features(features)
    assert label == "deadlift"
    assert view == "side"
    assert confidence >= 0.45

    metrics, scores = _internal_metrics(features)
    assert scores.deadlift_veto is True
    assert scores.adjusted["deadlift"] > scores.adjusted["squat"]


def test_squat_lateral():
    features = make_squat_features()

    label, view, confidence = classify_features(features)
    assert label == "squat"
    assert view == "side"
    assert confidence >= 0.45

    metrics, scores = _internal_metrics(features)
    assert scores.deadlift_veto is False
    assert metrics.wrist_shoulder_diff_norm <= 0.18


def test_low_bar_squat_remains_squat():
    features = make_low_bar_squat_features()

    label, view, _ = classify_features(features)
    assert label == "squat"
    assert view == "side"


def test_borderline_hinge_prefers_deadlift():
    features = make_borderline_hinge_features()

    label, view, confidence = classify_features(features)
    assert label == "deadlift"
    assert view == "side"
    assert confidence >= 0.4


def test_bench_press_front_view():
    features = make_bench_features()

    label, view, confidence = classify_features(features)
    assert label == "bench_press"
    assert view == "front"
    assert confidence >= 0.45


def test_unknown_for_short_clip():
    frames = 10
    data = {key: np.zeros(frames) for key in ["knee_angle_left", "knee_angle_right"]}
    features = FeatureSeries(data=data, sampling_rate=SAMPLING_RATE, valid_frames=frames, total_frames=frames)
    label, view, confidence = classify_features(features)
    assert label == "unknown"
    assert view == "unknown"
    assert confidence == 0.0


def test_view_classifier_front_vs_side():
    squat = make_squat_features()
    bench = make_bench_features()

    squat_series = _prepare_series(squat)
    bench_series = _prepare_series(bench)

    squat_view = classify_view(
        shoulder_yaw=squat_series["shoulder_yaw_deg"],
        shoulder_z_delta=squat_series["shoulder_z_delta_abs"],
        shoulder_width=squat_series["shoulder_width_norm"],
        ankle_width=squat_series["ankle_width_norm"],
    )
    bench_view = classify_view(
        shoulder_yaw=bench_series["shoulder_yaw_deg"],
        shoulder_z_delta=bench_series["shoulder_z_delta_abs"],
        shoulder_width=bench_series["shoulder_width_norm"],
        ankle_width=bench_series["ankle_width_norm"],
    )

    assert squat_view.label == "side"
    assert bench_view.label == "front"


def _internal_metrics(features: FeatureSeries):
    series = _prepare_series(features)
    sr = float(features.sampling_rate or DEFAULT_SAMPLING_RATE)
    smoothed = {key: smooth(values, sr) for key, values in series.items() if key in SMOOTH_KEYS}

    def get(name: str) -> np.ndarray:
        return smoothed.get(name, series.get(name))

    torso_scale = _resolve_torso_scale(get("torso_length"), get("torso_length_world"))
    side = _select_visible_side(series)

    knee = get(f"knee_angle_{side}")
    hip = get(f"hip_angle_{side}")
    elbow = get(f"elbow_angle_{side}")
    torso = get("torso_tilt_deg")

    wrist_y_mean = _nanmean_pair(get("wrist_left_y"), get("wrist_right_y"))
    wrist_x_mean = _nanmean_pair(get("wrist_left_x"), get("wrist_right_x"))

    rep_slices = segment_reps(knee, wrist_y_mean, torso_scale, sr)

    series_bundle = {
        "knee_angle": knee,
        "hip_angle": hip,
        "elbow_angle": elbow,
        "torso_tilt": torso,
        "wrist_y": get(f"wrist_{side}_y"),
        "wrist_x": get(f"wrist_{side}_x"),
        "shoulder_y": get(f"shoulder_{side}_y"),
        "hip_y": get(f"hip_{side}_y"),
        "knee_x": get(f"knee_{side}_x"),
        "knee_y": get(f"knee_{side}_y"),
        "ankle_x": get(f"ankle_{side}_x"),
        "ankle_y": get(f"ankle_{side}_y"),
        "bar_y": wrist_y_mean,
        "bar_x": wrist_x_mean,
    }

    metrics = compute_metrics(rep_slices, series_bundle, torso_scale, sr)
    _, _, scores, _ = classify_exercise(metrics)
    return metrics, scores


def make_deadlift_features() -> FeatureSeries:
    t = np.linspace(0, 1, FRAMES)
    cycle = 0.5 - 0.5 * np.cos(2 * np.pi * t)

    knee = 155 - 25 * cycle
    hip = 170 - 65 * cycle
    elbow = 178 - 2 * cycle
    torso = 25 + 55 * cycle

    shoulder_y = np.full(FRAMES, 0.35)
    hip_y = np.full(FRAMES, 0.55)
    knee_y = 0.7 - 0.05 * cycle
    ankle_y = np.full(FRAMES, 0.95)

    ankle_x = np.full(FRAMES, 0.2)
    knee_x = 0.22 + 0.01 * cycle
    wrist_x = 0.21 + 0.005 * cycle

    wrist_y = 0.55 + 0.25 * cycle

    data = _assemble_clip(
        knee=knee,
        hip=hip,
        elbow=elbow,
        torso=torso,
        shoulder_y=shoulder_y,
        hip_y=hip_y,
        knee_x=knee_x,
        knee_y=knee_y,
        ankle_x=ankle_x,
        ankle_y=ankle_y,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
        shoulder_width=0.45,
        shoulder_yaw=35.0,
        shoulder_z=0.13,
        ankle_width=0.32,
    )
    return FeatureSeries(data=data, sampling_rate=SAMPLING_RATE, valid_frames=FRAMES, total_frames=FRAMES)


def make_squat_features() -> FeatureSeries:
    t = np.linspace(0, 1, FRAMES)
    cycle = 0.5 - 0.5 * np.cos(2 * np.pi * t)

    knee = 170 - 65 * cycle
    hip = 165 - 55 * cycle
    elbow = 110 - 5 * np.sin(2 * np.pi * t)
    torso = 18 + 18 * cycle

    shoulder_y = np.full(FRAMES, 0.35)
    hip_y = np.full(FRAMES, 0.55)
    knee_y = 0.7 - 0.05 * cycle
    ankle_y = np.full(FRAMES, 0.95)

    ankle_x = np.full(FRAMES, 0.2)
    knee_x = 0.2 + 0.12 * cycle
    wrist_x = 0.25 + 0.03 * cycle
    wrist_y = 0.38 + 0.03 * np.sin(2 * np.pi * t)

    data = _assemble_clip(
        knee=knee,
        hip=hip,
        elbow=elbow,
        torso=torso,
        shoulder_y=shoulder_y,
        hip_y=hip_y,
        knee_x=knee_x,
        knee_y=knee_y,
        ankle_x=ankle_x,
        ankle_y=ankle_y,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
        shoulder_width=0.47,
        shoulder_yaw=32.0,
        shoulder_z=0.12,
        ankle_width=0.34,
    )
    return FeatureSeries(data=data, sampling_rate=SAMPLING_RATE, valid_frames=FRAMES, total_frames=FRAMES)


def make_low_bar_squat_features() -> FeatureSeries:
    features = make_squat_features()
    data = {key: value.copy() for key, value in features.data.items()}
    data["torso_tilt_deg"] = 22 + 24 * (0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, FRAMES)))
    data["shoulder_yaw_deg"] = np.full(FRAMES, 28.0)
    return FeatureSeries(data=data, sampling_rate=SAMPLING_RATE, valid_frames=FRAMES, total_frames=FRAMES)


def make_borderline_hinge_features() -> FeatureSeries:
    features = make_deadlift_features()
    data = {key: value.copy() for key, value in features.data.items()}
    data["knee_angle_left"] = data["knee_angle_right"] = 160 - 40 * (0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, FRAMES)))
    data["wrist_left_y"] = data["wrist_right_y"] = 0.52 + 0.25 * (0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, FRAMES)))
    data["torso_tilt_deg"] = 25 + 50 * (0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, FRAMES)))
    return FeatureSeries(data=data, sampling_rate=SAMPLING_RATE, valid_frames=FRAMES, total_frames=FRAMES)


def make_bench_features() -> FeatureSeries:
    t = np.linspace(0, 1, FRAMES)
    cycle = 0.5 - 0.5 * np.cos(2 * np.pi * t)

    knee = np.full(FRAMES, 175.0)
    hip = np.full(FRAMES, 165.0)
    elbow = 150 - 90 * cycle
    torso = np.full(FRAMES, 78.0)

    shoulder_y = np.full(FRAMES, 0.2)
    hip_y = np.full(FRAMES, 0.25)
    knee_y = np.full(FRAMES, 0.3)
    ankle_y = np.full(FRAMES, 0.35)

    ankle_x = np.full(FRAMES, 0.15)
    knee_x = np.full(FRAMES, 0.18)
    wrist_x = 0.2 + 0.01 * np.sin(2 * np.pi * t)
    wrist_y = 0.3 + 0.28 * cycle

    data = _assemble_clip(
        knee=knee,
        hip=hip,
        elbow=elbow,
        torso=torso,
        shoulder_y=shoulder_y,
        hip_y=hip_y,
        knee_x=knee_x,
        knee_y=knee_y,
        ankle_x=ankle_x,
        ankle_y=ankle_y,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
        shoulder_width=0.62,
        shoulder_yaw=6.0,
        shoulder_z=0.03,
        ankle_width=0.58,
    )
    return FeatureSeries(data=data, sampling_rate=SAMPLING_RATE, valid_frames=FRAMES, total_frames=FRAMES)


def _assemble_clip(
    *,
    knee: np.ndarray,
    hip: np.ndarray,
    elbow: np.ndarray,
    torso: np.ndarray,
    shoulder_y: np.ndarray,
    hip_y: np.ndarray,
    knee_x: np.ndarray,
    knee_y: np.ndarray,
    ankle_x: np.ndarray,
    ankle_y: np.ndarray,
    wrist_x: np.ndarray,
    wrist_y: np.ndarray,
    shoulder_width: float,
    shoulder_yaw: float,
    shoulder_z: float,
    ankle_width: float,
) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {
        "knee_angle_left": knee,
        "knee_angle_right": knee.copy(),
        "hip_angle_left": hip,
        "hip_angle_right": hip.copy(),
        "elbow_angle_left": elbow,
        "elbow_angle_right": elbow.copy(),
        "torso_tilt_deg": torso,
        "torso_length": np.full(FRAMES, 1.0),
        "torso_length_world": np.full(FRAMES, 1.05),
        "shoulder_left_y": shoulder_y,
        "shoulder_right_y": shoulder_y.copy(),
        "hip_left_y": hip_y,
        "hip_right_y": hip_y.copy(),
        "knee_left_x": knee_x,
        "knee_left_y": knee_y,
        "knee_right_x": knee_x.copy(),
        "knee_right_y": knee_y.copy(),
        "ankle_left_x": ankle_x,
        "ankle_left_y": ankle_y,
        "ankle_right_x": ankle_x.copy(),
        "ankle_right_y": ankle_y.copy(),
        "wrist_left_x": wrist_x,
        "wrist_right_x": wrist_x.copy(),
        "wrist_left_y": wrist_y,
        "wrist_right_y": wrist_y.copy(),
        "shoulder_width_norm": np.full(FRAMES, shoulder_width),
        "shoulder_yaw_deg": np.full(FRAMES, shoulder_yaw),
        "shoulder_z_delta_abs": np.full(FRAMES, shoulder_z),
        "ankle_width_norm": np.full(FRAMES, ankle_width),
    }
    return data

