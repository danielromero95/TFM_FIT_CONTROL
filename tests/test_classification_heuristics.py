from __future__ import annotations

from exercise_detection.classifier import classify_exercise
from exercise_detection.types import AggregateMetrics


def _make_agg(**overrides: float) -> AggregateMetrics:
    defaults = dict(
        knee_min=150.0,
        hip_min=160.0,
        elbow_bottom=150.0,
        torso_tilt_bottom=30.0,
        wrist_shoulder_diff_norm=0.25,
        wrist_hip_diff_norm=0.1,
        knee_forward_norm=0.02,
        tibia_angle_deg=20.0,
        bar_ankle_diff_norm=0.1,
        knee_rom=20.0,
        hip_rom=20.0,
        elbow_rom=20.0,
        bar_range_norm=0.05,
        hip_range_norm=0.05,
        bar_horizontal_std_norm=0.02,
        duration_s=3.0,
        rep_count=3,
    )
    defaults.update(overrides)
    return AggregateMetrics(**defaults)


def test_classify_squat_clear_metrics() -> None:
    metrics = _make_agg(
        knee_min=80.0,
        hip_min=105.0,
        elbow_bottom=85.0,
        torso_tilt_bottom=20.0,
        wrist_shoulder_diff_norm=0.05,
        wrist_hip_diff_norm=0.05,
        knee_forward_norm=0.12,
        tibia_angle_deg=22.0,
        knee_rom=60.0,
        hip_rom=45.0,
    )
    label, confidence, *_ = classify_exercise(metrics)
    assert label == "squat"
    assert confidence >= 0.75


def test_classify_deadlift_clear_metrics() -> None:
    metrics = _make_agg(
        knee_min=140.0,
        hip_min=160.0,
        elbow_bottom=175.0,
        torso_tilt_bottom=55.0,
        wrist_shoulder_diff_norm=0.35,
        wrist_hip_diff_norm=0.32,
        knee_forward_norm=0.0,
        bar_ankle_diff_norm=0.02,
        hip_rom=60.0,
        bar_range_norm=0.16,
        bar_horizontal_std_norm=0.02,
    )
    label, confidence, *_ = classify_exercise(metrics)
    assert label == "deadlift"
    assert confidence >= 0.75


def test_classify_bench_clear_metrics() -> None:
    metrics = _make_agg(
        torso_tilt_bottom=70.0,
        elbow_rom=50.0,
        bar_range_norm=0.2,
        knee_rom=10.0,
        hip_rom=10.0,
        hip_range_norm=0.04,
        bar_horizontal_std_norm=0.02,
    )
    label, confidence, *_ = classify_exercise(metrics)
    assert label == "bench_press"
    assert confidence >= 0.75


def test_ambiguous_metrics_yield_unknown() -> None:
    metrics = _make_agg(
        knee_min=130.0,
        hip_min=150.0,
        elbow_bottom=150.0,
        torso_tilt_bottom=38.0,
        wrist_shoulder_diff_norm=0.2,
        wrist_hip_diff_norm=0.15,
        knee_forward_norm=0.04,
        knee_rom=25.0,
        hip_rom=20.0,
        bar_range_norm=0.08,
    )
    label, confidence, *_ = classify_exercise(metrics)
    assert label == "unknown"
    assert confidence <= 0.55
