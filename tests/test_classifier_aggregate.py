from __future__ import annotations

from exercise_detection.classifier import classify_exercise
from exercise_detection.types import AggregateMetrics


def _make_metrics(**overrides: float) -> AggregateMetrics:
    base = {
        "knee_min": 150.0,
        "hip_min": 160.0,
        "elbow_bottom": 150.0,
        "torso_tilt_bottom": 40.0,
        "wrist_shoulder_diff_norm": 0.2,
        "wrist_hip_diff_norm": 0.1,
        "knee_forward_norm": 0.04,
        "tibia_angle_deg": 28.0,
        "bar_ankle_diff_norm": 0.1,
        "knee_rom": 25.0,
        "hip_rom": 25.0,
        "elbow_rom": 20.0,
        "bar_range_norm": 0.1,
        "hip_range_norm": 0.05,
        "bar_horizontal_std_norm": 0.03,
        "rep_count": 1,
    }
    base.update(overrides)
    return AggregateMetrics(**base)


def test_aggregate_metrics_clear_squat():
    metrics = _make_metrics(
        knee_min=85.0,
        hip_min=110.0,
        elbow_bottom=90.0,
        torso_tilt_bottom=25.0,
        wrist_shoulder_diff_norm=0.08,
        wrist_hip_diff_norm=0.05,
        knee_forward_norm=0.1,
        tibia_angle_deg=25.0,
        bar_ankle_diff_norm=0.15,
        knee_rom=60.0,
        hip_rom=40.0,
        elbow_rom=15.0,
        bar_range_norm=0.08,
    )
    label, confidence, _, _, _ = classify_exercise(metrics)
    assert label == "squat"
    assert confidence >= 0.75


def test_aggregate_metrics_clear_deadlift():
    metrics = _make_metrics(
        knee_min=130.0,
        hip_min=160.0,
        elbow_bottom=175.0,
        torso_tilt_bottom=65.0,
        wrist_shoulder_diff_norm=0.32,
        wrist_hip_diff_norm=0.3,
        knee_forward_norm=0.01,
        tibia_angle_deg=20.0,
        bar_ankle_diff_norm=0.04,
        knee_rom=20.0,
        hip_rom=60.0,
        elbow_rom=10.0,
        bar_range_norm=0.2,
    )
    label, confidence, _, _, _ = classify_exercise(metrics)
    assert label == "deadlift"
    assert confidence >= 0.75


def test_aggregate_metrics_clear_bench():
    metrics = _make_metrics(
        knee_min=175.0,
        hip_min=175.0,
        elbow_bottom=100.0,
        torso_tilt_bottom=75.0,
        wrist_shoulder_diff_norm=0.1,
        wrist_hip_diff_norm=0.05,
        knee_forward_norm=0.02,
        tibia_angle_deg=20.0,
        bar_ankle_diff_norm=0.2,
        knee_rom=10.0,
        hip_rom=10.0,
        elbow_rom=70.0,
        bar_range_norm=0.25,
        hip_range_norm=0.02,
        bar_horizontal_std_norm=0.02,
    )
    label, confidence, _, _, _ = classify_exercise(metrics)
    assert label == "bench_press"
    assert confidence >= 0.75


def test_aggregate_metrics_ambiguous():
    metrics = _make_metrics(
        knee_min=118.0,
        hip_min=150.0,
        elbow_bottom=150.0,
        torso_tilt_bottom=40.0,
        wrist_shoulder_diff_norm=0.2,
        wrist_hip_diff_norm=0.15,
        knee_forward_norm=0.04,
        tibia_angle_deg=30.0,
        bar_ankle_diff_norm=0.1,
        knee_rom=25.0,
        hip_rom=25.0,
        elbow_rom=15.0,
        bar_range_norm=0.08,
    )
    label, confidence, _, _, _ = classify_exercise(metrics)
    assert label == "unknown"
    assert confidence == 0.0
