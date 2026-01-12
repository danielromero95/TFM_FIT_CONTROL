import pytest

from src.exercise_detection.classifier import classify_exercise
from src.exercise_detection.types import AggregateMetrics


def _make_agg(**overrides):
    base = dict(
        knee_min=150.0,
        hip_min=150.0,
        elbow_bottom=150.0,
        torso_tilt_bottom=45.0,
        wrist_shoulder_diff_norm=0.25,
        wrist_hip_diff_norm=0.1,
        knee_forward_norm=0.0,
        tibia_angle_deg=20.0,
        bar_ankle_diff_norm=0.15,
        knee_rom=20.0,
        hip_rom=20.0,
        elbow_rom=20.0,
        bar_range_norm=0.05,
        hip_range_norm=0.05,
        bar_vertical_range_norm=0.05,
        bar_horizontal_std_norm=0.02,
        duration_s=2.0,
        rep_count=1,
    )
    base.update(overrides)
    return AggregateMetrics(**base)


@pytest.mark.parametrize(
    "agg, expected_label",
    [
        (
            _make_agg(
                knee_min=80.0,
                hip_min=100.0,
                elbow_bottom=95.0,
                torso_tilt_bottom=30.0,
                wrist_shoulder_diff_norm=0.05,
                wrist_hip_diff_norm=0.05,
                knee_forward_norm=0.12,
                knee_rom=60.0,
                hip_rom=45.0,
                bar_range_norm=0.08,
                bar_ankle_diff_norm=0.12,
            ),
            "squat",
        ),
        (
            _make_agg(
                knee_min=135.0,
                hip_min=150.0,
                elbow_bottom=180.0,
                torso_tilt_bottom=50.0,
                wrist_shoulder_diff_norm=0.35,
                wrist_hip_diff_norm=0.35,
                knee_forward_norm=0.0,
                knee_rom=30.0,
                hip_rom=70.0,
                bar_range_norm=0.2,
                bar_ankle_diff_norm=0.02,
            ),
            "deadlift",
        ),
        (
            _make_agg(
                knee_min=160.0,
                hip_min=170.0,
                elbow_bottom=150.0,
                torso_tilt_bottom=70.0,
                wrist_shoulder_diff_norm=0.05,
                wrist_hip_diff_norm=0.0,
                knee_forward_norm=0.0,
                knee_rom=10.0,
                hip_rom=10.0,
                elbow_rom=45.0,
                bar_range_norm=0.18,
                hip_range_norm=0.05,
                bar_horizontal_std_norm=0.02,
            ),
            "bench_press",
        ),
    ],
)
def test_clear_classifications_have_high_confidence(agg, expected_label):
    label, confidence, _scores, _probabilities, _diagnostics = classify_exercise(agg)
    assert label == expected_label
    assert confidence > 0.75


def test_ambiguous_metrics_return_unknown():
    agg = _make_agg(
        knee_min=115.0,
        hip_min=145.0,
        elbow_bottom=150.0,
        torso_tilt_bottom=40.0,
        wrist_shoulder_diff_norm=0.2,
        wrist_hip_diff_norm=0.18,
        knee_forward_norm=0.04,
        knee_rom=30.0,
        hip_rom=25.0,
        elbow_rom=15.0,
        bar_range_norm=0.09,
    )
    label, confidence, _scores, _probabilities, _diagnostics = classify_exercise(agg)
    assert label == "unknown"
    assert confidence == 0.0
