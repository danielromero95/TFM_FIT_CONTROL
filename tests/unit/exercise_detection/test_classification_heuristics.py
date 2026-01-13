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
        bar_above_hip_norm=0.0,
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
        arms_high_fraction=0.3,
        bar_near_shoulders_fraction=0.2,
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


def test_bar_above_hip_clamps_deadlift_score():
    agg = _make_agg(
        knee_min=105.0,
        hip_min=120.0,
        elbow_bottom=170.0,
        torso_tilt_bottom=40.0,
        wrist_shoulder_diff_norm=0.08,
        wrist_hip_diff_norm=0.25,
        bar_above_hip_norm=0.18,
        knee_forward_norm=0.1,
        knee_rom=55.0,
        hip_rom=55.0,
        bar_range_norm=0.18,
        bar_ankle_diff_norm=0.03,
    )
    label, _confidence, scores, _probabilities, diagnostics = classify_exercise(agg)
    assert diagnostics["deadlift_arm_gate"]["active"] is True
    assert scores.adjusted["deadlift"] < scores.adjusted["squat"]
    assert label == "squat"


def test_bar_below_hip_penalizes_squat():
    agg = _make_agg(
        knee_min=95.0,
        hip_min=115.0,
        elbow_bottom=175.0,
        torso_tilt_bottom=42.0,
        wrist_shoulder_diff_norm=0.08,
        wrist_hip_diff_norm=0.3,
        bar_above_hip_norm=-0.2,
        knee_forward_norm=0.02,
        knee_rom=45.0,
        hip_rom=65.0,
        bar_range_norm=0.2,
        bar_ankle_diff_norm=0.02,
    )
    label, _confidence, scores, _probabilities, diagnostics = classify_exercise(agg)
    assert diagnostics["bar_above_hip_norm"] < 0.0
    assert scores.adjusted["deadlift"] > scores.adjusted["squat"]
    assert label == "deadlift"


def test_deadlift_veto_does_not_trigger_when_bar_is_high():
    agg = _make_agg(
        torso_tilt_bottom=55.0,
        elbow_bottom=175.0,
        knee_min=140.0,
        wrist_hip_diff_norm=0.35,
        hip_rom=0.2,
        bar_range_norm=0.2,
        bar_above_hip_norm=0.20,
    )
    _label, _confidence, _scores, _probabilities, diagnostics = classify_exercise(agg)
    assert diagnostics["deadlift_veto"]["active"] is False
    assert diagnostics["deadlift_veto"]["cues"].get("bar_above_hip_block") is True
