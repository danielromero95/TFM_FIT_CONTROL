import pytest

np = pytest.importorskip("numpy")

from exercise_detection.classification import classify_features
from exercise_detection.types import FeatureSeries


def _make_series(length: int, top: float, bottom: float) -> np.ndarray:
    segment = length // 3
    down = np.linspace(top, bottom, segment, endpoint=False)
    bottom_seg = np.full(segment, bottom)
    up = np.linspace(bottom, top, length - 2 * segment, endpoint=True)
    return np.concatenate([down, bottom_seg, up])


def _build_features(
    *,
    knee_bottom: float,
    hip_bottom: float,
    elbow_bottom: float,
    torso_bottom: float,
    wrist_shoulder_diff: float,
    wrist_hip_diff: float,
    knee_forward: float,
    length: int = 60,
) -> FeatureSeries:
    torso_length = 1.0
    hip_height = 0.5
    shoulder_height = hip_height + wrist_shoulder_diff + wrist_hip_diff
    wrist_bottom = shoulder_height - wrist_shoulder_diff
    wrist_top = wrist_bottom + 0.18

    knee_series = _make_series(length, 170.0, knee_bottom)
    hip_series = _make_series(length, 165.0, hip_bottom)
    elbow_top = max(elbow_bottom + 5.0, 160.0)
    elbow_series = _make_series(length, elbow_top, elbow_bottom)
    torso_top = max(torso_bottom - 15.0, 5.0)
    torso_series = _make_series(length, torso_top, torso_bottom)
    wrist_series = _make_series(length, wrist_top, wrist_bottom)

    knee_x = np.full(length, 0.12 + knee_forward)
    ankle_x = np.full(length, 0.12)
    knee_y = np.full(length, 0.35)
    ankle_y = np.full(length, 0.0)

    data = {
        "knee_angle_left": knee_series.copy(),
        "knee_angle_right": knee_series.copy(),
        "hip_angle_left": hip_series.copy(),
        "hip_angle_right": hip_series.copy(),
        "elbow_angle_left": elbow_series.copy(),
        "elbow_angle_right": elbow_series.copy(),
        "torso_tilt_deg": torso_series.copy(),
        "torso_length": np.full(length, torso_length),
        "torso_length_world": np.full(length, torso_length),
        "wrist_left_y": wrist_series.copy(),
        "wrist_right_y": wrist_series.copy(),
        "wrist_left_x": np.full(length, 0.08),
        "wrist_right_x": np.full(length, 0.16),
        "shoulder_left_y": np.full(length, shoulder_height),
        "shoulder_right_y": np.full(length, shoulder_height),
        "hip_left_y": np.full(length, hip_height),
        "hip_right_y": np.full(length, hip_height),
        "knee_left_x": knee_x.copy(),
        "knee_right_x": knee_x.copy(),
        "knee_left_y": knee_y.copy(),
        "knee_right_y": knee_y.copy(),
        "ankle_left_x": ankle_x.copy(),
        "ankle_right_x": ankle_x.copy(),
        "ankle_left_y": ankle_y.copy(),
        "ankle_right_y": ankle_y.copy(),
        "shoulder_width_norm": np.full(length, 0.38),
        "shoulder_yaw_deg": np.full(length, 32.0),
        "shoulder_z_delta_abs": np.full(length, 0.18),
        "ankle_width_norm": np.full(length, 0.26),
        "hip_left_x": np.full(length, 0.06),
        "hip_right_x": np.full(length, 0.18),
        "shoulder_left_x": np.full(length, 0.02),
        "shoulder_right_x": np.full(length, 0.22),
        "pelvis_y": np.full(length, hip_height),
    }

    length_values = next(iter(data.values())).size
    return FeatureSeries(
        data=data,
        sampling_rate=30.0,
        valid_frames=length_values,
        total_frames=length_values,
    )


def test_deadlift_veto_prefers_deadlift_when_hinge_cues_present() -> None:
    features = _build_features(
        knee_bottom=130.0,
        hip_bottom=110.0,
        elbow_bottom=175.0,
        torso_bottom=55.0,
        wrist_shoulder_diff=0.45,
        wrist_hip_diff=0.45,
        knee_forward=0.02,
    )

    label, _, confidence = classify_features(features)

    assert label == "deadlift"
    assert confidence >= 0.45


def test_squat_requires_arms_up_and_is_selected() -> None:
    features = _build_features(
        knee_bottom=95.0,
        hip_bottom=120.0,
        elbow_bottom=90.0,
        torso_bottom=30.0,
        wrist_shoulder_diff=0.05,
        wrist_hip_diff=0.10,
        knee_forward=0.12,
    )

    label, _, confidence = classify_features(features)

    assert label == "squat"
    assert confidence >= 0.45


def test_deadlift_veto_blocks_squat_on_borderline_case() -> None:
    features = _build_features(
        knee_bottom=118.0,
        hip_bottom=130.0,
        elbow_bottom=168.0,
        torso_bottom=40.0,
        wrist_shoulder_diff=0.35,
        wrist_hip_diff=0.25,
        knee_forward=0.03,
    )

    label, _, confidence = classify_features(features)

    assert label == "deadlift"
    assert confidence >= 0.45
