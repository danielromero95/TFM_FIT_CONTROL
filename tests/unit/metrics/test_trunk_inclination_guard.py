import numpy as np

from src.B_pose_estimation.pipeline.compute import calculate_metrics_from_sequence
from src.B_pose_estimation.types import Landmark


def _make_frame(torso_height: float) -> list[Landmark]:
    points = [Landmark(0.0, 0.0, 0.0, 1.0) for _ in range(33)]
    # Hips
    points[23] = Landmark(0.1, 0.0, 0.0, 1.0)
    points[24] = Landmark(0.3, 0.0, 0.0, 1.0)
    # Shoulders aligned above hips
    points[11] = Landmark(0.1, torso_height, 0.0, 1.0)
    points[12] = Landmark(0.3, torso_height, 0.0, 1.0)
    # Knees/ankles to keep angles finite
    points[25] = Landmark(0.1, -0.4, 0.0, 1.0)
    points[26] = Landmark(0.3, -0.4, 0.0, 1.0)
    points[27] = Landmark(0.1, -0.8, 0.0, 1.0)
    points[28] = Landmark(0.3, -0.8, 0.0, 1.0)
    return points


def test_trunk_collapse_masks_pose_ok():
    normal = [_make_frame(1.0) for _ in range(3)]
    collapsed = [_make_frame(0.02) for _ in range(2)]
    sequence = normal + collapsed + [_make_frame(0.9)]

    df = calculate_metrics_from_sequence(sequence, fps=30.0)

    assert {"trunk_len", "trunk_dx", "trunk_dy", "trunk_ok"}.issubset(df.columns)
    assert "trunk_quality" in df.attrs

    # Valid frames should keep stable inclination near zero
    valid_angles = df.loc[df["trunk_ok"] > 0.5, "trunk_inclination_deg"].dropna()
    assert np.all(valid_angles < 5.0)

    # Collapsed torso frames must be masked out
    invalid_mask = df["trunk_ok"] < 0.5
    assert invalid_mask.sum() >= 2
    assert df.loc[invalid_mask, "trunk_inclination_deg"].isna().all()

    # Pose quality should stay true when only the trunk collapses
    assert np.isclose(df["pose_ok"].sum(), len(df))


def test_flat_trunk_does_not_mask_joint_angles():
    sequence = []
    for _ in range(5):
        frame = _make_frame(0.01)
        frame[11] = Landmark(0.5, 0.01, 0.0, 1.0)
        frame[12] = Landmark(0.7, 0.01, 0.0, 1.0)
        sequence.append(frame)

    df = calculate_metrics_from_sequence(sequence, fps=30.0)

    # Even with a flat trunk, pose_ok should reflect the quality mask only
    assert np.isclose(df["pose_ok"].sum(), len(df))

    # Joint angles must remain available
    assert df["left_knee"].notna().sum() == len(df)
    assert df["right_knee"].notna().sum() == len(df)

    # Trunk data can be invalidated independently
    assert (df["trunk_ok"] == 0).all()
    assert df["trunk_inclination_deg"].isna().all()
