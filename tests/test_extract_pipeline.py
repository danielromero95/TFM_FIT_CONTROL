import numpy as np
import pytest

from src.B_pose_estimation.pipeline.extract import extract_landmarks_from_frames
from src.B_pose_estimation.types import PoseResult


class _DummyEstimator:
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def estimate(self, _image):
        self.calls += 1
        return PoseResult(
            landmarks=[{"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 1.0}],
            annotated_image=None,
            crop_box=None,
            pose_ok=True,
        )

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return None


def test_extract_landmarks_includes_pose_ok_only_when_requested(monkeypatch):
    monkeypatch.setattr(
        "src.B_pose_estimation.pipeline.extract.PoseEstimator", _DummyEstimator
    )
    monkeypatch.setattr(
        "src.B_pose_estimation.pipeline.extract.CroppedPoseEstimator", _DummyEstimator
    )
    monkeypatch.setattr(
        "src.B_pose_estimation.pipeline.extract.RoiPoseEstimator", _DummyEstimator
    )

    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]

    df_without = extract_landmarks_from_frames(frames, include_pose_ok=False)
    assert "pose_ok" not in df_without.columns
    assert np.issubdtype(df_without["frame_idx"].dtype, np.integer)

    df_with = extract_landmarks_from_frames(frames, include_pose_ok=True)
    assert "pose_ok" in df_with.columns
    assert np.issubdtype(df_with["pose_ok"].dtype, np.floating)
    assert df_with.loc[0, "pose_ok"] == pytest.approx(1.0)
