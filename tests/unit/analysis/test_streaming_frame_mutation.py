from __future__ import annotations

import numpy as np

from src import config
from src.B_pose_estimation.types import Landmark, PoseResult
from src.C_analysis.streaming import stream_pose_and_detection


def test_stream_pose_does_not_mutate_frames(monkeypatch, tmp_path) -> None:
    cfg = config.load_default()
    cfg.pose.use_crop = False
    cfg.pose.use_roi_tracking = False

    frames = [np.full((4, 4, 3), fill_value=idx, dtype=np.uint8) for idx in range(3)]
    original_frames = [frame.copy() for frame in frames]

    class _FakePoseEstimator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def estimate(self, _image_bgr: np.ndarray) -> PoseResult:
            landmarks = [Landmark(x=0.1, y=0.2, z=0.0, visibility=1.0) for _ in range(33)]
            return PoseResult(landmarks=landmarks, annotated_image=None, crop_box=None, pose_ok=True)

    monkeypatch.setattr("src.C_analysis.streaming.PoseEstimator", _FakePoseEstimator)

    class _DummyWriter:
        def write(self, _frame: np.ndarray) -> None:
            return None

        def release(self) -> None:
            return None

    monkeypatch.setattr(
        "src.C_analysis.streaming._open_debug_writer",
        lambda *_args, **_kwargs: _DummyWriter(),
    )

    def _preview_callback(_frame: np.ndarray, _idx: int, _ts_ms: float) -> None:
        return None

    stream_pose_and_detection(
        frames,
        cfg,
        detection_enabled=False,
        detection_source_fps=30.0,
        debug_video_path=tmp_path / "debug.mp4",
        debug_video_fps=30.0,
        preview_callback=_preview_callback,
        preview_fps=10.0,
    )

    for frame, original in zip(frames, original_frames):
        assert np.array_equal(frame, original)
