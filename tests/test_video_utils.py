"""Tests for video validation helpers."""

import cv2
import numpy as np
import pytest

from src.A_preprocessing.video_utils import validate_video


def create_dummy_video(path, width=64, height=48, num_frames=30, fps=15):
    """Create a small black-frame video for validation tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not open VideoWriter to create test video")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


def test_validate_video_ok(tmp_path):
    """validate_video should succeed on a freshly created dummy video."""
    dummy_video = tmp_path / "test_ok.mp4"
    create_dummy_video(str(dummy_video), width=64, height=48, num_frames=30, fps=15)

    info = validate_video(str(dummy_video))
    assert isinstance(info, dict)
    assert abs(info["fps"] - 15) < 1e-3
    assert info["frame_count"] == 30
    assert abs(info["duration"] - 2.0) < 1e-3


def test_validate_video_no_exist():
    """validate_video should raise when the path does not exist."""
    with pytest.raises(IOError):
        validate_video("path/that/does/not/exist/video.avi")


def test_validate_video_corrupt(tmp_path, monkeypatch):
    """Simulate a corrupt video by forcing FPS=0."""
    dummy_video = tmp_path / "corrupt.mp4"
    create_dummy_video(str(dummy_video), width=64, height=48, num_frames=10, fps=10)

    class FakeCap:
        def __init__(self, path):
            self.opened = True

        def isOpened(self):  # pragma: no cover - test double
            return True

        def get(self, prop):  # pragma: no cover - test double
            if prop == cv2.CAP_PROP_FPS:
                return 0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 10
            return 0

        def release(self):  # pragma: no cover - test double
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: FakeCap(_))

    with pytest.raises(ValueError):
        validate_video(str(dummy_video))
