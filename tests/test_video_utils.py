"""Tests for video validation helpers."""

from __future__ import annotations

import cv2
import pytest

from src.A_preprocessing.video_utils import validate_video


class _FakeCapture:
    """Simple stub implementing the VideoCapture interface used in the tests."""

    def __init__(self, *, opened: bool, fps: float, frame_count: int) -> None:
        self._opened = opened
        self._fps = float(fps)
        self._frame_count = int(frame_count)

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._frame_count
        return 0.0

    def release(self) -> None:  # pragma: no cover - no state to clean up
        self._opened = False


def _touch(path: str) -> None:
    with open(path, "wb") as handle:
        handle.write(b"")


def test_validate_video_ok(tmp_path, monkeypatch):
    """validate_video should succeed when metadata is well-formed."""

    dummy_video = tmp_path / "test_ok.mp4"
    _touch(dummy_video)

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: _FakeCapture(opened=True, fps=15.0, frame_count=30),
    )

    info = validate_video(str(dummy_video))
    assert isinstance(info, dict)
    assert abs(info["fps"] - 15.0) < 1e-6
    assert info["frame_count"] == 30
    assert abs(info["duration"] - 2.0) < 1e-6


def test_validate_video_no_exist():
    """validate_video should raise when the path does not exist."""

    with pytest.raises(IOError):
        validate_video("path/that/does/not/exist/video.avi")


def test_validate_video_corrupt(tmp_path, monkeypatch):
    """Simulate a corrupt video by forcing FPS=0."""

    dummy_video = tmp_path / "corrupt.mp4"
    _touch(dummy_video)

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: _FakeCapture(opened=True, fps=0.0, frame_count=10),
    )

    with pytest.raises(ValueError):
        validate_video(str(dummy_video))
