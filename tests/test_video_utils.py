"""Pruebas para las utilidades de validación de vídeos."""

from __future__ import annotations

import cv2
import pytest

from src.A_preprocessing.video_utils import validate_video


class _FakeCapture:
    """Stub sencillo que implementa la interfaz de VideoCapture usada en las pruebas."""

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

    def release(self) -> None:  # pragma: no cover - sin estado que limpiar
        self._opened = False


def _touch(path: str) -> None:
    with open(path, "wb") as handle:
        handle.write(b"")


def test_validate_video_ok(tmp_path, monkeypatch):
    """`validate_video` debería funcionar cuando los metadatos son correctos."""

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
    """`validate_video` debe lanzar una excepción cuando la ruta no existe."""

    with pytest.raises(IOError):
        validate_video("path/that/does/not/exist/video.avi")


def test_validate_video_corrupt(tmp_path, monkeypatch):
    """Simula un vídeo corrupto forzando FPS=0."""

    dummy_video = tmp_path / "corrupt.mp4"
    _touch(dummy_video)

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: _FakeCapture(opened=True, fps=0.0, frame_count=10),
    )

    with pytest.raises(ValueError):
        validate_video(str(dummy_video))
