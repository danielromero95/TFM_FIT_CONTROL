"""Unit tests for domain-specific pipeline exceptions."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Provide lightweight stubs for heavy optional dependencies so analysis_service imports cleanly.
if "cv2" not in sys.modules:
    class _FakeVideoCapture:
        _next_open: bool = False

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._opened = self.__class__._next_open

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:  # pragma: no cover - not exercised
            self._opened = False

        def set(self, *_args: Any, **_kwargs: Any) -> bool:  # pragma: no cover - stubbed
            return False

        def grab(self) -> bool:  # pragma: no cover - stubbed
            return False

        def read(self) -> tuple[bool, Any]:  # pragma: no cover - stubbed
            return False, None

        def get(self, *_args: Any, **_kwargs: Any) -> float:  # pragma: no cover - stubbed
            return 0.0

    class _FakeCv2(ModuleType):
        ROTATE_90_CLOCKWISE = 0
        ROTATE_180 = 1
        ROTATE_90_COUNTERCLOCKWISE = 2
        CAP_PROP_POS_MSEC = 0
        CAP_PROP_POS_FRAMES = 1
        CAP_PROP_FRAME_WIDTH = 2
        CAP_PROP_FRAME_HEIGHT = 3
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_COUNT = 7
        CAP_PROP_FOURCC = 10
        INTER_AREA = 0
        INTER_LINEAR = 1

        def __init__(self) -> None:
            super().__init__("cv2")
            self.VideoCapture = _FakeVideoCapture

        @staticmethod
        def rotate(frame: Any, _rot: Any) -> Any:  # pragma: no cover - stubbed
            return frame

        @staticmethod
        def resize(frame: Any, _size: Any, *, interpolation: Any = None) -> Any:  # pragma: no cover - stubbed
            return frame

        @staticmethod
        def cvtColor(frame: Any, _code: Any) -> Any:  # pragma: no cover - stubbed
            return frame

    sys.modules["cv2"] = _FakeCv2()

if "numpy" not in sys.modules:
    fake_numpy = ModuleType("numpy")

    def _stub_np(*args: Any, **kwargs: Any):  # pragma: no cover - should not be called in these tests
        raise NotImplementedError("numpy stub")

    fake_numpy.array = _stub_np
    fake_numpy.arange = _stub_np
    fake_numpy.full = _stub_np
    fake_numpy.where = _stub_np
    fake_numpy.isnan = _stub_np
    fake_numpy.interp = _stub_np
    sys.modules["numpy"] = fake_numpy

if "pandas" not in sys.modules:
    fake_pandas = ModuleType("pandas")

    class _FakeDataFrame:  # pragma: no cover - only to satisfy type hints
        pass

    def _stub_pd(*args: Any, **kwargs: Any):  # pragma: no cover - should not be called in these tests
        raise NotImplementedError("pandas stub")

    fake_pandas.DataFrame = _FakeDataFrame
    fake_pandas.Series = _FakeDataFrame
    fake_pandas.notna = _stub_pd
    fake_pandas.isna = _stub_pd
    sys.modules["pandas"] = fake_pandas

if "scipy" not in sys.modules:
    fake_scipy = ModuleType("scipy")
    fake_signal = ModuleType("scipy.signal")

    def _stub_scipy(*args: Any, **kwargs: Any):  # pragma: no cover - should not be called in these tests
        raise NotImplementedError("scipy stub")

    fake_signal.find_peaks = _stub_scipy
    fake_signal.savgol_filter = _stub_scipy
    fake_scipy.signal = fake_signal
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.signal"] = fake_signal

from src import config
from src.A_preprocessing.video_metadata import VideoInfo
from src.services import analysis_service
from src.services.errors import NoFramesExtracted, VideoOpenError


def _make_cfg(tmp_path: Path) -> config.Config:
    cfg = config.Config()
    base_dir = tmp_path / "outputs"
    cfg.output.base_dir = base_dir
    cfg.output.counts_dir = base_dir / "counts"
    cfg.output.poses_dir = base_dir / "poses"
    return cfg


def test_run_pipeline_raises_video_open_error(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    video_path = tmp_path / "missing_video.mp4"

    # Ensure the next capture reports failure to open.
    analysis_service.cv2.VideoCapture._next_open = False  # type: ignore[attr-defined]

    with pytest.raises(VideoOpenError):
        analysis_service.run_pipeline(str(video_path), cfg)
    analysis_service.cv2.VideoCapture._next_open = False  # type: ignore[attr-defined]


def test_run_pipeline_raises_no_frames_extracted(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    video_path = tmp_path / "video.mp4"
    video_path.touch()

    fake_info = VideoInfo(
        path=video_path,
        width=640,
        height=480,
        fps=30.0,
        frame_count=30,
        duration_sec=1.0,
        rotation=0,
        codec="",
        fps_source="metadata",
    )

    monkeypatch.setattr(analysis_service, "read_video_file_info", lambda *_, **__: fake_info)

    class _DummyCap:
        def release(self) -> None:
            pass

    monkeypatch.setattr(analysis_service, "_open_video_cap", lambda _path: _DummyCap())
    monkeypatch.setattr(
        analysis_service,
        "extract_and_preprocess_frames",
        lambda **_: ([], 0.0),
    )

    with pytest.raises(NoFramesExtracted):
        analysis_service.run_pipeline(str(video_path), cfg)
