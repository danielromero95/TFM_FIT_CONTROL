from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# Proporciona stubs ligeros para dependencias opcionales pesadas, de modo que
# el paquete de análisis pueda importarse aunque falten esas ruedas en el entorno.
if "cv2" not in sys.modules:
    class _FakeVideoCapture:
        _next_open: bool = False

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._opened = self.__class__._next_open

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:  # pragma: no cover - no se ejecuta en estas pruebas
            self._opened = False

        def set(self, *_args: Any, **_kwargs: Any) -> bool:  # pragma: no cover - stub simulado
            return False

        def grab(self) -> bool:  # pragma: no cover - stub simulado
            return False

        def read(self) -> tuple[bool, Any]:  # pragma: no cover - stub simulado
            return False, None

        def get(self, *_args: Any, **_kwargs: Any) -> float:  # pragma: no cover - stub simulado
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
        def rotate(frame: Any, _rot: Any) -> Any:  # pragma: no cover - stub simulado
            return frame

        @staticmethod
        def resize(frame: Any, _size: Any, *, interpolation: Any = None) -> Any:  # pragma: no cover - stub simulado
            return frame

        @staticmethod
        def cvtColor(frame: Any, _code: Any) -> Any:  # pragma: no cover - stub simulado
            return frame

    sys.modules["cv2"] = _FakeCv2()

if "numpy" not in sys.modules:
    fake_numpy = ModuleType("numpy")

    def _stub_np(*_args: Any, **_kwargs: Any):  # pragma: no cover - no debería usarse en estas pruebas
        raise NotImplementedError("numpy stub")

    fake_numpy.array = _stub_np
    fake_numpy.arange = _stub_np
    fake_numpy.full = _stub_np
    fake_numpy.where = _stub_np
    fake_numpy.isnan = _stub_np
    fake_numpy.interp = _stub_np
    fake_numpy.linspace = _stub_np
    fake_numpy.isfinite = _stub_np
    fake_numpy.asarray = _stub_np
    fake_numpy.median = _stub_np
    fake_numpy.percentile = _stub_np
    fake_numpy.float32 = float
    fake_numpy.float64 = float
    fake_numpy.int32 = int
    sys.modules["numpy"] = fake_numpy

if "pandas" not in sys.modules:
    fake_pandas = ModuleType("pandas")

    class _FakeDataFrame:  # pragma: no cover - marcador mínimo
        empty = True

        def copy(self, *_args: Any, **_kwargs: Any) -> "_FakeDataFrame":
            return self

        def to_csv(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - stub simulado
            pass

        @property
        def columns(self):  # pragma: no cover - stub simulado
            return []

    def _stub_pd(*_args: Any, **_kwargs: Any):  # pragma: no cover - no debería ejecutarse aquí
        raise NotImplementedError("pandas stub")

    fake_pandas.DataFrame = _FakeDataFrame
    fake_pandas.Series = _FakeDataFrame
    fake_pandas.notna = _stub_pd
    fake_pandas.isna = _stub_pd
    sys.modules["pandas"] = fake_pandas

if "scipy" not in sys.modules:
    fake_scipy = ModuleType("scipy")
    fake_signal = ModuleType("scipy.signal")

    def _stub_scipy(*_args: Any, **_kwargs: Any):  # pragma: no cover - no debería usarse en estas pruebas
        raise NotImplementedError("scipy stub")

    fake_signal.find_peaks = _stub_scipy
    fake_signal.savgol_filter = _stub_scipy
    fake_scipy.signal = fake_signal
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.signal"] = fake_signal

from src import config
from src.A_preprocessing.video_metadata import VideoInfo
from src.C_analysis import run_pipeline
from src.C_analysis.errors import NoFramesExtracted, VideoOpenError
from src.C_analysis.streaming import StreamingPoseResult


def _make_cfg(tmp_path: Path) -> config.Config:
    cfg = config.Config()
    base_dir = tmp_path / "outputs"
    cfg.output.base_dir = base_dir
    cfg.output.counts_dir = base_dir / "counts"
    cfg.output.poses_dir = base_dir / "poses"
    cfg.debug.generate_debug_video = False
    cfg.debug.debug_mode = False
    return cfg


def test_run_pipeline_raises_video_open_error(monkeypatch, tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    video_path = tmp_path / "missing_video.mp4"

    def _fail_open(_path: str) -> None:
        raise VideoOpenError("could not open video")

    monkeypatch.setattr("src.C_analysis.pipeline.open_video_cap", _fail_open)

    with pytest.raises(VideoOpenError):
        run_pipeline(str(video_path), cfg)


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

    class _DummyCap:
        def release(self) -> None:
            pass

        def get(self, *_args: Any, **_kwargs: Any) -> float:
            return 30.0

    monkeypatch.setattr("src.C_analysis.pipeline.open_video_cap", lambda _path: _DummyCap())
    monkeypatch.setattr(
        "src.C_analysis.pipeline.read_info_and_initial_sampling",
        lambda _cap, _video_path: (fake_info, 30.0, None, False),
    )
    monkeypatch.setattr(
        "src.C_analysis.pipeline.extract_processed_frames_stream",
        lambda **_: [],
    )
    monkeypatch.setattr(
        "src.C_analysis.pipeline.stream_pose_and_detection",
        lambda *_args, **_kwargs: StreamingPoseResult(
            df_landmarks=sys.modules["pandas"].DataFrame(),
            frames_processed=0,
            detection=None,
            debug_video_path=None,
            processed_size=None,
        ),
    )

    with pytest.raises(NoFramesExtracted):
        run_pipeline(str(video_path), cfg)
