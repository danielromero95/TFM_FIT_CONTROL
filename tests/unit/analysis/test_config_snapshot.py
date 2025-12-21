"""Pruebas unitarias para las rutas de salida del *snapshot* de configuración."""

from __future__ import annotations

import sys
from pathlib import Path
import json
from types import ModuleType
from typing import Any


# Proporciona stubs ligeros para dependencias opcionales pesadas, permitiendo importar el paquete de análisis sin problemas.
if "cv2" not in sys.modules:
    class _FakeVideoCapture:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._opened = False

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:  # pragma: no cover - no se ejecuta en estas pruebas
            self._opened = False

        def set(self, *_args: Any, **_kwargs: Any) -> bool:  # pragma: no cover - stub simulado
            return False

        def read(self) -> tuple[bool, Any]:  # pragma: no cover - no se ejecuta en estas pruebas
            return False, None

        def get(self, *_args: Any, **_kwargs: Any) -> float:  # pragma: no cover - stub simulado
            return 0.0

    class _FakeCv2(ModuleType):
        ROTATE_90_CLOCKWISE = 0
        ROTATE_180 = 1
        ROTATE_90_COUNTERCLOCKWISE = 2
        CAP_PROP_POS_MSEC = 0
        CAP_PROP_POS_FRAMES = 1
        COLOR_BGR2GRAY = 10
        INTER_AREA = 0

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

    def _stub_np(*args, **kwargs):  # pragma: no cover - no debería invocarse en estas pruebas
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

    class _FakeDataFrame:  # pragma: no cover - solo para satisfacer las anotaciones
        pass

    def _stub_pd(*args, **kwargs):  # pragma: no cover - no debería invocarse en estas pruebas
        raise NotImplementedError("pandas stub")

    fake_pandas.DataFrame = _FakeDataFrame
    fake_pandas.Series = _FakeDataFrame
    fake_pandas.notna = _stub_pd
    fake_pandas.isna = _stub_pd
    sys.modules["pandas"] = fake_pandas

if "scipy" not in sys.modules:
    fake_scipy = ModuleType("scipy")
    fake_signal = ModuleType("scipy.signal")

    def _stub_scipy(*args, **kwargs):  # pragma: no cover - no debería invocarse en estas pruebas
        raise NotImplementedError("scipy stub")

    fake_signal.find_peaks = _stub_scipy
    fake_signal.savgol_filter = _stub_scipy
    fake_scipy.signal = fake_signal
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.signal"] = fake_signal

from src import config
from src.C_analysis import _prepare_output_paths, _write_used_config


def test_config_snapshot_uses_session_directory(tmp_path) -> None:
    cfg = config.Config()

    base_dir = tmp_path / "outputs"
    cfg.output.base_dir = base_dir
    cfg.output.counts_dir = base_dir / "counts"
    cfg.output.poses_dir = base_dir / "poses"

    video_path = tmp_path / "sample_video.mp4"
    output_paths = _prepare_output_paths(video_path, cfg.output)

    config_path = output_paths.session_dir / "config_used.json"
    expected_path = output_paths.base_dir / Path(video_path).stem / "config_used.json"

    assert config_path == expected_path
    assert config_path.parent == output_paths.session_dir
    assert config_path != output_paths.base_dir / "config_used.json"


def test_config_snapshot_strips_output_section(tmp_path) -> None:
    cfg = config.Config()

    base_dir = tmp_path / "outputs"
    cfg.output.base_dir = base_dir
    cfg.output.poses_dir = base_dir / "poses"

    video_path = tmp_path / "sample_video.mp4"
    output_paths = _prepare_output_paths(video_path, cfg.output)

    written_path = _write_used_config(output_paths.session_dir, cfg)

    saved_config = json.loads(written_path.read_text(encoding="utf-8"))
    assert "output" not in saved_config
