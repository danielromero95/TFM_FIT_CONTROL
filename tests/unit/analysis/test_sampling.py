import math
import sys
from types import ModuleType, SimpleNamespace


# Provide lightweight stubs for heavy optional dependencies so the analysis package imports cleanly.
if "cv2" not in sys.modules:
    class _FakeVideoCapture:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._opened = False

        def isOpened(self) -> bool:
            return self._opened

        def release(self) -> None:  # pragma: no cover - not exercised
            self._opened = False

        def set(self, *_args: object, **_kwargs: object) -> bool:  # pragma: no cover - stubbed
            return False

        def read(self) -> tuple[bool, object]:  # pragma: no cover - not exercised
            return False, None

        def get(self, *_args: object, **_kwargs: object) -> float:  # pragma: no cover - stubbed
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
        def rotate(frame: object, _rot: object) -> object:  # pragma: no cover - stubbed
            return frame

        @staticmethod
        def resize(frame: object, _size: object, *, interpolation: object = None) -> object:  # pragma: no cover - stubbed
            return frame

        @staticmethod
        def cvtColor(frame: object, _code: object) -> object:  # pragma: no cover - stubbed
            return frame

    sys.modules["cv2"] = _FakeCv2()

if "numpy" not in sys.modules:
    fake_numpy = ModuleType("numpy")
    fake_numpy.nan = float("nan")

    def _stub_np(*args, **kwargs):  # pragma: no cover - should not be called in these tests
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

    def _stub_pd(*args, **kwargs):  # pragma: no cover - should not be called in these tests
        raise NotImplementedError("pandas stub")

    fake_pandas.DataFrame = _FakeDataFrame
    fake_pandas.Series = _FakeDataFrame
    fake_pandas.notna = _stub_pd
    fake_pandas.isna = _stub_pd
    sys.modules["pandas"] = fake_pandas

if "scipy" not in sys.modules:
    fake_scipy = ModuleType("scipy")
    fake_signal = ModuleType("scipy.signal")

    def _stub_scipy(*args, **kwargs):  # pragma: no cover - should not be called in these tests
        raise NotImplementedError("scipy stub")

    fake_signal.find_peaks = _stub_scipy
    fake_signal.savgol_filter = _stub_scipy
    fake_scipy.signal = fake_signal
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.signal"] = fake_signal

from src.C_analysis import SamplingPlan, make_sampling_plan


def make_cfg(target_fps=None, manual_sample_rate=None):
    # Duck-typing: only ``video.target_fps`` and ``video.manual_sample_rate`` are required.
    return SimpleNamespace(
        video=SimpleNamespace(target_fps=target_fps, manual_sample_rate=manual_sample_rate)
    )


def test_make_sampling_plan_metadata_target_downsamples_correctly():
    cfg = make_cfg(target_fps=10.0)
    plan = make_sampling_plan(
        fps_metadata=30.0,
        fps_from_reader=0.0,
        prefer_reader_fps=False,
        initial_sample_rate=1,
        cfg=cfg,
        fps_warning=None,
    )
    assert isinstance(plan, SamplingPlan)
    assert plan.sample_rate == 3
    assert math.isclose(plan.fps_effective, 10.0, rel_tol=1e-6)
    assert math.isclose(plan.fps_base, 30.0, rel_tol=1e-6)
    assert plan.warnings == []


def test_make_sampling_plan_respects_existing_stride():
    cfg = make_cfg(target_fps=10.0)
    plan = make_sampling_plan(
        fps_metadata=30.0,
        fps_from_reader=0.0,
        prefer_reader_fps=False,
        initial_sample_rate=3,
        cfg=cfg,
        fps_warning=None,
    )
    assert plan.sample_rate == 3
    assert math.isclose(plan.fps_effective, 10.0, rel_tol=1e-6)


def test_make_sampling_plan_prefers_reader_fps_when_requested():
    cfg = make_cfg(target_fps=10.0)
    plan = make_sampling_plan(
        fps_metadata=0.0,
        fps_from_reader=25.0,
        prefer_reader_fps=True,
        initial_sample_rate=1,
        cfg=cfg,
        fps_warning="Reader FPS selected",
    )
    assert math.isclose(plan.fps_base, 25.0, rel_tol=1e-6)
    assert plan.sample_rate == 2  # round(25/10) == 2
    assert math.isclose(plan.fps_effective, 12.5, rel_tol=1e-6)
    assert any("Using FPS value" in warning for warning in plan.warnings)


def test_make_sampling_plan_falls_back_to_single_fps_with_warning():
    cfg = make_cfg(target_fps=15.0)
    plan = make_sampling_plan(
        fps_metadata=0.0,
        fps_from_reader=0.0,
        prefer_reader_fps=False,
        initial_sample_rate=1,
        cfg=cfg,
        fps_warning=None,
    )
    assert plan.sample_rate == 1
    assert math.isclose(plan.fps_base, 1.0, rel_tol=1e-6)
    assert math.isclose(plan.fps_effective, 1.0, rel_tol=1e-6)
    assert any("Unable to determine a valid FPS" in warning for warning in plan.warnings)
