import math
import sys
import types
from types import ModuleType, SimpleNamespace


# Provide lightweight stubs for heavy optional dependencies so analysis_service imports cleanly.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(resize=lambda frame, size: frame)

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
    fake_scipy.signal = fake_signal
    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.signal"] = fake_signal

from src.services.analysis_service import _plan_sampling


def make_cfg(target_fps=None, manual_sample_rate=None):
    # Duck-typing: sólo necesitamos .video.target_fps y .video.manual_sample_rate
    return SimpleNamespace(
        video=SimpleNamespace(target_fps=target_fps, manual_sample_rate=manual_sample_rate)
    )


def test_plan_sampling_metadata_valid_with_target_fps():
    cfg = make_cfg(target_fps=30.0)
    sample_rate, fps_eff, fps_base, warnings = _plan_sampling(
        fps_metadata=60.0,
        fps_from_reader=0.0,
        prefer_reader_fps=False,
        initial_sample_rate=1,
        cfg=cfg,
        fps_warning=None,
    )
    assert sample_rate == 2
    assert math.isclose(fps_eff, 30.0, rel_tol=1e-6)
    assert math.isclose(fps_base, 60.0, rel_tol=1e-6)
    assert warnings == []


def test_plan_sampling_prefers_reader_when_metadata_invalid():
    cfg = make_cfg(target_fps=None)
    sample_rate, fps_eff, fps_base, warnings = _plan_sampling(
        fps_metadata=0.0,
        fps_from_reader=29.97,
        prefer_reader_fps=True,
        initial_sample_rate=1,
        cfg=cfg,
        fps_warning=None,
    )
    assert sample_rate == 1
    assert math.isclose(fps_eff, 29.97, rel_tol=1e-6)
    assert math.isclose(fps_base, 29.97, rel_tol=1e-6)
    assert warnings == []


def test_plan_sampling_fallback_to_one_fps_when_both_invalid():
    cfg = make_cfg(target_fps=30.0)  # con 1 fps, sample_rate se queda en 1 (max(1, round(1/30)))
    sample_rate, fps_eff, fps_base, warnings = _plan_sampling(
        fps_metadata=0.0,
        fps_from_reader=0.0,
        prefer_reader_fps=False,
        initial_sample_rate=1,
        cfg=cfg,
        fps_warning=None,
    )
    assert sample_rate == 1
    assert math.isclose(fps_eff, 1.0, rel_tol=1e-6)
    assert math.isclose(fps_base, 1.0, rel_tol=1e-6)
    assert any("Unable to determine a valid FPS" in w for w in warnings)


def test_plan_sampling_respects_initial_sample_rate_gt_one():
    cfg = make_cfg(target_fps=10.0)
    sample_rate, fps_eff, fps_base, _ = _plan_sampling(
        fps_metadata=60.0,
        fps_from_reader=0.0,
        prefer_reader_fps=False,
        initial_sample_rate=3,  # ya extrajimos con stride 3
        cfg=cfg,
        fps_warning=None,
    )
    # No debe re-submuestrear; se mantiene el initial_sample_rate
    assert sample_rate == 3
    assert math.isclose(fps_base, 60.0, rel_tol=1e-6)
    assert math.isclose(fps_eff, 60.0 / 3.0, rel_tol=1e-6)


def test_plan_sampling_appends_fps_warning_message():
    cfg = make_cfg(target_fps=None)
    msg = "Invalid metadata FPS. Estimated from video duration (25.00 fps)."
    sample_rate, fps_eff, fps_base, warnings = _plan_sampling(
        fps_metadata=25.0,
        fps_from_reader=0.0,
        prefer_reader_fps=False,
        initial_sample_rate=1,
        cfg=cfg,
        fps_warning=msg,
    )
    assert sample_rate == 1
    assert math.isclose(fps_base, 25.0, rel_tol=1e-6)
    assert math.isclose(fps_eff, 25.0, rel_tol=1e-6)
    assert any("FPS final utilizado" in w for w in warnings)
