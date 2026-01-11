import math

from src.ui.metrics_sync.bands import build_phase_bands


def test_phase_bands_skip_zero_length_down() -> None:
    bands, _ = build_phase_bands([(0.0, 0.0, 1.0)], fps=30.0, x_mode="time")

    assert len(bands) == 1
    assert bands[0]["phase"] == "up"


def test_phase_bands_skip_zero_length_up() -> None:
    bands, _ = build_phase_bands([(0.0, 1.0, 1.0)], fps=30.0, x_mode="time")

    assert len(bands) == 1
    assert bands[0]["phase"] == "down"


def test_phase_bands_preserve_end_beyond_samples() -> None:
    bands, max_end = build_phase_bands([(0.0, 0.5, 1.2)], fps=30.0, x_mode="time")

    assert bands
    assert math.isclose(max_end or 0.0, 1.2)
    assert bands[-1]["x1"] == 1.2


def test_phase_bands_keep_normal_reps() -> None:
    bands, max_end = build_phase_bands([(0.0, 0.5, 1.0)], fps=10.0, x_mode="time")

    assert len(bands) == 2
    assert bands[0]["phase"] == "down"
    assert bands[0]["x0"] == 0.0
    assert bands[0]["x1"] == 0.5
    assert bands[1]["phase"] == "up"
    assert bands[1]["x0"] == 0.5
    assert bands[1]["x1"] == 1.0
    assert max_end == 1.0


def test_phase_bands_skip_micro_rep_and_preserve_max_end() -> None:
    bands, max_end = build_phase_bands(
        [(0.0, 0.5, 1.0), (1.1, 1.16, 1.22)], fps=30.0, x_mode="time"
    )

    assert len(bands) == 2
    assert max_end == 1.0
