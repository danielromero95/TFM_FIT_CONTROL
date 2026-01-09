import pandas as pd
import pytest

from src.ui.metrics_sync.component import _build_payload, _rep_intervals_to_seconds


def test_build_payload_stride_and_nan_mapping():
    n = 105
    fps = 50.0
    # Secuencias puramente en Python (evitamos depender de NumPy en la prueba)
    frame_idx = [float(i) for i in range(n)]
    a = [ (float(i)/(n-1)) if n > 1 else 0.0 for i in range(n) ]
    b = [ 10.0 + 10.0*(float(i)/(n-1)) if n > 1 else 10.0 for i in range(n) ]

    # Inyectar NaN
    nan = float("nan")
    a[5] = nan
    b[7] = nan
    b[-1] = nan

    df = pd.DataFrame({
        "frame_idx": frame_idx,
        "metric_a": a,
        "metric_b": b,
        "non_numeric": ["x"] * n,
    })

    payload = _build_payload(df, ["metric_a", "metric_b", "missing"], fps=fps, max_points=0)

    # Se conservan x_mode y el fps
    assert payload["x_mode"] == "time"
    assert abs(payload["fps"] - fps) < 1e-12

    # Sin reducción de muestras cuando max_points=0
    assert len(payload["times"]) == n

    # Las series seleccionadas aparecen en la salida; se ignoran columnas ausentes
    assert set(payload["series"].keys()) == {"metric_a", "metric_b"}

    # Los NaN se mapean a None (nulo en JSON)
    def has_none(lst):
        return any(v is None for v in lst)

    assert has_none(payload["series"]["metric_a"])
    assert has_none(payload["series"]["metric_b"])

    # Todas las series tienen la misma longitud que ``times``
    L = len(payload["times"])
    assert len(payload["series"]["metric_a"]) == L
    assert len(payload["series"]["metric_b"]) == L


def test_axis_times_use_cfr_timebase_even_with_source_time():
    source_time = [0.05, 0.12, 0.18, 0.27]
    df = pd.DataFrame(
        {
            "source_time_s": source_time,
            "source_frame_idx": [1, 2, 3, 4],
            "metric": [0.0, 0.5, 1.0, 1.5],
        }
    )

    fps = 30.0
    payload = _build_payload(df, ["metric"], fps=fps, max_points=10)

    expected_axis = [i / fps for i in range(len(source_time))]

    assert payload["axis_times"] == pytest.approx(expected_axis)
    assert payload["axis_times"][-1] == pytest.approx((len(source_time) - 1) / fps)


def test_axis_times_ignore_time_offset_input():
    time_values = [1.5, 1.54, 1.6]
    df = pd.DataFrame({"time_s": time_values, "metric": [1, 2, 3]})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=10, time_offset_s=1.5)

    assert payload["axis_times"][0] == pytest.approx(0.0)
    assert payload["axis_times"][-1] == pytest.approx((len(time_values) - 1) / 30.0)


def test_axis_times_are_not_downsampled():
    n = 5
    times = [0.0, 0.02, 0.04, 0.06, 0.08]
    df = pd.DataFrame({"source_time_s": times, "metric": range(n)})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=2)

    assert len(payload["axis_times"]) == n
    assert len(payload["times"]) < n
    assert payload["times"][0] == pytest.approx(payload["axis_times"][0])
    assert payload["times"][1] == pytest.approx(payload["axis_times"][3])


def test_rep_intervals_converted_to_seconds():
    mapped = _rep_intervals_to_seconds([(0, 2), (2, 3)], fps_video=10.0)

    assert mapped == [(0.0, 0.2), (0.2, 0.3)]


def test_axis_times_match_cfr_duration():
    n = 12
    fps = 6.0
    df = pd.DataFrame({"metric": range(n)})

    payload = _build_payload(df, ["metric"], fps=fps, max_points=10)

    expected_duration = (n - 1) / fps
    assert payload["axis_times"][-1] == pytest.approx(expected_duration)
