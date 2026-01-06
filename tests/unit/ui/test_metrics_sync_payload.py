import pandas as pd
import pytest

from src.ui.metrics_sync.component import _build_payload, _map_rep_intervals_to_axis


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


def test_axis_times_use_source_time_with_auto_offset():
    source_time = [0.05, 0.12, 0.18, 0.27]
    df = pd.DataFrame(
        {
            "source_time_s": source_time,
            "source_frame_idx": [1, 2, 3, 4],
            "metric": [0.0, 0.5, 1.0, 1.5],
        }
    )

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=10)

    expected_axis = [v - source_time[0] for v in source_time]

    assert payload["axis_times"] == pytest.approx(expected_axis)
    assert payload["time_offset_s"] == pytest.approx(source_time[0])
    assert payload["axis_ref"] == "source"


def test_time_offset_applied_when_requested():
    time_values = [1.5, 1.54, 1.6]
    df = pd.DataFrame({"time_s": time_values, "metric": [1, 2, 3]})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=10, time_offset_s=1.5)

    assert payload["time_offset_s"] == pytest.approx(1.5)
    assert payload["axis_times"][0] == pytest.approx(0.0)
    assert payload["times"][0] == pytest.approx(0.0)
    assert payload["axis_times"][-1] == pytest.approx(time_values[-1] - 1.5)


def test_source_time_offset_preserved_and_rebased():
    source_time = [1.2, 1.25, 1.30]
    df = pd.DataFrame(
        {
            "source_time_s": source_time,
            "source_frame_idx": [10, 11, 12],
            "analysis_frame_idx": [0, 1, 2],
            "metric": [0.1, 0.2, 0.3],
        }
    )

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=10, time_offset_s=source_time[0])

    assert payload["axis_ref"] == "source"
    assert payload["time_offset_s"] == pytest.approx(source_time[0])
    assert payload["axis_times"][0] == pytest.approx(0.0)
    assert payload["axis_times"][-1] == pytest.approx(source_time[-1] - source_time[0])


def test_auto_offset_from_time_s_when_source_missing():
    time_values = [2.0, 2.05, 2.10]
    df = pd.DataFrame({"time_s": time_values, "metric": [1, 2, 3]})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=10)

    assert payload["time_offset_s"] == pytest.approx(time_values[0])
    assert payload["axis_times"][0] == pytest.approx(0.0)
    assert payload["axis_times"][-1] == pytest.approx(time_values[-1] - time_values[0])


def test_axis_times_are_not_downsampled():
    n = 5
    times = [0.0, 0.02, 0.04, 0.06, 0.08]
    df = pd.DataFrame({"source_time_s": times, "metric": range(n)})

    payload = _build_payload(df, ["metric"], fps=30.0, max_points=2)

    assert len(payload["axis_times"]) == n
    assert len(payload["times"]) < n
    assert payload["times"][0] == pytest.approx(payload["axis_times"][0])
    assert payload["times"][1] == pytest.approx(payload["axis_times"][3])


def test_rep_intervals_are_mapped_to_source_domain():
    df = pd.DataFrame(
        {
            "analysis_frame_idx": [0, 1, 2, 3],
            "source_frame_idx": [10, 12, 14, 16],
            "metric": [0.0, 0.5, 1.0, 1.5],
        }
    )

    mapped = _map_rep_intervals_to_axis([(0, 2), (2, 3)], df, axis_ref="source")

    assert mapped == [(10, 14), (14, 16)]
