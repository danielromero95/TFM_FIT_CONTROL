import pandas as pd

from src.ui.metrics_sync.component import _build_payload


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
