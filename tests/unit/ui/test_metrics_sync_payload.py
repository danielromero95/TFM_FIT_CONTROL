import pandas as pd

from src.ui.metrics_sync.viewer import _build_payload


def test_build_payload_stride_and_nan_mapping():
    n = 105
    fps = 50.0
    # Pure-Python sequences (avoid NumPy dependency in the test)
    frame_idx = [float(i) for i in range(n)]
    a = [ (float(i)/(n-1)) if n > 1 else 0.0 for i in range(n) ]
    b = [ 10.0 + 10.0*(float(i)/(n-1)) if n > 1 else 10.0 for i in range(n) ]

    # Inject NaNs
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

    payload = _build_payload(df, ["metric_a", "metric_b", "missing"], fps=fps, max_points=20)

    # x_mode and fps kept
    assert payload["x_mode"] == "time"
    assert abs(payload["fps"] - fps) < 1e-12

    # Stride reduced length (≤ max_points)
    assert len(payload["times"]) <= 20

    # Selected series present in output; missing columns ignored
    assert set(payload["series"].keys()) == {"metric_a", "metric_b"}

    # NaNs mapped to None (JSON null)
    def has_none(lst):
        return any(v is None for v in lst)

    assert has_none(payload["series"]["metric_a"])
    assert has_none(payload["series"]["metric_b"])

    # All series lengths match times length
    L = len(payload["times"])
    assert len(payload["series"]["metric_a"]) == L
    assert len(payload["series"]["metric_b"]) == L
