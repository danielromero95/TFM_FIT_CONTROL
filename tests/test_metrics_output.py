"""Pruebas de humo para el esquema CSV de métricas usando datos sintéticos."""

from __future__ import annotations

import pandas as pd
import pytest

from src.C_analysis.metrics import compute_metrics_and_angle
from src.C_analysis.pipeline import _attach_temporal_columns

EXPECTED_COLUMNS = {
    "analysis_frame_idx",
    "source_frame_idx",
    "time_s",
    "frame_idx",
    "left_knee",
    "right_knee",
    "left_elbow",
    "right_elbow",
    "shoulder_width",
    "foot_separation",
    "ang_vel_left_knee",
    "ang_vel_right_knee",
    "ang_vel_left_elbow",
    "ang_vel_right_elbow",
    "knee_symmetry",
    "elbow_symmetry",
}


@pytest.fixture
def metrics_csv(tmp_path):
    """Crea un CSV de métricas representativo para pruebas portables de regresión."""

    rows = [{col: float(idx) for col in EXPECTED_COLUMNS} for idx in range(5)]
    df = pd.DataFrame(rows)
    path = tmp_path / "metrics.csv"
    df.to_csv(path, index=False)
    return path


def test_metrics_csv_exists(metrics_csv):
    """Verifica que el archivo CSV sintético de métricas se haya creado."""

    assert metrics_csv.exists(), f"Metrics file not created: {metrics_csv}"


def test_metrics_csv_columns(metrics_csv):
    """Comprueba que el CSV de métricas exponga el conjunto mínimo de columnas esperado."""

    df = pd.read_csv(metrics_csv)
    actual_cols = set(df.columns)
    missing = EXPECTED_COLUMNS - actual_cols
    assert not missing, f"Missing columns in metrics CSV: {missing}"
    assert len(df) == 5, "Synthetic metrics CSV size changed unexpectedly."


def test_attach_temporal_columns_are_monotonic():
    """La columna ``time_s`` debe conservar monotonicidad y duración esperada."""

    metrics_df = pd.DataFrame({"frame_idx": [0, 1, 2, 3], "left_knee": [10, 11, 12, 13]})
    raw_df = pd.DataFrame(
        {
            "analysis_frame_idx": [0, 1, 2, 3],
            "source_frame_idx": [0, 2, 4, 6],
            "time_s": [0.0, 0.25, 0.50, 0.75],
        }
    )

    enriched = _attach_temporal_columns(metrics_df, raw_df)

    assert enriched["time_s"].is_monotonic_increasing
    assert pytest.approx(enriched["time_s"].iloc[-1], rel=1e-6) == 0.75
    assert list(enriched["analysis_frame_idx"]) == [0, 1, 2, 3]
    assert list(enriched["source_frame_idx"]) == [0, 2, 4, 6]


def test_nearest_time_maps_to_same_index():
    """El mapeo por tiempo debe devolver el mismo registro más cercano."""

    metrics_df = pd.DataFrame({"frame_idx": [0, 1, 2], "left_knee": [1.0, 2.0, 3.0]})
    raw_df = pd.DataFrame(
        {
            "analysis_frame_idx": [0, 1, 2],
            "source_frame_idx": [0, 1, 2],
            "time_s": [0.0, 0.4, 0.8],
        }
    )

    enriched = _attach_temporal_columns(metrics_df, raw_df)

    for target_t in (0.01, 0.39, 0.65):
        nearest_idx = (enriched["time_s"] - target_t).abs().idxmin()
        assert enriched.loc[nearest_idx, "analysis_frame_idx"] == nearest_idx


def test_attach_temporal_columns_pads_missing_rows():
    """Si el pipeline filtra filas, se deben rellenar huecos con NaN para conservar el eje."""

    metrics_df = pd.DataFrame({"analysis_frame_idx": [0, 3], "left_knee": [10.0, 11.0]})
    raw_df = pd.DataFrame(
        {
            "analysis_frame_idx": [0, 1, 2, 3],
            "source_frame_idx": [0, 1, 2, 3],
            "time_s": [0.0, 0.1, 0.2, 0.3],
        }
    )

    enriched = _attach_temporal_columns(metrics_df, raw_df)

    assert len(enriched) == len(raw_df)
    assert list(enriched["analysis_frame_idx"]) == [0, 1, 2, 3]
    assert enriched.loc[1, "left_knee"] is None or pd.isna(enriched.loc[1, "left_knee"])
    assert enriched.loc[2, "left_knee"] is None or pd.isna(enriched.loc[2, "left_knee"])
    assert pytest.approx(enriched["time_s"].iloc[-1], rel=1e-6) == 0.3


def test_attach_temporal_columns_uses_raw_index_when_present():
    """El eje temporal debe respetar el índice de análisis original cuando existe."""

    metrics_df = pd.DataFrame({"analysis_frame_idx": [10, 11, 12], "left_knee": [5.0, 6.0, 7.0]})
    raw_df = pd.DataFrame(
        {
            "analysis_frame_idx": [10, 11, 12, 13],
            "source_frame_idx": [100, 101, 102, 103],
            "time_s": [1.0, 1.1, 1.2, 1.3],
        }
    )

    enriched = _attach_temporal_columns(metrics_df, raw_df)

    assert len(enriched) == 4
    assert list(enriched["analysis_frame_idx"]) == [10, 11, 12, 13]
    assert list(enriched["source_frame_idx"]) == [100, 101, 102, 103]
    tail_val = enriched.loc[enriched["analysis_frame_idx"] == 13, "left_knee"].iloc[0]
    assert tail_val is None or pd.isna(tail_val)


def test_attach_temporal_columns_respects_kept_indices():
    """La fusión debe preservar huecos internos y mapear métricas al índice original."""

    metrics_df = pd.DataFrame(
        {"analysis_frame_idx": [0, 2, 3], "left_knee": [10.0, 20.0, 30.0]}
    )
    raw_df = pd.DataFrame(
        {
            "analysis_frame_idx": [0, 1, 2, 3],
            "source_frame_idx": [0, 1, 2, 3],
            "time_s": [0.0, 0.1, 0.2, 0.3],
        }
    )

    enriched = _attach_temporal_columns(metrics_df, raw_df)

    assert list(enriched["analysis_frame_idx"]) == [0, 1, 2, 3]
    assert enriched.loc[0, "left_knee"] == 10.0
    assert enriched.loc[1, "left_knee"] is None or pd.isna(enriched.loc[1, "left_knee"])
    assert enriched.loc[2, "left_knee"] == 20.0
    assert enriched.loc[3, "left_knee"] == 30.0


def test_attach_temporal_columns_handles_duplicates_and_gaps():
    """Se eliminan duplicados y se conservan los huecos en el eje temporal base."""

    metrics_df = pd.DataFrame({"analysis_frame_idx": [0, 2, 3], "left_knee": [1.0, 2.0, 3.0]})
    raw_df = pd.DataFrame(
        {
            "analysis_frame_idx": [2, 0, 1, 2, 3],
            "source_frame_idx": [20, 0, 10, 21, 30],
            "time_s": [0.2, 0.0, 0.1, 0.2, 0.3],
        }
    )

    enriched = _attach_temporal_columns(metrics_df, raw_df)

    assert list(enriched["analysis_frame_idx"]) == [0, 1, 2, 3]
    assert list(enriched["source_frame_idx"]) == [0, 10, 20, 30]
    assert enriched.loc[1, "left_knee"] is None or pd.isna(enriched.loc[1, "left_knee"])
    assert enriched.loc[2, "left_knee"] == 2.0
    assert enriched.loc[3, "left_knee"] == 3.0


def test_compute_metrics_and_angle_raises_on_mismatched_indices(monkeypatch):
    """Cuando los índices no cuadran con las métricas se debe lanzar un error claro."""

    fake_metrics = pd.DataFrame({"primary": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(
        "src.C_analysis.metrics.calculate_metrics_from_sequence", lambda *_args, **_kwargs: fake_metrics
    )

    with pytest.raises(ValueError):
        compute_metrics_and_angle(
            df_seq=None,
            primary_angle="primary",
            fps_effective=10.0,
            quality_mask=None,
            analysis_frame_idx=[0, 1],
        )


def test_compute_metrics_and_angle_propagates_analysis_indices(monkeypatch):
    """Los índices de análisis válidos se devuelven alineados con las métricas calculadas."""

    fake_metrics = pd.DataFrame({"primary": [5.0, 6.0]})
    monkeypatch.setattr(
        "src.C_analysis.metrics.calculate_metrics_from_sequence", lambda *_args, **_kwargs: fake_metrics
    )

    (
        df_metrics,
        _angle_range,
        _warnings,
        _skip_reason,
        _chosen_primary,
        used_idx,
    ) = compute_metrics_and_angle(
        df_seq=None,
        primary_angle="primary",
        fps_effective=10.0,
        quality_mask=None,
        analysis_frame_idx=[0, 2],
    )

    assert list(used_idx) == [0, 2]
    pd.testing.assert_frame_equal(df_metrics, fake_metrics)
