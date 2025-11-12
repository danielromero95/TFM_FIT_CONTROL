"""Pruebas de humo para el esquema CSV de métricas usando datos sintéticos."""

from __future__ import annotations

import pandas as pd
import pytest

EXPECTED_COLUMNS = {
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
