# tests/test_metrics_output.py

import os
import pandas as pd

METRICS_CSV = "data/processed/poses/1-Squat_Own_metrics.csv"

def test_metrics_csv_exists():
    """
    Comprueba que el CSV de métricas generado por el subcomando 'metrics' existe.
    """
    assert os.path.isfile(METRICS_CSV), f"No se encontró el fichero de métricas: {METRICS_CSV}"

def test_metrics_csv_columns():
    """
    Carga el CSV de métricas y verifica que tenga las columnas mínimas esperadas.
    """
    df = pd.read_csv(METRICS_CSV)
    # Columnas que siempre deberían aparecer
    expected_cols = {
        "frame_idx",
        "rodilla_izq", "rodilla_der",
        "codo_izq", "codo_der",
        "anchura_hombros", "separacion_pies",
        "vel_ang_rod_izq", "vel_ang_rod_der",
        "vel_ang_codo_izq", "vel_ang_codo_der",
        "sim_rodilla", "sim_codo"
    }
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    assert not missing, f"Faltan columnas en el CSV de métricas: {missing}"
    # Opcional: comprobar que no esté vacío
    assert len(df) > 0, "El CSV de métricas está vacío."
