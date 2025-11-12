"""Pruebas unitarias para el contador de repeticiones basado en valles."""

from __future__ import annotations

import pandas as pd

from src import config
from src.C_analysis.repetition_counter import count_repetitions_with_config


def _make_cfg(**overrides):
    cfg = config.CountingConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_count_repetitions_detects_two_valleys() -> None:
    """Dos valles bien separados deberían producir dos repeticiones."""
    angles = [
        170,
        150,
        120,
        90,
        120,
        150,
        170,
        150,
        110,
        80,
        110,
        150,
        170,
    ]
    df = pd.DataFrame({"left_knee": angles})
    cfg = _make_cfg(
        primary_angle="left_knee",
        min_prominence=5.0,
        min_distance_sec=0.1,
        refractory_sec=0.0,
    )

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0)

    assert reps == 2
    assert debug.valley_indices == [3, 9]
    assert debug.prominences and len(debug.prominences) == 2


def test_count_repetitions_handles_missing_column() -> None:
    """Si falta la columna principal se debe devolver cero repeticiones y depuración vacía."""
    df = pd.DataFrame({"right_knee": [170, 160, 150]})
    cfg = _make_cfg(primary_angle="left_knee")

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0)

    assert reps == 0
    assert debug.valley_indices == []
    assert debug.prominences == []


def test_refractory_filter_keeps_most_prominent_valley() -> None:
    """Los valles dentro de la ventana refractaria deben consolidarse en el más profundo."""
    angles = [
        170,
        155,
        130,
        85,
        120,
        90,
        150,
        170,
    ]
    df = pd.DataFrame({"left_knee": angles})
    cfg = _make_cfg(
        primary_angle="left_knee",
        min_prominence=5.0,
        min_distance_sec=0.1,
        refractory_sec=0.2,
    )

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0)

    assert reps == 1
    assert debug.valley_indices == [3]
    assert len(debug.prominences) == 1
