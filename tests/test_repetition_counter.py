"""Pruebas unitarias para el contador de repeticiones basado en valles."""

from __future__ import annotations

import pandas as pd
import numpy as np

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


def test_state_machine_counts_clean_sine() -> None:
    cfg = _make_cfg(primary_angle="left_knee")
    fps = 30.0
    t = np.linspace(0, 4 * np.pi, 120)
    angles = 150 + 30 * np.cos(t)
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})

    reps, debug = count_repetitions_with_config(df, cfg, fps=fps)

    assert reps == 2
    assert len(debug.valley_indices) == reps


def test_state_machine_handles_noise_and_short_nans() -> None:
    rng = np.random.default_rng(123)
    t = np.linspace(0, 2 * np.pi, 90)
    clean = 140 + 25 * np.cos(t)
    noisy = clean + rng.normal(0, 2, size=clean.shape)
    noisy[20:22] = np.nan  # gap dentro del límite
    df = pd.DataFrame({"left_knee": noisy, "pose_ok": 1.0})
    cfg = _make_cfg(primary_angle="left_knee")

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0)

    assert reps == 1


def test_counts_without_initial_peak_and_requires_recovery() -> None:
    angles = [
        90,
        120,
        150,
        170,
        150,
        120,
        90,
        115,
        140,
        165,
    ]
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        min_prominence=15.0,
        min_distance_sec=0.1,
        refractory_sec=0.0,
    )

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0)

    assert reps == 2
    assert debug.valley_indices == [0, 6]


def test_does_not_count_last_incomplete_valley() -> None:
    angles = [
        170,
        150,
        120,
        90,
        120,
        150,
        170,
        150,
        120,
        95,
        150,
        170,
        150,
        120,
        90,
        95,
    ]
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        min_prominence=25.0,
        min_distance_sec=0.1,
        refractory_sec=0.0,
    )

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0)

    assert reps == 2
    assert debug.valley_indices == [3, 9]


def test_state_machine_ignores_long_gaps() -> None:
    t = np.linspace(0, 2 * np.pi, 90)
    signal = 150 + 20 * np.cos(t)
    signal[30:40] = np.nan  # mayor que el hueco permitido
    df = pd.DataFrame({"left_knee": signal, "pose_ok": 1.0})
    cfg = _make_cfg(primary_angle="left_knee")

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0)

    assert reps == 0
