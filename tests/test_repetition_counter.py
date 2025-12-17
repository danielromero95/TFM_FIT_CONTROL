"""Pruebas unitarias para el contador de repeticiones basado en valles."""

from __future__ import annotations

import pandas as pd
import numpy as np

import numpy as np
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


def test_deadlift_front_counts_with_auto_hip_thresholds() -> None:
    right_hip = [
        None,
        None,
        None,
        None,
        None,
        136.063,
        132.363,
        128.52,
        129.34,
        130.133,
        131.222,
        132.486,
        133.747,
        135.179,
        137.114,
        140.301,
        169.924,
        170.665,
        170.413,
        170.108,
        169.786,
        169.592,
        169.699,
        169.843,
        169.942,
        170.028,
        170.033,
        169.947,
        169.723,
        169.37,
        168.941,
        168.458,
        167.898,
        167.179,
        162.758,
        155.352,
        148.786,
        143.845,
        140.236,
        137.642,
        135.832,
        134.595,
        133.915,
        133.697,
        133.894,
        134.498,
        135.433,
        136.514,
        137.515,
        138.279,
        138.719,
        138.799,
        138.543,
        137.992,
        133.99,
        134.277,
        134.943,
        135.97,
        137.208,
        138.55,
        139.902,
        141.214,
        142.483,
        143.699,
        144.836,
        145.883,
        146.804,
        172.03,
        171.893,
        171.617,
        171.253,
        170.826,
        170.356,
        169.881,
        169.461,
        169.15,
        168.955,
        168.862,
        168.805,
        168.733,
        168.559,
        168.231,
        167.724,
        167.056,
        162.89,
        155.816,
        149.355,
        144.256,
        140.486,
        137.726,
        135.792,
        134.451,
        133.49,
        132.748,
        132.143,
        136.63,
        137.11,
        137.994,
        139.22,
        140.65,
        142.145,
        143.607,
        144.985,
        146.268,
        147.452,
        148.534,
        149.517,
        150.408,
        151.214,
        151.941,
        171.216,
        170.938,
        170.489,
        169.942,
        169.375,
        168.851,
        168.402,
        168.019,
        167.66,
        167.265,
        166.784,
        166.18,
        161.879,
        154.624,
        148.067,
        142.977,
        139.222,
        136.494,
        134.625,
        133.465,
        132.832,
        132.559,
        132.521,
        132.642,
        132.877,
        133.198,
        133.586,
        134.043,
        134.586,
        135.226,
        136.002,
        136.978,
        138.23,
        139.798,
        143.793,
        145.254,
        147.098,
        149.239,
        151.533,
        153.822,
        155.967,
        157.869,
    ]
    pose_ok = [0] * 5 + [1] * (len(right_hip) - 5)
    df = pd.DataFrame({"right_hip": right_hip, "pose_ok": pose_ok})

    cfg = _make_cfg(exercise="deadlift", primary_angle="right_hip")
    reps, debug = count_repetitions_with_config(df, cfg, fps=10.0)

    assert reps == 3
    assert len(debug.valley_indices) == 3
    expected = [54, 93, 140]
    for found, target in zip(debug.valley_indices, expected):
        assert abs(found - target) <= 15
