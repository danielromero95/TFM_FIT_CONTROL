"""Pruebas unitarias para el contador de repeticiones basado en valles."""

from __future__ import annotations

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

    assert reps == 1


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


def test_deadlift_front_counts_three_reps_with_warmup_gap() -> None:
    right_hip = [
        None,
        None,
        None,
        None,
        None,
        130.757,
        129.143,
        128.498,
        133.184,
        138.229,
        139.282,
        142.161,
        145.095,
        152.499,
        158.111,
        162.87,
        170.022,
        171.125,
        170.806,
        170.363,
        169.878,
        169.275,
        168.604,
        167.854,
        166.911,
        165.715,
        164.302,
        162.911,
        161.826,
        161.083,
        160.571,
        160.174,
        159.767,
        159.251,
        158.554,
        157.606,
        156.283,
        154.748,
        152.936,
        151.09,
        149.099,
        147.019,
        144.913,
        142.856,
        140.946,
        139.321,
        138.123,
        137.437,
        137.26,
        137.514,
        138.07,
        138.781,
        139.516,
        140.171,
        140.677,
        140.995,
        141.109,
        141.028,
        140.782,
        140.41,
        139.948,
        139.414,
        138.795,
        138.033,
        137.026,
        135.677,
        133.931,
        131.84,
        129.623,
        127.646,
        126.241,
        125.582,
        125.659,
        126.311,
        127.343,
        128.598,
        129.939,
        131.229,
        132.338,
        133.183,
        133.744,
        134.058,
        134.207,
        134.281,
        134.355,
        134.475,
        134.672,
        134.968,
        135.38,
        135.915,
        136.571,
        137.339,
        138.203,
        139.145,
        140.145,
        141.19,
        142.279,
        143.42,
        144.624,
        145.901,
        147.251,
        148.648,
        150.033,
        151.32,
        152.423,
        153.276,
        153.862,
        154.227,
        154.453,
        154.615,
        154.758,
        154.906,
        155.079,
        155.303,
        155.613,
        156.05,
        156.663,
        157.5,
        158.571,
        159.807,
        161.058,
        162.143,
        162.921,
        163.318,
        163.322,
        162.996,
        162.451,
        161.815,
        161.205,
        160.705,
        160.363,
        160.184,
        160.144,
        160.207,
        160.334,
        160.49,
        160.639,
        160.748,
        160.785,
        160.72,
        160.528,
        160.196,
        159.722,
        159.116,
        158.387,
        157.546,
        156.609,
        155.6,
        154.55,
        153.495,
        152.469,
        151.493,
        150.574,
        149.707,
        148.881,
        148.078,
        147.277,
        146.461,
        145.629,
        144.789,
        143.957,
    ]
    pose_ok = [0] * 5 + [1] * (len(right_hip) - 5)
    df = pd.DataFrame({"right_hip": right_hip, "pose_ok": pose_ok})

    cfg = _make_cfg(
        exercise="deadlift",
        primary_angle="right_hip",
        min_angle_excursion_deg=15.0,
        min_prominence=10.0,
        min_distance_sec=0.4,
        refractory_sec=0.2,
    )

    reps, debug = count_repetitions_with_config(df, cfg, fps=10.0)

    assert reps == 3
    assert len(debug.valley_indices) == 3
