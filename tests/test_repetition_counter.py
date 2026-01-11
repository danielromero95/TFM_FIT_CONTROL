"""Pruebas unitarias para el contador de repeticiones basado en valles."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src import config
from src.C_analysis.rep_candidates import RepCandidate
from src.C_analysis.repetition_counter import (
    _filter_reps_by_thresholds,
    count_repetitions_with_config,
)


def _make_cfg(**overrides):
    cfg = config.CountingConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


FAULTS_DISABLED = config.FaultConfig(low_thresh=None, high_thresh=None)


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

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=FAULTS_DISABLED)

    assert reps == 2
    assert debug.valley_indices == [3, 9]
    assert debug.prominences and len(debug.prominences) == 2


def test_count_repetitions_handles_missing_column() -> None:
    """Si falta la columna principal se debe devolver cero repeticiones y depuración vacía."""
    df = pd.DataFrame({"right_knee": [170, 160, 150]})
    cfg = _make_cfg(primary_angle="left_knee")

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=FAULTS_DISABLED)

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

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=FAULTS_DISABLED)

    assert reps == 1
    assert debug.valley_indices == [3]
    assert len(debug.prominences) == 1


def test_state_machine_counts_clean_sine() -> None:
    cfg = _make_cfg(primary_angle="left_knee")
    fps = 30.0
    t = np.linspace(0, 4 * np.pi, 120)
    angles = 150 + 30 * np.cos(t)
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})

    reps, debug = count_repetitions_with_config(df, cfg, fps=fps, faults_cfg=FAULTS_DISABLED)

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

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=FAULTS_DISABLED)

    assert reps == 1


def test_state_machine_ignores_long_gaps() -> None:
    t = np.linspace(0, 2 * np.pi, 90)
    signal = 150 + 20 * np.cos(t)
    signal[30:40] = np.nan  # mayor que el hueco permitido
    df = pd.DataFrame({"left_knee": signal, "pose_ok": 1.0})
    cfg = _make_cfg(primary_angle="left_knee")

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=FAULTS_DISABLED)

    assert reps == 0


def test_threshold_filter_skipped_when_faults_missing() -> None:
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
        min_prominence=1.0,
        min_distance_sec=0.1,
        refractory_sec=0.0,
    )

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=FAULTS_DISABLED)

    assert reps == 2
    assert debug.raw_count == 2


def test_threshold_filter_drops_reps_not_crossing_limits() -> None:
    angles = np.array([150, 90, 150, 90, 150, 110, 150], dtype=float)
    valley_indices = [1, 3, 5]
    prominences = [10.0, 10.0, 10.0]

    filtered, _, rejected, _ = _filter_reps_by_thresholds(
        angles, valley_indices, prominences, low_thresh=100.0, high_thresh=140.0
    )

    assert len(filtered) == 2
    assert rejected == 1


def test_threshold_filter_rejects_nan_heavy_rep() -> None:
    angles = np.array([np.nan, np.nan, np.nan, np.nan])
    filtered_indices, _, rejected, reasons = _filter_reps_by_thresholds(
        angles, [1], [1.0], low_thresh=90.0, high_thresh=150.0
    )

    assert filtered_indices == []
    assert rejected == 1
    assert any("insufficient finite" in reason for reason in reasons)


def test_threshold_filter_can_be_opted_out() -> None:
    angles = [170, 150, 120, 90, 120, 150, 170, 150, 110, 80, 110, 150, 170]
    df = pd.DataFrame({"left_knee": angles})
    cfg = _make_cfg(
        primary_angle="left_knee",
        min_prominence=1.0,
        min_distance_sec=0.1,
        refractory_sec=0.0,
        enforce_low_thresh=False,
        enforce_high_thresh=False,
    )
    strict_faults = config.FaultConfig(low_thresh=200.0, high_thresh=10.0)

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=strict_faults)

    assert reps == 2
    assert debug.raw_count == 2
    assert debug.reps_rejected_threshold == 0


def test_threshold_filter_applies_when_enabled() -> None:
    angles = [170, 150, 120, 90, 120, 150, 170, 150, 110, 80, 110, 150, 170]
    df = pd.DataFrame({"left_knee": angles})
    cfg = _make_cfg(
        primary_angle="left_knee",
        min_prominence=1.0,
        min_distance_sec=0.1,
        refractory_sec=0.0,
        enforce_low_thresh=True,
        enforce_high_thresh=True,
    )
    strict_faults = config.FaultConfig(low_thresh=90.0, high_thresh=200.0)

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=strict_faults)

    assert debug.raw_count == 2
    assert reps == 0
    assert debug.reps_rejected_threshold == 2


def test_threshold_filter_applies_both_thresholds_when_enabled() -> None:
    angles = np.array([190.0, 170.0, 190.0])
    valley_indices = [1]
    prominences = [10.0]

    filtered, _, rejected, reasons = _filter_reps_by_thresholds(
        angles, valley_indices, prominences, low_thresh=160.0, high_thresh=185.0
    )

    assert filtered == []
    assert rejected == 1
    assert any("above low_thresh" in reason for reason in reasons)


def test_threshold_filter_requires_high_threshold_to_be_reached() -> None:
    angles = np.array([150.0, 159.9, 150.0])

    filtered, _, rejected, reasons = _filter_reps_by_thresholds(
        angles, [1], [1.0], low_thresh=None, high_thresh=160.0
    )

    assert filtered == []
    assert rejected == 1
    assert any("below high_thresh" in reason for reason in reasons)


def test_threshold_filter_accepts_exact_high_threshold() -> None:
    angles = np.array([150.0, 160.0, 150.0])

    filtered, _, rejected, reasons = _filter_reps_by_thresholds(
        angles, [1], [1.0], low_thresh=None, high_thresh=160.0
    )

    assert filtered == [1]
    assert rejected == 0
    assert reasons == []


def test_threshold_validation_uses_reference_signal() -> None:
    angles = [
        150,
        150,
        60,
        150,
        150,
        150,
        60,
        150,
        150,
        150,
        60,
        150,
        150,
    ]
    df = pd.DataFrame({"left_knee": angles})
    cfg = _make_cfg(
        primary_angle="left_knee",
        min_prominence=5.0,
        min_distance_sec=0.1,
        refractory_sec=0.0,
        enforce_low_thresh=True,
    )
    faults = config.FaultConfig(low_thresh=80.0, high_thresh=None)

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert debug.raw_count == 2
    assert reps == 2
    assert debug.reps_rejected_threshold == 0


def test_threshold_edges_count_boundary_reps() -> None:
    angles = []
    for _ in range(5):
        angles.extend([150.0, 150.0, 90.0, 90.0, 150.0, 150.0])
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
        min_distance_sec=0.0,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert reps == 5


def test_threshold_edges_avoid_flat_signals() -> None:
    flat_high = pd.DataFrame({"left_knee": [150.0] * 60, "pose_ok": 1.0})
    flat_mid = pd.DataFrame({"left_knee": [120.0] * 60, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    reps_high, _ = count_repetitions_with_config(flat_high, cfg, fps=30.0, faults_cfg=faults)
    reps_mid, _ = count_repetitions_with_config(flat_mid, cfg, fps=30.0, faults_cfg=faults)

    assert reps_high == 0
    assert reps_mid == 0


def test_threshold_edges_ignore_high_wobbles_without_low() -> None:
    angles = [150.0, 160.0, 155.0, 165.0, 150.0, 170.0, 145.0, 150.0]
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert reps == 0


def test_threshold_edges_reconcile_after_high_wobble() -> None:
    angles = [150.0, 145.0, 150.0]
    for _ in range(5):
        angles.extend([150.0, 150.0, 90.0, 90.0, 150.0, 150.0])
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
        min_distance_sec=0.0,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert reps == 5


def test_threshold_edges_do_not_change_internal_reps() -> None:
    angles = [
        170.0,
        150.0,
        120.0,
        90.0,
        120.0,
        150.0,
        170.0,
        150.0,
        110.0,
        80.0,
        110.0,
        150.0,
        170.0,
    ]
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
        min_distance_sec=0.0,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert reps == 2


def test_threshold_edges_do_not_double_count_existing_edges() -> None:
    angles = [150.0, 90.0, 150.0, 90.0, 150.0, 90.0, 150.0]
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
        min_distance_sec=0.0,
        refractory_sec=0.0,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert reps == 3


def test_threshold_edges_noop_when_thresholds_not_enforced() -> None:
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
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=False,
        enforce_high_thresh=False,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    reps_faults, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)
    reps_none, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=FAULTS_DISABLED)

    assert reps_faults == reps_none


def test_threshold_reconciliation_repairs_clipped_candidates(monkeypatch) -> None:
    angles = []
    for _ in range(5):
        angles.extend([150.0, 150.0, 90.0, 90.0, 150.0, 150.0])
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
        min_distance_sec=0.0,
        refractory_sec=0.0,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    def _clipped_candidates(*_args, **_kwargs):
        candidates = []
        rep_index = 0
        for cycle in range(5):
            start = cycle * 6
            low = start + 2
            end = start + 4
            if cycle >= 3:
                start = low
                end = start + 1
            candidates.append(
                RepCandidate(
                    rep_index=rep_index,
                    start_frame=start,
                    turning_frame=low,
                    end_frame=end,
                    min_angle=90.0,
                    max_angle=150.0 if cycle < 3 else 90.0,
                )
            )
            rep_index += 1
        return candidates

    monkeypatch.setattr(
        "src.C_analysis.repetition_counter.detect_rep_candidates", _clipped_candidates
    )

    reps, _ = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert reps == 5


def test_threshold_reconciliation_noop_when_candidates_match(monkeypatch) -> None:
    angles = []
    for _ in range(5):
        angles.extend([150.0, 150.0, 90.0, 90.0, 150.0, 150.0])
    df = pd.DataFrame({"left_knee": angles, "pose_ok": 1.0})
    cfg = _make_cfg(
        primary_angle="left_knee",
        enforce_low_thresh=True,
        enforce_high_thresh=True,
        min_distance_sec=0.0,
        refractory_sec=0.0,
    )
    faults = config.FaultConfig(low_thresh=100.0, high_thresh=140.0)

    def _aligned_candidates(*_args, **_kwargs):
        candidates = []
        rep_index = 0
        for cycle in range(5):
            start = cycle * 6
            low = start + 2
            end = start + 4
            candidates.append(
                RepCandidate(
                    rep_index=rep_index,
                    start_frame=start,
                    turning_frame=low,
                    end_frame=end,
                    min_angle=90.0,
                    max_angle=150.0,
                )
            )
            rep_index += 1
        return candidates

    monkeypatch.setattr(
        "src.C_analysis.repetition_counter.detect_rep_candidates", _aligned_candidates
    )

    reps, debug = count_repetitions_with_config(df, cfg, fps=30.0, faults_cfg=faults)

    assert reps == 5
    assert [c["start_frame"] for c in debug.rep_candidates] == [i * 6 for i in range(5)]
    assert [c["end_frame"] for c in debug.rep_candidates] == [i * 6 + 4 for i in range(5)]
