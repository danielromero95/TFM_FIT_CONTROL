"""Tests de sincronización del video de depuración."""

from __future__ import annotations

from src.C_analysis.streaming import compute_repeat


def test_compute_repeat_with_timestamps_vfr():
    repeat = compute_repeat(0.0, 0.10, 0, 1, 30.0)
    assert repeat == 3


def test_compute_repeat_small_dt():
    repeat = compute_repeat(0.0, 0.033, 0, 1, 30.0)
    assert repeat == 1


def test_compute_repeat_non_positive_dt():
    repeat = compute_repeat(1.0, 1.0, 0, 1, 30.0)
    assert repeat == 1


def test_compute_repeat_with_frame_gap():
    repeat = compute_repeat(float("nan"), float("nan"), 0, 4, 30.0)
    assert repeat == 4


def test_debug_duration_tracks_source_time():
    timestamps = [0.0, 0.08, 0.16, 0.35, 0.62]
    indices = list(range(len(timestamps)))
    fps = 30.0

    total_frames = 0
    prev_ts = timestamps[0]
    prev_idx = indices[0]

    for ts, idx in zip(timestamps[1:], indices[1:]):
        total_frames += compute_repeat(prev_ts, ts, prev_idx, idx, fps)
        prev_ts = ts
        prev_idx = idx

    total_frames += 1  # Último cuadro escrito al menos una vez.

    duration_debug = total_frames / fps
    duration_source = timestamps[-1] - timestamps[0]

    assert abs(duration_debug - duration_source) <= 1.0 / fps
