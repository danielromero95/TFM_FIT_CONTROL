"""Tests de sincronización del video de depuración."""

from __future__ import annotations

from src.core.video_timing import RetimeWriter


def test_retime_writer_with_timestamps_vfr():
    retimer = RetimeWriter(30.0)
    retimer.prime(0.0, 0.0)
    repeat = retimer.repeats_for_next(0.10, 0.10)
    assert repeat == 3


def test_retime_writer_small_dt():
    retimer = RetimeWriter(30.0)
    retimer.prime(0.0, 0.0)
    repeat = retimer.repeats_for_next(0.033, 0.033)
    assert repeat == 1


def test_retime_writer_non_positive_dt():
    retimer = RetimeWriter(30.0)
    retimer.prime(1.0, 1.0)
    repeat = retimer.repeats_for_next(1.0, 1.0)
    assert repeat == 1


def test_retime_writer_with_frame_gap():
    retimer = RetimeWriter(30.0)
    retimer.prime(float("nan"), 0.0)
    repeat = retimer.repeats_for_next(float("nan"), 4 / 30.0)
    assert repeat == 4


def test_debug_duration_tracks_source_time():
    timestamps = [0.0, 0.08, 0.16, 0.35, 0.62]
    fps = 30.0

    total_frames = 0
    retimer = RetimeWriter(fps)
    retimer.prime(timestamps[0], timestamps[0])

    for ts in timestamps[1:]:
        total_frames += retimer.repeats_for_next(ts, ts)

    total_frames += 1  # Último cuadro escrito al menos una vez.

    duration_debug = total_frames / fps
    duration_source = timestamps[-1] - timestamps[0]

    assert abs(duration_debug - duration_source) <= 1.0 / fps
