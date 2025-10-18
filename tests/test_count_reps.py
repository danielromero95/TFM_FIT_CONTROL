"""Unit tests for the legacy repetition counter."""

import pytest

from src.D_modeling.count_reps import count_reps_from_angles


def test_count_reps_simple_two_reps():
    """Synthetic knee angle sequence that produces two full repetitions."""
    angle_sequence = [
        170,
        165,
        150,
        130,
        100,
        60,
        80,
        120,
        150,
        170,  # First repetition
        165,
        150,
        130,
        100,
        50,
        70,
        120,
        150,
        170,  # Second repetition
    ]
    reps = count_reps_from_angles(angle_sequence, low_thresh=90.0, high_thresh=160.0)
    assert reps == 2


def test_count_reps_no_cycles():
    """Sequences that never complete a low/high cycle should yield zero reps."""
    seq_never_down = [170, 165, 160, 155, 150, 145]
    assert count_reps_from_angles(seq_never_down, low_thresh=90.0, high_thresh=160.0) == 0

    seq_never_up = [170, 165, 150, 130, 100, 60, 80, 120, 150, 155, 158]
    assert count_reps_from_angles(seq_never_up, low_thresh=90.0, high_thresh=160.0) == 0


def test_count_reps_partial_cycles():
    """Only complete down-up cycles should be counted."""
    seq = [
        170,
        165,
        130,
        80,
        85,
        100,
        120,
        130,
        150,
        155,
        158,
        150,
        130,
        80,
        50,
        90,
        150,
        170,
    ]
    reps = count_reps_from_angles(seq, low_thresh=90.0, high_thresh=160.0)
    assert reps == 1


def test_count_reps_custom_thresholds():
    """Custom thresholds should also produce consistent repetitions."""
    seq = [
        150,
        145,
        140,
        120,
        95,
        105,
        130,
        145,
        150,
        145,
        140,
        120,
        90,
        110,
        130,
        145,
    ]
    reps = count_reps_from_angles(seq, low_thresh=100.0, high_thresh=140.0)
    assert reps == 2
