import numpy as np

from src.C_analysis.rep_candidates import RejectionReason, detect_rep_candidates


def _build_cycles(start_high: bool, cycles: int, low: float, high: float, partial_last: bool = False):
    sequence: list[float] = []
    for i in range(cycles):
        if start_high:
            sequence.extend([high, high, low, low, high, high])
        else:
            sequence.extend([low, low, high, high, low, low])
    if partial_last:
        if start_high:
            sequence.extend([high, low, low, high])
        else:
            sequence.extend([low, high, high])
    return np.asarray(sequence, dtype=float)


def test_squat_cycles_generate_down_up_phases():
    angles = _build_cycles(start_high=True, cycles=5, low=0.0, high=30.0)
    reps = detect_rep_candidates(angles, low_thresh=5.0, high_thresh=25.0, exercise_key="squat")
    assert len(reps) == 5
    assert all(rep.accepted for rep in reps)
    assert all(rep.down_start is not None and rep.down_end is not None for rep in reps)
    assert all(rep.up_start is not None and rep.up_end is not None for rep in reps)


def test_deadlift_cycles_start_with_up_phase():
    angles = _build_cycles(start_high=False, cycles=5, low=0.0, high=30.0)
    reps = detect_rep_candidates(angles, low_thresh=5.0, high_thresh=25.0, exercise_key="deadlift")
    assert len(reps) == 5
    assert all(rep.accepted for rep in reps)
    assert all(rep.up_start == rep.start_frame for rep in reps)
    assert all(rep.down_end == rep.end_frame for rep in reps)


def test_bench_cycles_match_squat_semantics():
    angles = _build_cycles(start_high=True, cycles=5, low=0.0, high=30.0)
    reps = detect_rep_candidates(angles, low_thresh=5.0, high_thresh=25.0, exercise_key="bench")
    assert len(reps) == 5
    assert all(rep.accepted for rep in reps)


def test_threshold_rejection_keeps_rep():
    lows = [8.0, 0.0, 0.0, 0.0, 0.0]
    angles = []
    for low in lows:
        angles.extend([30.0, 30.0, low, low, 30.0, 30.0])
    angles = np.asarray(angles, dtype=float)
    reps = detect_rep_candidates(angles, low_thresh=5.0, high_thresh=25.0, exercise_key="squat")
    assert len(reps) == 5
    accepted = [r for r in reps if r.accepted]
    rejected = [r for r in reps if not r.accepted]
    assert len(accepted) == 4
    assert len(rejected) == 1
    assert rejected[0].rejection_reason == RejectionReason.LOW_THRESH


def test_incomplete_rep_is_flagged():
    angles = _build_cycles(start_high=False, cycles=2, low=0.0, high=30.0, partial_last=True)
    reps = detect_rep_candidates(angles, low_thresh=5.0, high_thresh=25.0, exercise_key="deadlift")
    assert len(reps) == 3
    incomplete = [r for r in reps if r.rejection_reason == RejectionReason.INCOMPLETE]
    assert len(incomplete) == 1
    assert incomplete[0].end_frame is None


def test_squat_mid_segments_keep_phase_order():
    angles = np.asarray([30, 25, 20, 15, 10, 15, 20, 25, 30] * 2, dtype=float)
    reps = detect_rep_candidates(angles, low_thresh=8.0, high_thresh=28.0, exercise_key="squat")
    assert len(reps) == 2
    assert all(rep.down_start is not None and rep.up_start is not None for rep in reps)
    assert all(rep.down_start < rep.up_start for rep in reps)
