import pandas as pd

import pandas as pd

from src.C_analysis.repetition_counter import count_repetitions_with_config
from src.config.models import CountingConfig, FaultConfig


def test_deadlift_fallback_closes_incomplete_rep():
    # Primary angle loses the return-to-bottom, but trunk metric retains it.
    df = pd.DataFrame(
        {
            "frame_idx": list(range(7)),
            "right_hip": [5.0, 5.0, 20.0, 30.0, 20.0, float("nan"), float("nan")],
            "trunk_inclination_deg": [60.0, 62.0, 40.0, 20.0, 35.0, 55.0, 65.0],
        }
    )

    counting_cfg = CountingConfig(
        exercise="deadlift",
        primary_angle="right_hip",
        enforce_low_thresh=False,
        enforce_high_thresh=False,
    )
    faults = FaultConfig(low_thresh=10.0, high_thresh=25.0)

    reps, debug = count_repetitions_with_config(df, counting_cfg, fps=30.0, faults_cfg=faults)

    assert reps == 1
    assert len(debug.rep_candidates) == 1
    candidate = debug.rep_candidates[0]
    assert candidate["end_frame"] is not None
    assert candidate["rejection_reason"] == "NONE"


def test_threshold_enforcement_optional():
    df = pd.DataFrame(
        {"frame_idx": list(range(6)), "angle": [30.0, 30.0, 12.0, 12.0, 30.0, 30.0]}
    )
    faults = FaultConfig(low_thresh=10.0, high_thresh=25.0)

    cfg_disabled = CountingConfig(
        exercise="squat",
        primary_angle="angle",
        enforce_low_thresh=False,
        enforce_high_thresh=False,
    )
    reps_free, debug_free = count_repetitions_with_config(df, cfg_disabled, fps=30.0, faults_cfg=faults)

    assert reps_free == 1
    assert debug_free.rep_candidates[0]["accepted"] is True

    cfg_enforced = CountingConfig(
        exercise="squat",
        primary_angle="angle",
        enforce_low_thresh=True,
        enforce_high_thresh=False,
    )
    reps_strict, debug_strict = count_repetitions_with_config(df, cfg_enforced, fps=30.0, faults_cfg=faults)

    assert reps_strict == 0
    assert debug_strict.rep_candidates[0]["rejection_reason"] == "LOW_THRESH"


def test_deadlift_respects_low_threshold_when_enforced():
    df = pd.DataFrame(
        {
            "frame_idx": list(range(6)),
            "right_hip": [35.0, 35.0, 55.0, 55.0, 35.0, 35.0],
        }
    )
    faults = FaultConfig(low_thresh=30.0, high_thresh=50.0)

    cfg_enforced = CountingConfig(
        exercise="deadlift",
        primary_angle="right_hip",
        enforce_low_thresh=True,
        enforce_high_thresh=False,
    )
    reps_enforced, debug_enforced = count_repetitions_with_config(
        df, cfg_enforced, fps=30.0, faults_cfg=faults
    )

    assert reps_enforced == 0
    assert debug_enforced.rep_candidates[0]["rejection_reason"] == "LOW_THRESH"

    cfg_not_enforced = CountingConfig(
        exercise="deadlift",
        primary_angle="right_hip",
        enforce_low_thresh=False,
        enforce_high_thresh=False,
    )
    reps_free, debug_free = count_repetitions_with_config(
        df, cfg_not_enforced, fps=30.0, faults_cfg=faults
    )

    assert reps_free == 1
    assert debug_free.rep_candidates[0]["rejection_reason"] == "NONE"
