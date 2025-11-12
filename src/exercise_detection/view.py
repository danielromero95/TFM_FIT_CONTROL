"""Camera view classification independent from the exercise label."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .constants import (
    ANKLE_FRONT_WIDTH_THRESHOLD,
    ANKLE_SIDE_WIDTH_MAX,
    ANKLE_WIDTH_STD_THRESHOLD,
    SIDE_WIDTH_MAX,
    VIEW_FRONT_ANKLE_STD_WEIGHT,
    VIEW_FRONT_ANKLE_WIDTH_WEIGHT,
    VIEW_FRONT_FALLBACK_YAW_DEG,
    VIEW_FRONT_WIDTH_WEIGHT,
    VIEW_FRONT_WIDTH_THRESHOLD,
    VIEW_FRONT_WIDTH_STD_WEIGHT,
    VIEW_FRONT_YAW_WEIGHT,
    VIEW_FRONT_Z_WEIGHT,
    VIEW_SCORE_MARGIN,
    VIEW_SCORE_MIN,
    VIEW_SCORE_PER_EVIDENCE_THRESHOLD,
    VIEW_SIDE_ANKLE_STD_WEIGHT,
    VIEW_SIDE_ANKLE_WIDTH_WEIGHT,
    VIEW_SIDE_FALLBACK_YAW_DEG,
    VIEW_SIDE_WIDTH_WEIGHT,
    VIEW_SIDE_WIDTH_STD_WEIGHT,
    VIEW_SIDE_YAW_WEIGHT,
    VIEW_SIDE_Z_WEIGHT,
    VIEW_STRONG_CONTRADICTION_YAW_DEG,
    VIEW_WIDTH_STD_THRESHOLD,
    YAW_FRONT_MAX_DEG,
    YAW_SIDE_MIN_DEG,
    Z_DELTA_FRONT_MAX,
)
from .stats import safe_nanmean, safe_nanmedian, safe_nanpercentile, safe_nanstd
from .types import ViewResult


def classify_view(
    shoulder_yaw: np.ndarray | None,
    shoulder_z_delta: np.ndarray | None,
    shoulder_width: np.ndarray | None,
    ankle_width: np.ndarray | None,
) -> ViewResult:
    """Return a stable front/side view label and associated scores."""

    yaw = _to_array(shoulder_yaw)
    z = _to_array(shoulder_z_delta)
    width = _to_array(shoulder_width)
    ankle = _to_array(ankle_width)

    stats = {
        "yaw_med": safe_nanmedian(yaw),
        "yaw_p75": safe_nanpercentile(yaw, 75.0),
        "z_med": safe_nanmedian(z),
        "width_mean": safe_nanmean(width),
        "width_std": safe_nanstd(width),
        "width_p10": safe_nanpercentile(width, 10.0),
        "ankle_mean": safe_nanmean(ankle),
        "ankle_std": safe_nanstd(ankle),
        "ankle_p10": safe_nanpercentile(ankle, 10.0),
    }

    scores = {"front": 0.0, "side": 0.0}
    votes = {"front": 0, "side": 0}

    def add_front(feature: str, score: float, vote: bool) -> None:
        if score > 0:
            scores["front"] += score
        if vote:
            votes["front"] += 1

    def add_side(feature: str, score: float, vote: bool) -> None:
        if score > 0:
            scores["side"] += score
        if vote:
            votes["side"] += 1

    yaw_med = stats["yaw_med"]
    yaw_p75 = stats["yaw_p75"]
    z_med = stats["z_med"]
    width_mean = stats["width_mean"]
    width_std = stats["width_std"]
    width_p10 = stats["width_p10"]
    ankle_mean = stats["ankle_mean"]
    ankle_std = stats["ankle_std"]
    ankle_p10 = stats["ankle_p10"]

    add_front(
        "yaw",
        VIEW_FRONT_YAW_WEIGHT * _score_below(yaw_med, YAW_FRONT_MAX_DEG),
        _margin_check(yaw_med, YAW_FRONT_MAX_DEG, below=True),
    )
    add_side(
        "yaw",
        VIEW_SIDE_YAW_WEIGHT * _score_above(yaw_p75, YAW_SIDE_MIN_DEG),
        _margin_check(yaw_p75, YAW_SIDE_MIN_DEG, below=False),
    )

    add_front(
        "z",
        VIEW_FRONT_Z_WEIGHT * _score_below(z_med, Z_DELTA_FRONT_MAX),
        _margin_check(z_med, Z_DELTA_FRONT_MAX, below=True),
    )
    add_side(
        "z",
        VIEW_SIDE_Z_WEIGHT * _score_above(z_med, Z_DELTA_FRONT_MAX),
        _margin_check(z_med, Z_DELTA_FRONT_MAX, below=False),
    )

    add_front(
        "width_mean",
        VIEW_FRONT_WIDTH_WEIGHT * _score_above(width_mean, VIEW_FRONT_WIDTH_THRESHOLD),
        _margin_check(width_mean, VIEW_FRONT_WIDTH_THRESHOLD, below=False),
    )
    add_side(
        "width_mean",
        VIEW_SIDE_WIDTH_WEIGHT * _score_below(width_mean, SIDE_WIDTH_MAX),
        _margin_check(width_mean, SIDE_WIDTH_MAX, below=True),
    )

    add_front(
        "width_std",
        VIEW_FRONT_WIDTH_STD_WEIGHT * _score_below(width_std, VIEW_WIDTH_STD_THRESHOLD),
        _margin_check(width_std, VIEW_WIDTH_STD_THRESHOLD, below=True),
    )
    add_side(
        "width_std",
        VIEW_SIDE_WIDTH_STD_WEIGHT * _score_above(width_std, VIEW_WIDTH_STD_THRESHOLD),
        _margin_check(width_std, VIEW_WIDTH_STD_THRESHOLD, below=False),
    )

    add_front(
        "ankle_mean",
        VIEW_FRONT_ANKLE_WIDTH_WEIGHT * _score_above(ankle_mean, ANKLE_FRONT_WIDTH_THRESHOLD),
        _margin_check(ankle_mean, ANKLE_FRONT_WIDTH_THRESHOLD, below=False),
    )
    add_side(
        "ankle_mean",
        VIEW_SIDE_ANKLE_WIDTH_WEIGHT * _score_below(ankle_mean, ANKLE_SIDE_WIDTH_MAX),
        _margin_check(ankle_mean, ANKLE_SIDE_WIDTH_MAX, below=True),
    )

    add_front(
        "ankle_std",
        VIEW_FRONT_ANKLE_STD_WEIGHT * _score_below(ankle_std, ANKLE_WIDTH_STD_THRESHOLD),
        _margin_check(ankle_std, ANKLE_WIDTH_STD_THRESHOLD, below=True),
    )
    add_side(
        "ankle_std",
        VIEW_SIDE_ANKLE_STD_WEIGHT * _score_above(ankle_std, ANKLE_WIDTH_STD_THRESHOLD),
        _margin_check(ankle_std, ANKLE_WIDTH_STD_THRESHOLD, below=False),
    )

    add_front(
        "ankle_p10",
        VIEW_FRONT_ANKLE_WIDTH_WEIGHT * _score_above(ankle_p10, ANKLE_FRONT_WIDTH_THRESHOLD),
        _margin_check(ankle_p10, ANKLE_FRONT_WIDTH_THRESHOLD, below=False),
    )
    add_side(
        "ankle_p10",
        VIEW_SIDE_ANKLE_WIDTH_WEIGHT * _score_below(ankle_p10, ANKLE_SIDE_WIDTH_MAX),
        _margin_check(ankle_p10, ANKLE_SIDE_WIDTH_MAX, below=True),
    )

    label = _decide_label(scores, votes)

    if label == "unknown":
        if np.isfinite(yaw_med) and yaw_med <= VIEW_FRONT_FALLBACK_YAW_DEG:
            label = "front"
        elif np.isfinite(yaw_p75) and yaw_p75 >= VIEW_SIDE_FALLBACK_YAW_DEG:
            label = "side"

    if label == "side" and np.isfinite(yaw_med) and yaw_med <= YAW_FRONT_MAX_DEG and np.isfinite(z_med) and z_med <= Z_DELTA_FRONT_MAX:
        label = "front"
    elif label == "front" and np.isfinite(yaw_p75) and yaw_p75 >= VIEW_STRONG_CONTRADICTION_YAW_DEG:
        label = "side"

    return ViewResult(label=label, scores=scores, votes=votes)


def _decide_label(scores: Mapping[str, float], votes: Mapping[str, int]) -> str:
    front_score = scores["front"]
    side_score = scores["side"]
    best_label = "front" if front_score >= side_score else "side"
    other_label = "side" if best_label == "front" else "front"

    best_score = scores[best_label]
    other_score = scores[other_label]
    if best_score < VIEW_SCORE_MIN:
        return "unknown"
    if best_score - other_score < VIEW_SCORE_MARGIN:
        return "unknown"
    if votes[best_label] <= votes[other_label] and votes[best_label] < 2:
        return "unknown"
    return best_label


def _score_below(value: float, threshold: float) -> float:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return 0.0
    margin = threshold - value
    if margin <= 0:
        return 0.0
    return margin / max(abs(threshold), 1e-6)


def _score_above(value: float, threshold: float) -> float:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return 0.0
    margin = value - threshold
    if margin <= 0:
        return 0.0
    return margin / max(abs(threshold), 1e-6)


def _margin_check(value: float, threshold: float, *, below: bool) -> bool:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return False
    if below:
        margin = threshold - value
    else:
        margin = value - threshold
    return margin > VIEW_SCORE_PER_EVIDENCE_THRESHOLD * abs(threshold)


def _to_array(series: np.ndarray | None) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    return np.asarray(series, dtype=float)

