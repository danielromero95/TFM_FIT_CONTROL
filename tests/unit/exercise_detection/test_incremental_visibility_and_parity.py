"""Unit tests for incremental exercise feature extraction visibility handling."""

from __future__ import annotations

from math import pi, sin

from src.exercise_detection.exercise_detector import IncrementalExerciseFeatureExtractor


def _mk_landmark(x: float, y: float, z: float, vis: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z), "visibility": float(vis)}


def test_add_landmarks_respects_min_visibility() -> None:
    """Landmarks below the configured visibility threshold should be ignored."""

    lm_low_vis = [_mk_landmark(0.5, 0.5, 0.0, 0.6) for _ in range(33)]
    extractor = IncrementalExerciseFeatureExtractor(
        target_fps=10.0,
        source_fps=10.0,
        max_frames=10,
        min_visibility=0.9,
    )
    for frame_idx in range(10):
        extractor.add_landmarks(frame_idx, lm_low_vis, 640, 480, ts_ms=100 * frame_idx)
    result = extractor.finalize()

    assert result is not None
    assert extractor._samples > 0
    assert extractor._valid_samples == 0


def _squat_like_series(frame_count: int) -> list[list[dict[str, float]]]:
    base = [_mk_landmark(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    frames: list[list[dict[str, float]]] = []
    for index in range(frame_count):
        knee_value = 0.5 + 0.2 * sin(2 * pi * index / 20)
        ankle_value = 0.8 - 0.2 * sin(2 * pi * index / 20)
        frame_landmarks = [landmark.copy() for landmark in base]
        frame_landmarks[25]["y"] = float(knee_value)
        frame_landmarks[28]["y"] = float(ankle_value)
        frames.append(frame_landmarks)
    return frames


def test_sampling_and_valid_frames_incremental() -> None:
    """The incremental extractor should subsample and count valid frames."""

    frames = _squat_like_series(60)
    extractor = IncrementalExerciseFeatureExtractor(
        target_fps=5.0,
        source_fps=30.0,
        max_frames=60,
        min_visibility=0.5,
    )
    for frame_idx, frame_landmarks in enumerate(frames):
        extractor.add_landmarks(
            frame_idx,
            frame_landmarks,
            640,
            480,
            ts_ms=1000 * frame_idx / 30.0,
        )
    result = extractor.finalize()

    assert extractor._samples > 0
    assert extractor._valid_samples > 0
    assert result is not None
