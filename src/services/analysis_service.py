"""Service layer for running the video analysis pipeline."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import pandas as pd

from src import config
from src.A_preprocessing.frame_extraction import extract_and_preprocess_frames
from src.A_preprocessing.video_metadata import get_video_rotation
from src.B_pose_estimation.processing import (
    calculate_metrics_from_sequence,
    extract_landmarks_from_frames,
    filter_and_interpolate_landmarks,
)
from src.D_modeling.count_reps import count_repetitions_with_config
from src.F_visualization.video_renderer import render_landmarks_on_video_hq
from src.detect.exercise_detector import detect_exercise
from src.pipeline import OutputPaths, Report, RunStats


logger = logging.getLogger(__name__)


def run_pipeline(
    video_path: str,
    cfg: config.Config,
    progress_callback: Optional[Callable[[int], None]] = None,
    *,
    prefetched_detection: Optional[Tuple[str, str, float]] = None,
) -> Report:
    """Execute the full analysis pipeline using ``cfg`` as the single source of truth."""

    def notify(progress: int, message: str) -> None:
        logger.info(message)
        if progress_callback:
            progress_callback(progress)

    config_sha1 = cfg.fingerprint()
    logger.info("CONFIG_SHA1=%s", config_sha1)

    output_paths = _prepare_output_paths(Path(video_path), cfg.output)
    config_path = output_paths.base_dir / "config_used.json"
    config_path.write_text(
        json.dumps(cfg.to_serializable_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    notify(5, "STAGE 1: Extracting and rotating frames...")
    fps_original, _frame_count, fps_warning, prefer_reader_fps = _probe_video_metadata(video_path)

    initial_sample_rate = _compute_sample_rate(fps_original, cfg) if fps_original > 0 else 1

    rotate = cfg.pose.rotate
    if rotate is None:
        rotate = get_video_rotation(video_path)

    frames, fps_from_reader = extract_and_preprocess_frames(
        video_path=video_path,
        rotate=rotate,
        sample_rate=initial_sample_rate,
    )
    if not frames:
        raise ValueError("No frames could be extracted from the video.")

    warnings: list[str] = []
    skip_reason: Optional[str] = None

    if (fps_original <= 0 or prefer_reader_fps) and fps_from_reader > 0:
        fps_original = float(fps_from_reader)
    fps_original = float(fps_original)

    if fps_original <= 0 and fps_from_reader <= 0:
        warnings.append(
            "Unable to determine a valid FPS from metadata or reader. Falling back to 1 FPS."
        )
        fps_original = 1.0

    sample_rate = initial_sample_rate
    if initial_sample_rate == 1 and cfg.video.target_fps and cfg.video.target_fps > 0:
        recomputed = _compute_sample_rate(fps_original, cfg)
        if recomputed > 1:
            frames = frames[::recomputed]
            sample_rate = recomputed

    fps_effective = fps_original / sample_rate if sample_rate > 0 else fps_original
    frames_processed = len(frames)

    detected_label = "unknown"
    detected_view = "unknown"
    detected_confidence = 0.0
    if prefetched_detection is not None:
        detected_label, detected_view, detected_confidence = prefetched_detection
    else:
        try:
            detected_label, detected_view, detected_confidence = detect_exercise(video_path)
        except Exception:  # pragma: no cover - defensive logging only
            logger.exception("Automatic exercise detection failed")

    if fps_warning:
        message = f"{fps_warning} FPS final utilizado: {fps_original:.2f}."
        logger.warning(message)
        warnings.append(message)

    if frames_processed < cfg.video.min_frames:
        skip_reason = (
            f"Solo se procesaron {frames_processed} fotogramas (< {cfg.video.min_frames}). "
            "Se omite el conteo de repeticiones."
        )
        warnings.append(skip_reason)
    if fps_effective < cfg.video.min_fps:
        message = (
            f"Effective FPS {fps_effective:.2f} below the minimum ({cfg.video.min_fps}). "
            "Se omite el conteo de repeticiones."
        )
        warnings.append(message)
        if skip_reason is None:
            skip_reason = message

    target_size = (cfg.pose.target_width, cfg.pose.target_height)
    processed_frames = [cv2.resize(frame, target_size) for frame in frames]

    notify(25, "STAGE 2: Estimating pose on frames...")
    df_raw_landmarks = extract_landmarks_from_frames(
        frames=processed_frames,
        use_crop=cfg.pose.use_crop,
        visibility_threshold=config.MIN_DETECTION_CONFIDENCE,
    )

    notify(50, "STAGE 3: Filtering and interpolating landmarks...")
    filtered_sequence, crop_boxes = filter_and_interpolate_landmarks(df_raw_landmarks)

    debug_video_path: Optional[Path] = None
    if cfg.debug.generate_debug_video:
        notify(65, "EXTRA STAGE: Rendering HQ debug video...")
        debug_video_path = output_paths.session_dir / f"{output_paths.session_dir.name}_debug_HQ.mp4"
        render_landmarks_on_video_hq(
            frames,
            filtered_sequence,
            crop_boxes,
            str(debug_video_path),
            fps_effective,
        )

    notify(75, "STAGE 4: Computing biomechanical metrics...")
    df_metrics = calculate_metrics_from_sequence(filtered_sequence, fps_effective)

    primary_angle = cfg.counting.primary_angle
    angle_range = 0.0
    if not df_metrics.empty and primary_angle in df_metrics.columns:
        series = df_metrics[primary_angle].dropna()
        if not series.empty:
            angle_range = float(series.max() - series.min())
        else:
            warnings.append(
                f"No valid values could be obtained for the primary angle '{primary_angle}'."
            )
            if skip_reason is None:
                skip_reason = "The primary angle column does not contain valid data."
    else:
        warnings.append(
            f"The primary angle column '{primary_angle}' is not present in the metrics."
        )
        if skip_reason is None:
            skip_reason = "The primary angle column was not found in the metrics."

    if angle_range < cfg.counting.min_angle_excursion_deg and skip_reason is None:
        skip_reason = (
            f"The range of motion ({angle_range:.2f}°) is below the required minimum "
            f"({cfg.counting.min_angle_excursion_deg}°)."
        )
        warnings.append(skip_reason)

    reps = 0
    if skip_reason is None:
        notify(90, "STAGE 5: Counting repetitions...")
        reps, debug_info = count_repetitions_with_config(df_metrics, cfg.counting, fps_effective)
        if reps == 0 and not debug_info.valley_indices:
            warnings.append("No repetitions were detected with the current parameters.")
    else:
        notify(90, "STAGE 5: Counting skipped due to quality constraints.")

    if cfg.debug.debug_mode:
        logger.info("DEBUG MODE: Saving intermediate data...")
        df_raw_landmarks.to_csv(output_paths.session_dir / "1_raw_landmarks.csv", index=False)
        df_metrics.to_csv(output_paths.session_dir / "2_metrics.csv", index=False)

    stats = RunStats(
        config_sha1=config_sha1,
        fps_original=float(fps_original),
        fps_effective=float(fps_effective),
        frames=frames_processed,
        exercise_selected=cfg.counting.exercise,
        exercise_detected=detected_label,
        view_detected=detected_view,
        detection_confidence=float(detected_confidence),
        primary_angle=primary_angle if primary_angle in df_metrics.columns else None,
        angle_range_deg=float(angle_range),
        min_prominence=float(cfg.counting.min_prominence),
        min_distance_sec=float(cfg.counting.min_distance_sec),
        refractory_sec=float(cfg.counting.refractory_sec),
        warnings=warnings,
        skip_reason=skip_reason,
        config_path=config_path,
    )

    notify(100, "PIPELINE COMPLETADO")
    return Report(
        repetitions=reps if skip_reason is None else 0,
        metrics=df_metrics,
        debug_video_path=debug_video_path,
        stats=stats,
        config_used=cfg,
    )


def run_full_pipeline_in_memory(
    video_path: str,
    settings: Dict[str, Any],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper maintained for legacy front-ends."""
    cfg = config.load_default()
    _apply_settings(cfg, settings)
    report = run_pipeline(video_path, cfg, progress_callback)
    return report.to_legacy_dict()


# --- Helpers ------------------------------------------------------------------


def _prepare_output_paths(video_path: Path, output_cfg: config.OutputConfig) -> OutputPaths:
    base_dir = Path(output_cfg.base_dir).expanduser().resolve()
    counts_dir = Path(output_cfg.counts_dir).expanduser()
    poses_dir = Path(output_cfg.poses_dir).expanduser()

    for path in {base_dir, counts_dir, poses_dir}:
        path.mkdir(parents=True, exist_ok=True)

    session_dir = base_dir / video_path.stem
    session_dir.mkdir(parents=True, exist_ok=True)

    return OutputPaths(base_dir=base_dir, counts_dir=counts_dir, poses_dir=poses_dir, session_dir=session_dir)


def _compute_sample_rate(fps: float, cfg: config.Config) -> int:
    if cfg.video.manual_sample_rate and cfg.video.manual_sample_rate > 0:
        return max(1, int(cfg.video.manual_sample_rate))
    if fps > 0 and cfg.video.target_fps and cfg.video.target_fps > 0:
        return max(1, int(round(fps / cfg.video.target_fps)))
    return 1


def _probe_video_metadata(video_path: str) -> tuple[float, int, Optional[str], bool]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"Could not open the video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fallback_warning: Optional[str] = None
    prefer_reader_fps = False

    if fps <= 1.0 or not math.isfinite(fps):
        duration_sec = _estimate_duration_seconds(capture, frame_count)
        if duration_sec > 0 and frame_count > 0:
            estimated_fps = frame_count / duration_sec
            fallback_warning = (
                "Invalid metadata FPS. Estimated from video duration "
                f"({estimated_fps:.2f} fps)."
            )
            fps = estimated_fps
        else:
            fallback_warning = (
                "Invalid metadata FPS and unreliable duration. Falling back to the reader-reported FPS."
            )
            fps = 0.0
            prefer_reader_fps = True

    capture.release()
    return float(fps), frame_count, fallback_warning, prefer_reader_fps


def _estimate_duration_seconds(capture: cv2.VideoCapture, frame_count: int) -> float:
    if frame_count <= 1:
        return 0.0
    try:
        last_frame_index = max(frame_count - 1, 0)
        capture.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        if capture.grab():
            duration_msec = capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            return float(duration_msec) / 1000.0 if duration_msec > 0 else 0.0
    except Exception:  # pragma: no cover - backend dependent fallbacks
        return 0.0
    return 0.0


def _apply_settings(cfg: config.Config, settings: Dict[str, Any]) -> None:
    """Bridge legacy dictionary settings into the structured ``Config`` object."""
    if "output_dir" in settings:
        base_dir = Path(settings["output_dir"]).expanduser()
        cfg.output.base_dir = base_dir
        cfg.output.counts_dir = base_dir / "counts"
        cfg.output.poses_dir = base_dir / "poses"
    if "sample_rate" in settings:
        cfg.video.manual_sample_rate = int(settings["sample_rate"])
    if "rotate" in settings:
        cfg.pose.rotate = settings["rotate"]
    if "target_width" in settings:
        cfg.pose.target_width = int(settings["target_width"])
    if "target_height" in settings:
        cfg.pose.target_height = int(settings["target_height"])
    if "use_crop" in settings:
        cfg.pose.use_crop = bool(settings["use_crop"])
    if "generate_debug_video" in settings:
        cfg.debug.generate_debug_video = bool(settings["generate_debug_video"])
    if "debug_mode" in settings:
        cfg.debug.debug_mode = bool(settings["debug_mode"])
    if "low_thresh" in settings:
        cfg.faults.low_thresh = float(settings["low_thresh"])
    if "high_thresh" in settings:
        cfg.faults.high_thresh = float(settings["high_thresh"])
    # ``fps_override`` is intentionally ignored – the pipeline relies on metadata FPS.
