"""Service layer for running the video analysis pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union
import time

import cv2

from src import config
from src.config.constants import MIN_DETECTION_CONFIDENCE
from src.A_preprocessing.frame_extraction import extract_and_preprocess_frames
from src.A_preprocessing.video_metadata import get_video_rotation, probe_video_metadata
from src.B_pose_estimation.processing import (
    calculate_metrics_from_sequence,
    extract_landmarks_from_frames,
    filter_and_interpolate_landmarks,
)
from src.D_modeling.count_reps import count_repetitions_with_config
from src.F_visualization.video_renderer import render_landmarks_on_video_hq
from src.exercise_detection.exercise_detector import DetectionResult, detect_exercise
from src.core.types import ExerciseType, ViewType, as_exercise, as_view
from src.pipeline import OutputPaths, Report, RunStats


logger = logging.getLogger(__name__)

def _notify(cb: Optional[Callable[..., None]], progress: int, message: str) -> None:
    """Call progress callback with (progress, message) if supported; fallback to (progress)."""
    if not cb:
        return
    try:
        # Nueva firma preferida: (progress, message)
        cb(progress, message)
    except TypeError:
        # Compatibilidad hacia atrás: (progress)
        cb(progress)


def run_pipeline(
    video_path: str,
    cfg: config.Config,
    progress_callback: Optional[Callable[..., None]] = None,
    *,
    prefetched_detection: Optional[Union[Tuple[str, str, float], DetectionResult]] = None,
) -> Report:
    """Execute the full analysis pipeline using ``cfg`` as the single source of truth."""

    def notify(progress: int, message: str) -> None:
        logger.info(message)
        _notify(progress_callback, progress, message)

    config_sha1 = cfg.fingerprint()
    logger.info("CONFIG_SHA1=%s", config_sha1)

    output_paths = _prepare_output_paths(Path(video_path), cfg.output)
    config_path = output_paths.base_dir / "config_used.json"
    config_path.write_text(
        json.dumps(cfg.to_serializable_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    notify(5, "STAGE 1: Extracting and rotating frames...")
    fps_original, _frame_count, fps_warning, prefer_reader_fps = probe_video_metadata(video_path)

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
    # Plan de muestreo unificado: decide fps_final, sample_rate y fps_effective.
    sample_rate, fps_effective, fps_original, _sampling_warnings = _plan_sampling(
        fps_metadata=fps_original,
        fps_from_reader=float(fps_from_reader),
        prefer_reader_fps=prefer_reader_fps,
        initial_sample_rate=initial_sample_rate,
        cfg=cfg,
        fps_warning=fps_warning,
    )
    warnings.extend(_sampling_warnings)
    # Si el plan decide mayor sample_rate que el usado en extracción inicial (solo ocurrirá si initial==1),
    # submuestreamos la lista de frames en memoria.
    if sample_rate != initial_sample_rate:
        stride = max(1, int(sample_rate // max(1, initial_sample_rate)))
        frames = frames[::stride]
    frames_processed = len(frames)

    detected_label = ExerciseType.UNKNOWN
    detected_view = ViewType.UNKNOWN
    detected_confidence = 0.0
    if prefetched_detection is not None:
        detected_label, detected_view, detected_confidence = _normalize_detection(prefetched_detection)
    else:
        try:
            detected_label, detected_view, detected_confidence = _normalize_detection(
                detect_exercise(video_path)
            )
        except Exception:  # pragma: no cover - defensive logging only
            logger.exception("Automatic exercise detection failed")

    # Mensaje de aviso ya incorporado en _plan_sampling (incluye FPS final).

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

    t1 = time.perf_counter()
    notify(25, "STAGE 2: Estimating pose on frames...")
    df_raw_landmarks = extract_landmarks_from_frames(
        frames=processed_frames,
        use_crop=cfg.pose.use_crop,
        visibility_threshold=MIN_DETECTION_CONFIDENCE,
    )

    t2 = time.perf_counter()
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

    t3 = time.perf_counter()
    notify(75, "STAGE 4: Computing biomechanical metrics...")
    df_metrics = calculate_metrics_from_sequence(filtered_sequence, fps_effective)
    t_metrics_end = time.perf_counter()

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
        t4 = time.perf_counter()
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

    notify(100, "PIPELINE COMPLETADO")
    t5 = time.perf_counter()
    extract_ms = (t1 - t0) * 1000
    pose_ms = (t2 - t1) * 1000
    filter_ms = (t3 - t2) * 1000
    metrics_ms = (t_metrics_end - t3) * 1000
    count_ms = (t5 - t4) * 1000 if "t4" in locals() else 0.0
    total_ms = (t5 - t0) * 1000

    stats = RunStats(
        config_sha1=config_sha1,
        fps_original=float(fps_original),
        fps_effective=float(fps_effective),
        frames=frames_processed,
        exercise_selected=as_exercise(cfg.counting.exercise),
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
        t_extract_ms=float(extract_ms),
        t_pose_ms=float(pose_ms),
        t_filter_ms=float(filter_ms),
        t_metrics_ms=float(metrics_ms),
        t_count_ms=float(count_ms),
        t_total_ms=float(total_ms),
    )

    logger.info(
        "TIMINGS extract=%.0fms pose=%.0fms filter=%.0fms metrics=%.0fms count=%.0fms total=%.0fms",
        extract_ms,
        pose_ms,
        filter_ms,
        metrics_ms,
        count_ms,
        total_ms,
    )
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


def _normalize_detection(
    detection: Union[DetectionResult, Tuple[str, str, float]]
) -> Tuple[ExerciseType, ViewType, float]:
    """Return ``(exercise, view, confidence)`` normalized to enums."""
    if isinstance(detection, DetectionResult):
        label, view, confidence = detection.label, detection.view, float(detection.confidence)
    else:
        label, view, confidence = detection
    return as_exercise(label), as_view(view), float(confidence)


def _plan_sampling(
    *,
    fps_metadata: float,
    fps_from_reader: float,
    prefer_reader_fps: bool,
    initial_sample_rate: int,
    cfg: config.Config,
    fps_warning: Optional[str],
) -> tuple[int, float, float, list[str]]:
    """
    Decide el FPS efectivo y el sample_rate final en un único lugar.

    Returns:
        sample_rate (int): stride final para muestrear frames.
        fps_effective (float): fps_metadata_final / sample_rate.
        fps_original_final (float): FPS base decidido (metadata/reader/fallback).
        warnings (list[str]): advertencias de calidad/estimación.
    """
    warnings: list[str] = []

    # 1) Elegir FPS base
    fps_base = float(fps_metadata)
    if (fps_base <= 0.0 or prefer_reader_fps) and fps_from_reader > 0.0:
        fps_base = float(fps_from_reader)
    if fps_base <= 0.0 and fps_from_reader <= 0.0:
        warnings.append(
            "Unable to determine a valid FPS from metadata or reader. Falling back to 1 FPS."
        )
        fps_base = 1.0

    # 2) Calcular sample_rate final manteniendo la semántica: si ya extrajimos con sample_rate>1,
    # no volvemos a submuestrear. Sólo si initial_sample_rate == 1 permitimos aumentar el stride.
    sample_rate = int(initial_sample_rate)
    if initial_sample_rate == 1 and cfg.video.target_fps and cfg.video.target_fps > 0:
        recomputed = _compute_sample_rate(fps_base, cfg)
        if recomputed > 1:
            sample_rate = recomputed

    fps_effective = fps_base / sample_rate if sample_rate > 0 else fps_base

    # 3) Avisos
    if fps_warning:
        message = f"{fps_warning} FPS final utilizado: {fps_base:.2f}."
        logger.warning(message)
        warnings.append(message)

    return sample_rate, float(fps_effective), float(fps_base), warnings


# (moved) probe_video_metadata and _estimate_duration_seconds now live in:
# src/A_preprocessing/video_metadata.py

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
