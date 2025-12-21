"""Orquestador principal de la pipeline de análisis de vídeo."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from src import config
from src.A_preprocessing.frame_extraction import extract_processed_frames_stream
from src.A_preprocessing.frame_extraction.utils import normalize_rotation_deg
from src.core.types import ExerciseType, ViewType, as_exercise, as_view
from src.pipeline_data import OutputPaths, Report, RunStats
from .errors import NoFramesExtracted

from .config_bridge import apply_settings
from .metrics import (
    auto_counting_params,
    compute_metrics_and_angle,
    filter_landmarks,
    maybe_count_reps,
)
from .overlay import generate_overlay_video
from .sampling import (
    compute_sample_rate,
    make_sampling_plan,
    normalize_detection,
    open_video_cap,
    read_info_and_initial_sampling,
)
from .streaming import (
    infer_upright_quadrant_from_sequence,
    stream_pose_and_detection,
)

logger = logging.getLogger(__name__)


def _notify(cb: Optional[Callable[..., None]], progress: int, message: str) -> None:
    """Invocar *callbacks* compatibles con firmas antiguas y nuevas."""

    if not cb:
        return
    try:
        cb(progress, message)
    except TypeError:
        cb(progress)


def _prepare_output_paths(video_path: Path, output_cfg: config.OutputConfig) -> OutputPaths:
    """Crear las carpetas de salida necesarias para el análisis."""

    base_dir = Path(output_cfg.base_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M")
    session_name = f"{video_path.stem}-{timestamp}"

    session_dir = base_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    return OutputPaths(base_dir=base_dir, session_dir=session_dir)


def run_pipeline(
    video_path: str,
    cfg: config.Config,
    progress_callback: Optional[Callable[..., None]] = None,
    *,
    prefetched_detection: Optional[Union[Tuple[str, str, float], 'DetectionResult']] = None,
    preview_callback: Optional[Callable[[np.ndarray, int, float], None]] = None,
    preview_fps: Optional[float] = None,
) -> Report:
    """Ejecutar la pipeline completa utilizando `cfg` como fuente de verdad."""

    def notify(progress: int, message: str) -> None:
        logger.info(message)
        _notify(progress_callback, progress, message)

    config_sha1 = cfg.fingerprint()
    logger.info("CONFIG_SHA1=%s", config_sha1)

    output_paths = _prepare_output_paths(Path(video_path), cfg.output)
    config_path = output_paths.session_dir / "config_used.json"
    metrics_path = output_paths.session_dir / "metrics.csv"

    t0 = time.perf_counter()
    notify(5, "STAGE 1: Extracting and rotating frames...")

    cap = open_video_cap(video_path)
    manual_rotate = cfg.pose.rotate
    processing_rotate = normalize_rotation_deg(int(manual_rotate)) if manual_rotate is not None else 0
    warnings: list[str] = []
    debug_notes: list[str] = []
    skip_reason: Optional[str] = None
    counting_accuracy_warning: Optional[str] = None
    fps_from_reader = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_effective = 0.0
    sample_rate = 1
    frames_processed = 0
    df_raw_landmarks: Optional[pd.DataFrame] = None
    fps_original = 0.0
    t1 = t0

    detected_label = ExerciseType.UNKNOWN
    detected_view = ViewType.UNKNOWN
    detected_confidence = 0.0
    debug_video_path_stream: Optional[Path] = None
    processed_frame_size: Optional[tuple[int, int]] = None

    target_size = (cfg.pose.target_width, cfg.pose.target_height)
    target_fps_for_sampling: Optional[float] = None

    file_size_bytes = 0
    width = 0
    height = 0
    megapixels = 0.0
    is_heavy_by_size = False
    is_heavy_by_mp = False
    is_heavy_media = False

    try:
        info, fps_original, fps_warning, prefer_reader_fps = read_info_and_initial_sampling(cap, video_path)

        video_path_obj = Path(video_path)
        try:
            file_size_bytes = int(video_path_obj.stat().st_size)
        except Exception:
            file_size_bytes = 0

        width = int(info.width or 0)
        height = int(info.height or 0)
        megapixels = (width * height) / 1e6 if (width > 0 and height > 0) else 0.0

        is_heavy_by_size = (
            cfg.debug.overlay_disable_over_bytes > 0
            and file_size_bytes >= cfg.debug.overlay_disable_over_bytes
        )
        is_heavy_by_mp = (
            cfg.debug.preview_disable_over_mp > 0.0
            and megapixels >= cfg.debug.preview_disable_over_mp
        )
        is_heavy_media = bool(is_heavy_by_size or is_heavy_by_mp)

        logger.info(
            "HEAVY CHECK — size=%.1fMB, res=%dx%d (%.2fMP) -> heavy_size=%s heavy_mp=%s",
            file_size_bytes / (1024 * 1024) if file_size_bytes else 0.0,
            width,
            height,
            megapixels,
            is_heavy_by_size,
            is_heavy_by_mp,
        )

        initial_sample_rate = compute_sample_rate(fps_original, cfg) if fps_original > 0 else 1

        metadata_rotation = normalize_rotation_deg(int(info.rotation or 0))
        if manual_rotate is None:
            processing_rotate = metadata_rotation
        else:
            processing_rotate = normalize_rotation_deg(int(manual_rotate))

        plan = make_sampling_plan(
            fps_metadata=fps_original,
            fps_from_reader=fps_from_reader,
            prefer_reader_fps=prefer_reader_fps,
            initial_sample_rate=initial_sample_rate,
            cfg=cfg,
            fps_warning=fps_warning,
        )
        sample_rate = plan.sample_rate
        fps_effective = plan.fps_effective
        fps_original = plan.fps_base
        warnings.extend(plan.warnings)

        if cfg.video.target_fps and cfg.video.target_fps > 0:
            target_fps_for_sampling = float(cfg.video.target_fps)

        if target_fps_for_sampling and target_fps_for_sampling > 0:
            raw_iter = extract_processed_frames_stream(
                video_path=video_path,
                rotate=processing_rotate,
                resize_to=target_size,
                cap=cap,
                prefetched_info=info,
                progress_callback=None,
                target_fps=target_fps_for_sampling,
            )
            fps_effective = float(target_fps_for_sampling)
            if fps_original > 0:
                sample_rate = max(1, int(round(fps_original / fps_effective)))
            if fps_warning:
                logger.info(
                    "Using time-based sampling at %.3f FPS despite metadata warning: %s",
                    fps_effective,
                    fps_warning,
                )
        else:
            raw_iter = extract_processed_frames_stream(
                video_path=video_path,
                rotate=processing_rotate,
                resize_to=target_size,
                cap=cap,
                prefetched_info=info,
                progress_callback=None,
                every_n=sample_rate,
            )

        t1 = time.perf_counter()
        notify(25, "STAGE 2: Estimating pose on frames...")
        detection_source = fps_effective if fps_effective > 0 else fps_original
        if detection_source <= 0:
            detection_source = fps_from_reader
        preview_cb_to_use = preview_callback
        preview_fps_to_use = (
            preview_fps
            if preview_fps is not None
            else getattr(getattr(cfg, "debug", object()), "preview_fps", None)
        )

        if is_heavy_media:
            # Ajustamos las vistas previas para evitar saturar la UI con medios pesados.
            if (
                cfg.debug.preview_disable_over_mp > 0.0
                and megapixels >= cfg.debug.preview_disable_over_mp
            ):
                preview_cb_to_use = None
                logger.info("Preview desactivado por resolución alta (%.2f MP)", megapixels)
            else:
                if preview_fps_to_use is None or preview_fps_to_use <= 0:
                    preview_fps_to_use = cfg.debug.preview_fps_heavy
                else:
                    preview_fps_to_use = float(min(preview_fps_to_use, cfg.debug.preview_fps_heavy))
                logger.info(
                    "Preview limitado a %.1f FPS por tamaño de archivo",
                    preview_fps_to_use,
                )

        streaming_result = stream_pose_and_detection(
            raw_iter,
            cfg,
            detection_enabled=prefetched_detection is None,
            detection_source_fps=detection_source,
            debug_video_path=None,
            debug_video_fps=fps_effective if fps_effective > 0 else detection_source,
            preview_callback=preview_cb_to_use,
            preview_fps=(
                preview_fps_to_use
                if preview_fps_to_use is not None
                else getattr(getattr(cfg, "debug", object()), "preview_fps", None)
            ),
        )
        df_raw_landmarks = streaming_result.df_landmarks
        frames_processed = streaming_result.frames_processed
        debug_video_path_stream = streaming_result.debug_video_path
        processed_frame_size = streaming_result.processed_size
        if prefetched_detection is not None:
            detected_label, detected_view, detected_confidence = normalize_detection(
                prefetched_detection
            )
        else:
            detection_output = streaming_result.detection
            if detection_output is None:
                detected_label = ExerciseType.UNKNOWN
                detected_view = ViewType.UNKNOWN
                detected_confidence = 0.0
            else:
                detected_label = detection_output.label
                detected_view = detection_output.view
                detected_confidence = float(detection_output.confidence)
        if df_raw_landmarks.empty:
            raise NoFramesExtracted("No frames could be extracted from the video.")
    finally:
        cap.release()

    if df_raw_landmarks is None:
        raise NoFramesExtracted("No frames could be extracted from the video.")

    if frames_processed < cfg.video.min_frames:
        skip_reason = (
            "Only {frames_processed} frames processed (< {min_frames}). "
            "Skipping repetition counting."
        ).format(frames_processed=frames_processed, min_frames=cfg.video.min_frames)
        warnings.append(skip_reason)
    if fps_effective < cfg.video.min_fps:
        message = (
            f"Effective FPS {fps_effective:.2f} below the recommended minimum "
            f"({cfg.video.min_fps}). Repetition counting accuracy may be affected."
        )
        warnings.append(message)
        counting_accuracy_warning = message

    t2 = time.perf_counter()
    notify(50, "STAGE 3: Filtering and interpolating landmarks...")
    filtered_sequence, crop_boxes, quality_mask = filter_landmarks(df_raw_landmarks)
    overlay_rotate_cw = normalize_rotation_deg(
        infer_upright_quadrant_from_sequence(filtered_sequence)
    )
    logger.info(
        "Overlay rotation (clockwise) inferred from landmarks: %d°",
        overlay_rotate_cw,
    )

    debug_video_path: Optional[Path] = None
    overlay_video_path: Optional[Path] = None
    overlay_video_stream_path: Optional[Path] = None
    overlay_disabled = False
    overlay_scale_side: Optional[int] = None
    if cfg.debug.generate_debug_video:
        if debug_video_path_stream is not None:
            notify(65, "EXTRA STAGE: Debug video saved during streaming.")
            debug_video_path = debug_video_path_stream
        else:
            notify(65, "EXTRA STAGE: Debug video skipped (no frames recorded).")

        if is_heavy_by_size:
            warnings.append(
                f"Overlay desactivado por tamaño de archivo ({file_size_bytes / (1024*1024):.1f} MB)."
            )
            overlay_disabled = True
            logger.info(
                "Overlay desactivado por tamaño de archivo (%.1f MB)",
                file_size_bytes / (1024 * 1024) if file_size_bytes else 0.0,
            )
        elif is_heavy_by_mp and cfg.debug.overlay_max_long_side > 0:
            overlay_scale_side = int(cfg.debug.overlay_max_long_side)
            warnings.append(
                f"Overlay reescalado (lado máx. {overlay_scale_side}px) por alta resolución ({megapixels:.2f} MP)."
            )
            logger.info(
                "Overlay reescalado a lado máximo %d px por resolución alta (%.2f MP)",
                overlay_scale_side,
                megapixels,
            )

        if cfg.debug.generate_debug_video and not overlay_disabled:
            try:

                def _overlay_progress(written: int, total: int) -> None:
                    base = 65
                    top = 75
                    if total and total > 0:
                        frac = min(1.0, max(0.0, written / float(total)))
                        pct = int(base + (top - base) * frac)
                    else:
                        pct = base
                    notify(pct, f"EXTRA STAGE: Rendering debug video... ({written}/{total})")

                base_overlay_fps = fps_effective if fps_effective > 0 else fps_original
                overlay_cap_raw = getattr(cfg.debug, "overlay_fps_cap", 0.0)
                try:
                    overlay_cap_value = float(overlay_cap_raw)
                except (TypeError, ValueError):
                    overlay_cap_value = 0.0
                overlay_cap = overlay_cap_value if overlay_cap_value > 0 else 0.0
                fps_for_overlay = base_overlay_fps if base_overlay_fps > 0 else overlay_cap
                if overlay_cap > 0 and fps_for_overlay > 0:
                    fps_for_overlay = min(fps_for_overlay, overlay_cap)
                elif fps_for_overlay <= 0:
                    fps_for_overlay = overlay_cap

                overlay_result = generate_overlay_video(
                    Path(video_path),
                    output_paths.session_dir,
                    frame_sequence=filtered_sequence,
                    crop_boxes=crop_boxes,
                    processed_size=processed_frame_size or target_size,
                    rotate=processing_rotate,
                    sample_rate=sample_rate,
                    target_fps=target_fps_for_sampling,
                    fps_for_writer=fps_for_overlay,
                    output_rotate=overlay_rotate_cw,
                    progress_cb=_overlay_progress,
                    overlay_max_long_side=overlay_scale_side,
                )
                if overlay_result is not None:
                    overlay_video_path = overlay_result.raw_path
                    overlay_video_stream_path = overlay_result.stream_path
                    debug_video_path = overlay_video_path
                    logger.info(
                        "Overlay video generated at %s (web_safe=%s stream=%s)",
                        overlay_video_path,
                        overlay_result.web_safe_ok,
                        overlay_video_stream_path,
                    )
            except Exception:
                logger.exception("Failed to render overlay video")
    elif cfg.debug.generate_debug_video:
        logger.info("Overlay omitido por heurísticas de medio pesado.")

    t3 = time.perf_counter()
    notify(75, "STAGE 4: Computing biomechanical metrics...")
    (
        df_metrics,
        angle_range,
        metrics_warnings,
        metrics_skip_reason,
        chosen_primary,
    ) = compute_metrics_and_angle(
        filtered_sequence,
        cfg.counting.primary_angle,
        fps_effective,
        exercise=detected_label,
        view=detected_view,
        quality_mask=quality_mask,
    )
    warnings.extend(metrics_warnings)
    if skip_reason is None and metrics_skip_reason is not None:
        skip_reason = metrics_skip_reason
    t_metrics_end = time.perf_counter()

    primary_angle = chosen_primary or cfg.counting.primary_angle

    auto_params = auto_counting_params(
        detected_label,
        df_metrics,
        chosen_primary,
        fps_effective,
        cfg.counting,
        view=detected_view,
    )
    cfg.counting.min_prominence = float(auto_params.min_prominence)
    cfg.counting.min_distance_sec = float(auto_params.min_distance_sec)
    cfg.counting.refractory_sec = float(auto_params.refractory_sec)
    debug_notes.append(
        "Auto-tuned thresholds "
        f"→ prominence ≥ {auto_params.min_prominence:.1f}°, min distance = {auto_params.min_distance_sec:.2f}s, "
        f"refractory = {auto_params.refractory_sec:.2f}s."
    )
    logger.info(
        "COUNT AUTO-TUNE: exercise=%s primary=%s -> prominence=%.1f° distance=%.2fs refractory=%.2fs",
        getattr(as_exercise(detected_label), "value", str(detected_label)),
        primary_angle,
        auto_params.min_prominence,
        auto_params.min_distance_sec,
        auto_params.refractory_sec,
    )

    if cfg.debug.debug_mode:
        smoothing_info = df_metrics.attrs.get("smoothing", {})
        logger.info("DEBUG TELEMETRY: smoothing=%s", smoothing_info)
        logger.info(
            "DEBUG TELEMETRY: view=%s multipliers=%s cadence=%.2fs IQR=%.1f° → prominence=%.1f° distance=%.2fs refractory=%.2fs",
            getattr(as_view(detected_view), "value", str(detected_view)),
            auto_params.multipliers,
            auto_params.cadence_period_sec if auto_params.cadence_period_sec is not None else float("nan"),
            auto_params.iqr_deg if auto_params.iqr_deg is not None else float("nan"),
            auto_params.min_prominence,
            auto_params.min_distance_sec,
            auto_params.refractory_sec,
        )

    if angle_range < cfg.counting.min_angle_excursion_deg and skip_reason is None:
        skip_reason = (
            f"The range of motion ({angle_range:.2f}°) is below the required minimum "
            f"({cfg.counting.min_angle_excursion_deg}°)."
        )
        warnings.append(skip_reason)

    count_overrides = {
        "min_prominence": float(auto_params.min_prominence),
        "min_distance_sec": float(auto_params.min_distance_sec),
        "refractory_sec": float(auto_params.refractory_sec),
    }
    reps = 0
    if chosen_primary and chosen_primary != cfg.counting.primary_angle:
        cfg.counting.primary_angle = chosen_primary
    if skip_reason is None:
        t4 = time.perf_counter()
        notify(90, "STAGE 5: Counting repetitions...")
        reps, count_warnings = maybe_count_reps(
            df_metrics, cfg, fps_effective, skip_reason, overrides=count_overrides
        )
        warnings.extend(count_warnings)
    else:
        notify(90, "STAGE 5: Counting skipped due to quality constraints.")
        reps, count_warnings = maybe_count_reps(
            df_metrics, cfg, fps_effective, skip_reason, overrides=count_overrides
        )
        warnings.extend(count_warnings)

    if cfg.debug.debug_mode:
        logger.info("DEBUG MODE: Saving intermediate data...")
        df_raw_landmarks.to_csv(output_paths.session_dir / "1_raw_landmarks.csv", index=False)

    df_metrics.to_csv(metrics_path, index=False)

    effective_cfg_dict = cfg.to_serializable_dict()
    effective_cfg_dict.pop("output", None)
    config_path.write_text(
        json.dumps(effective_cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    notify(100, "PIPELINE COMPLETED")
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
        counting_accuracy_warning=counting_accuracy_warning,
        warnings=warnings,
        debug_notes=debug_notes,
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
        overlay_video_path=overlay_video_path,
        overlay_video_stream_path=overlay_video_stream_path,
        stats=stats,
        config_used=cfg,
        metrics_path=metrics_path,
        effective_config_path=config_path,
    )


def run_full_pipeline_in_memory(
    video_path: str,
    settings: Dict[str, Any],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Envoltura de compatibilidad pensada para interfaces antiguas."""

    cfg = config.load_default()
    apply_settings(cfg, settings)
    report = run_pipeline(video_path, cfg, progress_callback)
    return report.to_legacy_dict()
