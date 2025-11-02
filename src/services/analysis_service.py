"""Service layer for running the video analysis pipeline."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from src import config
from src.config.constants import MIN_DETECTION_CONFIDENCE
from src.config.settings import (
    DEFAULT_LANDMARK_MIN_VISIBILITY,
    DETECTION_SAMPLE_FPS as DEFAULT_DETECTION_SAMPLE_FPS,
)
from src.A_preprocessing.frame_extraction import (
    extract_and_preprocess_frames,
    extract_frames_stream,
    extract_processed_frames_stream,
)
from src.A_preprocessing.video_metadata import VideoInfo, read_video_file_info
from src.B_pose_estimation.estimators import (
    CroppedPoseEstimator,
    PoseEstimator,
    RoiPoseEstimator,
)
from src.B_pose_estimation.processing import (
    calculate_metrics_from_sequence,
    filter_and_interpolate_landmarks,
)
from src.C_repetition_analysis.reps.api import count_repetitions_with_config
from src.D_visualization.video_landmarks import (
    OverlayStyle,
    draw_pose_on_frame,
    render_landmarks_video,
)
from src.exercise_detection.exercise_detector import (
    DetectionResult,
    IncrementalExerciseFeatureExtractor,
)
from src.core.types import ExerciseType, ViewType, as_exercise, as_view
from src.pipeline_data import OutputPaths, Report, RunStats
from src.services.errors import NoFramesExtracted, VideoOpenError


logger = logging.getLogger(__name__)


@dataclass
class _StreamingPoseResult:
    df_landmarks: "pd.DataFrame"
    frames_processed: int
    detection: Optional[DetectionResult]
    debug_video_path: Optional[Path]
    processed_size: Optional[tuple[int, int]] = None


# Prioriza mp4v para evitar dependencias de libopenh264 en Windows
_DEBUG_VIDEO_CODECS = ("mp4v", "avc1", "XVID", "H264")


class ProgressCallback(Protocol):
    def __call__(self, progress: int, message: str, /) -> None:
        """Preferred signature: receives percentage [0..100] and a human-readable message."""


@dataclass(frozen=True)
class SamplingPlan:
    """Encapsulate the sampling parameters derived for a pipeline run."""

    fps_base: float
    sample_rate: int
    fps_effective: float
    warnings: list[str]

def _notify(cb: Optional[Callable[..., None]], progress: int, message: str) -> None:
    """Invoke callbacks supporting either ``(progress, message)`` or legacy ``(progress)``."""
    if not cb:
        return
    try:
        # Preferred modern signature: (progress, message)
        cb(progress, message)
    except TypeError:
        # Backward-compatible legacy signature: (progress)
        cb(progress)


def _stream_pose_and_detection(
    frames: Iterable[np.ndarray],
    cfg: config.Config,
    *,
    detection_enabled: bool,
    detection_source_fps: float,
    debug_video_path: Optional[Path],
    debug_video_fps: float,
    preview_callback: Optional[Callable[[np.ndarray, int, float], None]] = None,
    preview_fps: Optional[float] = None,
) -> _StreamingPoseResult:
    """Iterate ``frames`` once running pose estimation, detection and optional previews."""

    rows: list[dict[str, float]] = []
    frames_processed = 0
    detection_result: Optional[DetectionResult] = None
    detection_error = False

    detection_target_fps = getattr(cfg.video, "detection_sample_fps", None)
    if not detection_target_fps or detection_target_fps <= 0:
        detection_target_fps = float(DEFAULT_DETECTION_SAMPLE_FPS)
    detection_source_fps = float(detection_source_fps or 0.0)
    detection_ts_fps = detection_source_fps if detection_source_fps > 0 else detection_target_fps

    # Resolve min_visibility before constructing the incremental detector so overrides are honored
    min_visibility = (
        float(getattr(cfg.debug, "min_visibility", DEFAULT_LANDMARK_MIN_VISIBILITY))
        if hasattr(cfg, "debug")
        else float(DEFAULT_LANDMARK_MIN_VISIBILITY)
    )
    detection_extractor: Optional[IncrementalExerciseFeatureExtractor] = None
    if detection_enabled:
        detection_extractor = IncrementalExerciseFeatureExtractor(
            target_fps=float(detection_target_fps),
            source_fps=detection_ts_fps,
            max_frames=300,
            min_visibility=min_visibility,
        )

    preview_active = bool(preview_callback)
    configured_preview_fps = float(preview_fps) if preview_fps and preview_fps > 0 else float(
        getattr(getattr(cfg, "debug", object()), "preview_fps", 0.0) or 0.0
    )
    if configured_preview_fps <= 0:
        configured_preview_fps = 10.0

    detection_stride_base = detection_ts_fps
    if detection_stride_base <= 0:
        detection_stride_base = detection_target_fps if detection_target_fps > 0 else configured_preview_fps
    stride_preview = 1
    if preview_active:
        try:
            stride_preview = max(1, int(round(detection_stride_base / configured_preview_fps)))
        except Exception:
            stride_preview = 1

    debug_writer: Optional[cv2.VideoWriter] = None
    debug_style = OverlayStyle()
    debug_path: Optional[Path] = None
    debug_fps = float(debug_video_fps or detection_target_fps)
    if debug_fps <= 0:
        debug_fps = float(detection_target_fps)

    estimator_cls = (
        RoiPoseEstimator
        if (getattr(cfg.pose, "use_crop", False) and getattr(cfg.pose, "use_roi_tracking", False))
        else (CroppedPoseEstimator if getattr(cfg.pose, "use_crop", False) else PoseEstimator)
    )

    processed_size: Optional[tuple[int, int]] = None

    try:
        with estimator_cls(min_detection_confidence=MIN_DETECTION_CONFIDENCE) as estimator:
            for frame_idx, frame in enumerate(frames):
                frames_processed += 1
                height, width = frame.shape[:2]

                if processed_size is None:
                    processed_size = (int(width), int(height))

                ts_ms = (
                    (frame_idx / detection_ts_fps) * 1000.0
                    if detection_ts_fps > 0
                    else float(frame_idx) * 1000.0
                )
                emit_preview = preview_active and (frame_idx % stride_preview == 0)
                overlay_points_needed = emit_preview or debug_video_path is not None

                landmarks, _annotated, crop_box = estimator.estimate(frame)
                row: dict[str, float] = {"frame_idx": int(frame_idx)}
                overlay_points: dict[int, tuple[int, int]] = {}

                if landmarks:
                    crop_values = crop_box if crop_box else [0, 0, width, height]
                    crop_x1, crop_y1, crop_x2, crop_y2 = map(float, crop_values)
                    crop_width = max(crop_x2 - crop_x1, 0.0)
                    crop_height = max(crop_y2 - crop_y1, 0.0)

                    for landmark_index, point in enumerate(landmarks):
                        visibility = float(point.get("visibility", 0.0))
                        x_value = float(point.get("x", np.nan))
                        y_value = float(point.get("y", np.nan))

                        if getattr(cfg.pose, "use_crop", False) and not getattr(
                            cfg.pose, "use_roi_tracking", False
                        ):
                            if crop_width > 0.0 and crop_height > 0.0:
                                x_value = (x_value * crop_width + crop_x1) / float(width)
                                y_value = (y_value * crop_height + crop_y1) / float(height)
                            else:
                                x_value = np.nan
                                y_value = np.nan

                        if visibility < min_visibility:
                            x_value = np.nan
                            y_value = np.nan

                        row[f"x{landmark_index}"] = x_value
                        row[f"y{landmark_index}"] = y_value
                        row[f"z{landmark_index}"] = float(point.get("z", np.nan))
                        row[f"v{landmark_index}"] = visibility

                        if overlay_points_needed and np.isfinite(x_value) and np.isfinite(y_value):
                            px = int(round(x_value * width))
                            py = int(round(y_value * height))
                            overlay_points[landmark_index] = (px, py)

                    row.update(
                        {
                            "crop_x1": crop_x1,
                            "crop_y1": crop_y1,
                            "crop_x2": crop_x2,
                            "crop_y2": crop_y2,
                        }
                    )
                else:
                    for landmark_index in range(33):
                        row[f"x{landmark_index}"] = np.nan
                        row[f"y{landmark_index}"] = np.nan
                        row[f"z{landmark_index}"] = np.nan
                        row[f"v{landmark_index}"] = np.nan
                    row.update({"crop_x1": np.nan, "crop_y1": np.nan, "crop_x2": np.nan, "crop_y2": np.nan})

                if detection_extractor is not None and not detection_error:
                    def _coerce(value: Any) -> float:
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return float("nan")

                    arr = np.empty((33, 4), dtype=float)
                    for landmark_index in range(33):
                        arr[landmark_index, 0] = _coerce(row[f"x{landmark_index}"])
                        arr[landmark_index, 1] = _coerce(row[f"y{landmark_index}"])
                        arr[landmark_index, 2] = _coerce(row[f"z{landmark_index}"])
                        arr[landmark_index, 3] = _coerce(row[f"v{landmark_index}"])
                    try:
                        detection_extractor.add_landmarks(
                            frame_idx,
                            arr,
                            width,
                            height,
                            ts_ms,
                        )
                    except Exception:  # pragma: no cover - best effort guard
                        logger.exception("Automatic exercise detection failed during streaming")
                        detection_error = True

                rows.append(row)

                overlay_frame: Optional[np.ndarray] = None

                def ensure_overlay_frame() -> np.ndarray:
                    nonlocal overlay_frame
                    if overlay_frame is None:
                        overlay_frame = frame.copy()
                        if overlay_points:
                            draw_pose_on_frame(overlay_frame, overlay_points, style=debug_style)
                    return overlay_frame

                if debug_video_path is not None:
                    if debug_writer is None:
                        try:
                            debug_writer = _open_debug_writer(debug_video_path, width, height, debug_fps)
                        except Exception as exc:
                            logger.warning("Debug video writer initialization failed: %s", exc)
                            debug_writer = None
                            debug_path = None
                            debug_video_path = None
                        else:
                            debug_path = debug_video_path
                    if debug_writer is not None:
                        debug_writer.write(ensure_overlay_frame())

                if emit_preview and preview_active and preview_callback is not None:
                    try:
                        frame_for_preview = ensure_overlay_frame()
                        preview_callback(frame_for_preview, int(frame_idx), float(ts_ms))
                    except Exception:  # pragma: no cover - UI level best effort
                        preview_active = False
                        logger.exception("Preview callback failed; disabling previews for this run")
    finally:
        if debug_writer is not None:
            debug_writer.release()

    if detection_extractor is not None:
        try:
            detection_result = detection_extractor.finalize()
        except Exception:  # pragma: no cover - detection fallback
            logger.exception("Automatic exercise detection failed during finalization")
            detection_result = DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)
    df_landmarks = pd.DataFrame.from_records(rows)
    return _StreamingPoseResult(
        df_landmarks=df_landmarks,
        frames_processed=frames_processed,
        detection=detection_result,
        debug_video_path=debug_path,
        processed_size=processed_size,
    )


def _open_debug_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """Open a VideoWriter trying multiple codecs until one succeeds.

    Prefer 'mp4v' to avoid libopenh264 version issues on some Windows builds.
    """
    size = (int(width), int(height))
    fps_value = max(float(fps), 1.0)
    last_error = None
    for code in _DEBUG_VIDEO_CODECS:
        try:
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*code), fps_value, size)
            if writer.isOpened():
                logger.info("Debug video writer opened with codec=%s fps=%.2f size=%s", code, fps_value, size)
                return writer
            # ensure resources are freed if the backend returned a closed writer
            writer.release()
            logger.warning("Debug writer attempt failed with codec=%s fps=%.2f size=%s", code, fps_value, size)
        except Exception as exc:
            last_error = exc
            logger.warning("Debug writer exception with codec=%s: %s", code, exc)
    raise RuntimeError(f"Could not open debug VideoWriter for path={output_path}") from last_error


def _iter_original_frames_for_overlay(
    video_path: Path,
    *,
    rotate: int,
    sample_rate: int,
    target_fps: Optional[float],
    max_frames: int,
):
    """Yield original-resolution frames using the same sampling strategy as the pipeline."""

    target = float(target_fps) if target_fps and target_fps > 0 else None
    sampling_mode = "time" if target is not None else "index"

    kwargs: dict[str, object] = {
        "video_path": str(video_path),
        "sampling": sampling_mode,
        "rotate": int(rotate),
        "resize_to": None,
    }
    if sampling_mode == "time":
        kwargs["target_fps"] = target
    else:
        kwargs["every_n"] = max(1, int(sample_rate))

    iterator = extract_frames_stream(**kwargs)
    produced = 0
    for finfo in iterator:
        yield finfo.array
        produced += 1
        if max_frames and produced >= max_frames:
            break


def _generate_overlay_video(
    video_path: Path,
    session_dir: Path,
    *,
    frame_sequence: np.ndarray,
    crop_boxes: Optional[np.ndarray],
    processed_size: Optional[tuple[int, int]],
    rotate: int,
    sample_rate: int,
    target_fps: Optional[float],
    fps_for_writer: float,
    rotate_from_metadata: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Optional[Path]:
    """Render a debug overlay video matching the original resolution."""

    if frame_sequence is None:
        return None

    total_frames = len(frame_sequence)
    if total_frames == 0:
        return None

    if not processed_size or processed_size[0] <= 0 or processed_size[1] <= 0:
        return None

    is_df = False
    try:
        import pandas as pd  # noqa: WPS433

        is_df = isinstance(frame_sequence, pd.DataFrame)
    except Exception:
        is_df = False

    if is_df:
        cols_x = [c for c in frame_sequence.columns if c.startswith("x")]
        num_landmarks = len(cols_x)
        normalized_sequence: list[list[dict[str, float]]] = []
        for _, row in frame_sequence.iterrows():
            frame_landmarks: list[dict[str, float]] = []
            for i in range(num_landmarks):
                x_val = float(row.get(f"x{i}", float("nan")))
                y_val = float(row.get(f"y{i}", float("nan")))
                frame_landmarks.append({"x": x_val, "y": y_val})
            normalized_sequence.append(frame_landmarks)
        frame_sequence = normalized_sequence

    processed_w = int(processed_size[0])
    processed_h = int(processed_size[1])

    overlay_path = session_dir / f"{session_dir.name}_overlay.mp4"
    fps_value = fps_for_writer if fps_for_writer and fps_for_writer > 0 else 1.0

    frames_iter = _iter_original_frames_for_overlay(
        video_path,
        rotate=rotate,
        sample_rate=sample_rate,
        target_fps=target_fps,
        max_frames=total_frames,
    )

    output_rotate = 0
    if rotate_from_metadata and rotate:
        output_rotate = (360 - int(rotate)) % 360

    stats = render_landmarks_video(
        frames_iter,
        frame_sequence,
        crop_boxes,
        str(overlay_path),
        fps=float(fps_value),
        processed_size=(processed_w, processed_h),
        output_rotate=output_rotate,
        tighten_to_subject=False,  # keep full frame; cropping handled upstream when needed
        subject_margin=0.15,
        progress_cb=progress_cb,
    )

    if stats.frames_written <= 0:
        overlay_path.unlink(missing_ok=True)
        return None

    return overlay_path


def run_pipeline(
    video_path: str,
    cfg: config.Config,
    progress_callback: Optional[Callable[..., None]] = None,
    *,
    prefetched_detection: Optional[Union[Tuple[str, str, float], DetectionResult]] = None,
    preview_callback: Optional[Callable[[np.ndarray, int, float], None]] = None,
    preview_fps: Optional[float] = None,
) -> Report:
    """Execute the full analysis pipeline using ``cfg`` as the single source of truth."""

    def notify(progress: int, message: str) -> None:
        logger.info(message)
        _notify(progress_callback, progress, message)

    config_sha1 = cfg.fingerprint()
    logger.info("CONFIG_SHA1=%s", config_sha1)

    output_paths = _prepare_output_paths(Path(video_path), cfg.output)
    config_path = output_paths.session_dir / "config_used.json"
    config_path.write_text(
        json.dumps(cfg.to_serializable_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    notify(5, "STAGE 1: Extracting and rotating frames...")

    cap = _open_video_cap(video_path)
    manual_rotate = cfg.pose.rotate
    rotate_from_metadata = manual_rotate is None
    processing_rotate = int(manual_rotate) if manual_rotate is not None else 0
    warnings: list[str] = []
    skip_reason: Optional[str] = None
    fps_from_reader = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_effective = 0.0
    sample_rate = 1
    frames_processed = 0
    df_raw_landmarks = None
    fps_original = 0.0
    t1 = t0

    detected_label = ExerciseType.UNKNOWN
    detected_view = ViewType.UNKNOWN
    detected_confidence = 0.0
    debug_video_path_stream: Optional[Path] = None
    processed_frame_size: Optional[tuple[int, int]] = None

    target_size = (cfg.pose.target_width, cfg.pose.target_height)
    target_fps_for_sampling: Optional[float] = None

    try:
        info, fps_original, fps_warning, prefer_reader_fps = _read_info_and_initial_sampling(
            cap, video_path
        )

        initial_sample_rate = _compute_sample_rate(fps_original, cfg) if fps_original > 0 else 1

        metadata_rotation = int(info.rotation or 0)
        if manual_rotate is None:
            processing_rotate = metadata_rotation
        else:
            processing_rotate = int(manual_rotate)

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
        streaming_result = _stream_pose_and_detection(
            raw_iter,
            cfg,
            detection_enabled=prefetched_detection is None,
            detection_source_fps=detection_source,
            debug_video_path=(
                output_paths.session_dir / f"{output_paths.session_dir.name}_debug_HQ.mp4"
                if cfg.debug.generate_debug_video
                else None
            ),
            debug_video_fps=fps_effective if fps_effective > 0 else detection_source,
            preview_callback=preview_callback,
            preview_fps=preview_fps if preview_fps is not None else getattr(
                getattr(cfg, "debug", object()), "preview_fps", None
            ),
        )
        df_raw_landmarks = streaming_result.df_landmarks
        frames_processed = streaming_result.frames_processed
        debug_video_path_stream = streaming_result.debug_video_path
        processed_frame_size = streaming_result.processed_size
        if prefetched_detection is not None:
            detected_label, detected_view, detected_confidence = _normalize_detection(
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
            f"Effective FPS {fps_effective:.2f} below the minimum ({cfg.video.min_fps}). "
            "Skipping repetition counting."
        )
        warnings.append(message)
        if skip_reason is None:
            skip_reason = message

    t2 = time.perf_counter()
    notify(50, "STAGE 3: Filtering and interpolating landmarks...")
    filtered_sequence, crop_boxes = _filter_landmarks(df_raw_landmarks)

    debug_video_path: Optional[Path] = None
    overlay_video_path: Optional[Path] = None
    if cfg.debug.generate_debug_video:
        if debug_video_path_stream is not None:
            notify(65, "EXTRA STAGE: Debug video saved during streaming.")
            debug_video_path = debug_video_path_stream
        else:
            notify(65, "EXTRA STAGE: Debug video skipped (no frames recorded).")

        try:
            def _overlay_progress(written: int, total: int) -> None:
                # total puede ser 0 si el renderer no lo sabe todavía
                base = 65
                top = 75
                if total and total > 0:
                    frac = min(1.0, max(0.0, written / float(total)))
                    pct = int(base + (top - base) * frac)
                else:
                    pct = base
                notify(pct, f"EXTRA STAGE: Rendering debug video... ({written}/{total})")

            overlay_video_path = _generate_overlay_video(
                Path(video_path),
                output_paths.session_dir,
                frame_sequence=filtered_sequence,
                crop_boxes=crop_boxes,
                processed_size=processed_frame_size or target_size,
                rotate=processing_rotate,
                sample_rate=sample_rate,
                target_fps=target_fps_for_sampling,
                fps_for_writer=fps_effective if fps_effective > 0 else fps_original,
                rotate_from_metadata=rotate_from_metadata,
                progress_cb=_overlay_progress,
            )
            if overlay_video_path is not None:
                logger.info("Overlay video generated at %s", overlay_video_path)
        except Exception:
            logger.exception("Failed to render overlay video")

    t3 = time.perf_counter()
    notify(75, "STAGE 4: Computing biomechanical metrics...")
    (
        df_metrics,
        angle_range,
        metrics_warnings,
        metrics_skip_reason,
    ) = _compute_metrics_and_angle(filtered_sequence, cfg.counting.primary_angle, fps_effective)
    warnings.extend(metrics_warnings)
    if skip_reason is None and metrics_skip_reason is not None:
        skip_reason = metrics_skip_reason
    t_metrics_end = time.perf_counter()

    primary_angle = cfg.counting.primary_angle

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
        reps, count_warnings = _maybe_count_reps(
            df_metrics, cfg, fps_effective, skip_reason
        )
        warnings.extend(count_warnings)
    else:
        notify(90, "STAGE 5: Counting skipped due to quality constraints.")
        reps, count_warnings = _maybe_count_reps(df_metrics, cfg, fps_effective, skip_reason)
        warnings.extend(count_warnings)

    if cfg.debug.debug_mode:
        logger.info("DEBUG MODE: Saving intermediate data...")
        df_raw_landmarks.to_csv(output_paths.session_dir / "1_raw_landmarks.csv", index=False)
        df_metrics.to_csv(output_paths.session_dir / "2_metrics.csv", index=False)

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
        overlay_video_path=overlay_video_path,
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


def make_sampling_plan(
    *,
    fps_metadata: float,
    fps_from_reader: float,
    prefer_reader_fps: bool,
    initial_sample_rate: int,
    cfg: config.Config,
    fps_warning: Optional[str],
) -> SamplingPlan:
    """Compute ``fps_base``, ``sample_rate``, and warnings using legacy heuristics."""

    warnings: list[str] = []

    fps_base = float(fps_metadata)
    if (fps_base <= 0.0 or prefer_reader_fps) and fps_from_reader > 0.0:
        fps_base = float(fps_from_reader)
    if fps_base <= 0.0 and fps_from_reader <= 0.0:
        warnings.append(
            "Unable to determine a valid FPS from metadata or reader. Falling back to 1 FPS."
        )
        fps_base = 1.0

    sample_rate = int(initial_sample_rate)
    if initial_sample_rate == 1 and cfg.video.target_fps and cfg.video.target_fps > 0:
        recomputed = _compute_sample_rate(fps_base, cfg)
        if recomputed > 1:
            sample_rate = recomputed

    fps_effective = fps_base / sample_rate if sample_rate > 0 else fps_base

    if fps_warning:
        message = f"{fps_warning} Using FPS value: {fps_base:.2f}."
        logger.warning(message)
        warnings.append(message)

    return SamplingPlan(
        fps_base=float(fps_base),
        sample_rate=int(sample_rate),
        fps_effective=float(fps_effective),
        warnings=warnings,
    )


# Video metadata handled by: src/A_preprocessing/video_metadata.read_video_file_info


def _open_video_cap(video_path: str) -> cv2.VideoCapture:
    """Open a ``VideoCapture`` for ``video_path`` ensuring the handle is valid."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoOpenError(f"Could not open the video: {video_path}")
    return cap


def _read_info_and_initial_sampling(
    cap: cv2.VideoCapture,
    video_path: str,
) -> tuple[VideoInfo, float, Optional[str], bool]:
    """Read metadata and derive FPS heuristics used by the sampling plan."""

    info = read_video_file_info(video_path, cap=cap)

    fps_original = float(info.fps or 0.0)
    fps_warning: Optional[str] = None
    prefer_reader_fps = False

    if info.fps_source == "estimated":
        fps_warning = f"Invalid metadata FPS. Estimated from video duration ({fps_original:.2f} fps)."
    elif info.fps_source == "reader":
        fps_warning = (
            "Invalid metadata FPS and unreliable duration. Falling back to the reader-reported FPS."
        )
        prefer_reader_fps = True

    return info, fps_original, fps_warning, prefer_reader_fps


def _extract_frames(
    video_path: str,
    cap: cv2.VideoCapture,
    info: VideoInfo,
    rotate: int,
    initial_sample_rate: int,
) -> tuple[list[object], float]:
    """Extract and pre-process frames using the configured sample rate."""

    frames, fps_from_reader = extract_and_preprocess_frames(
        video_path=video_path,
        rotate=rotate,
        sample_rate=initial_sample_rate,
        cap=cap,
        prefetched_info=info,
    )
    return frames, float(fps_from_reader)


def _apply_sampling_plan(
    frames: list[object], initial_sample_rate: int, plan: SamplingPlan
) -> list[object]:
    """Apply the sampling stride derived from ``plan`` to the extracted frames."""

    if plan.sample_rate != initial_sample_rate:
        stride = max(1, int(plan.sample_rate // max(1, initial_sample_rate)))
        return frames[::stride]
    return frames


def _prepare_processed_frames(
    frames: list[object], target_size: tuple[int, int]
) -> list[object]:
    """Resize frames to ``target_size`` using area interpolation when downscaling."""

    processed: list[object] = []
    target_width, target_height = target_size
    for frame in frames:
        height, width = frame.shape[:2]
        if width == target_width and height == target_height:
            processed.append(frame)
            continue
        interpolation = cv2.INTER_AREA if width > target_width or height > target_height else cv2.INTER_LINEAR
        processed.append(cv2.resize(frame, target_size, interpolation=interpolation))
    return processed


def _filter_landmarks(df_raw: "pd.DataFrame") -> tuple["pd.DataFrame", object]:
    """Filter and interpolate the pose landmarks, returning crop boxes as metadata."""

    return filter_and_interpolate_landmarks(df_raw)


def _compute_metrics_and_angle(
    df_seq: "pd.DataFrame", primary_angle: str, fps_effective: float
) -> tuple["pd.DataFrame", float, list[str], Optional[str]]:
    """Compute biomechanical metrics and derive the primary angle excursion."""

    warnings: list[str] = []
    skip_reason: Optional[str] = None

    df_metrics = calculate_metrics_from_sequence(df_seq, fps_effective)
    angle_range = 0.0

    if not df_metrics.empty and primary_angle in df_metrics.columns:
        series = df_metrics[primary_angle].dropna()
        if not series.empty:
            angle_range = float(series.max() - series.min())
        else:
            warnings.append(
                f"No valid values could be obtained for the primary angle '{primary_angle}'."
            )
            skip_reason = "The primary angle column does not contain valid data."
    else:
        warnings.append(
            f"The primary angle column '{primary_angle}' is not present in the metrics."
        )
        skip_reason = "The primary angle column was not found in the metrics."

    return df_metrics, angle_range, warnings, skip_reason


def _maybe_count_reps(
    df_metrics: "pd.DataFrame",
    cfg: config.Config,
    fps_effective: float,
    skip_reason: Optional[str],
) -> tuple[int, list[str]]:
    """Count repetitions unless ``skip_reason`` indicates the stage was skipped."""

    if skip_reason is not None:
        return 0, []

    result, debug_info = count_repetitions_with_config(df_metrics, cfg.counting, fps_effective)
    stage_warnings: list[str] = []
    if result == 0 and not debug_info.valley_indices:
        stage_warnings.append("No repetitions were detected with the current parameters.")
    return result, stage_warnings

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
