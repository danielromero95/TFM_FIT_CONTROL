"""Módulo enfocado en el procesamiento cuadro a cuadro de la pipeline de análisis.

Este archivo concentra la lógica necesaria para:
- Inferir la orientación correcta del sujeto antes de generar videos de depuración.
- Ejecutar la estimación de pose, la detección incremental del ejercicio y las vistas previas.
- Ajustar los *dataframes* de *landmarks* para reducir su huella de memoria.

Separar estas responsabilidades facilita reutilizar las piezas y aislar errores durante las
pruebas.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import pandas as pd

from src.A_preprocessing.frame_extraction.state import FrameInfo
from src.A_preprocessing.frame_extraction.utils import normalize_rotation_deg
from src.config.constants import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from src.config.settings import (
    DEFAULT_LANDMARK_MIN_VISIBILITY,
    DETECTION_SAMPLE_FPS as DEFAULT_DETECTION_SAMPLE_FPS,
    MODEL_COMPLEXITY,
)
from src.B_pose_estimation.estimators import (
    CroppedPoseEstimator,
    PoseEstimator,
    RoiPoseEstimator,
)
from src.D_visualization import OverlayStyle, draw_pose_on_frame
from src.exercise_detection.exercise_detector import (
    DetectionResult,
    IncrementalExerciseFeatureExtractor,
)
from src.core.types import ExerciseType, ViewType

logger = logging.getLogger(__name__)

# Prioriza mp4v para evitar dependencias de libopenh264 en Windows.
_DEBUG_VIDEO_CODECS = ("mp4v", "avc1", "XVID", "H264")


def compute_repeat(
    prev_ts: float,
    curr_ts: float,
    prev_source_idx: int,
    curr_source_idx: int,
    debug_fps: float,
) -> int:
    """Calcular cuántas veces repetir un *frame* para preservar el tiempo real.

    Se basa en los *timestamps* si son válidos; de lo contrario, usa el salto de
    índices de cuadros de origen como aproximación.
    """

    fps = float(debug_fps) if debug_fps and debug_fps > 0 else 30.0
    max_repeat = int(math.ceil(fps * 2.0))

    if np.isfinite(prev_ts) and np.isfinite(curr_ts):
        dt = float(curr_ts) - float(prev_ts)
        if dt <= 0:
            dt = max(1.0 / fps, 0.001)
        repeat = max(1, int(round(dt * fps)))
    else:
        gap = max(1, int(curr_source_idx) - int(prev_source_idx))
        repeat = int(gap)

    return int(min(max_repeat, repeat))


@dataclass
class StreamingPoseResult:
    """Objeto con los resultados del *streaming* de pose y detección."""

    df_landmarks: pd.DataFrame
    frames_processed: int
    detection: Optional[DetectionResult]
    debug_video_path: Optional[Path]
    processed_size: Optional[tuple[int, int]] = None


def downcast_landmarks_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convertir columnas numéricas a tipos más pequeños para ahorrar memoria."""

    if df is None or df.empty:
        return df

    for col in ("frame_idx", "analysis_frame_idx", "source_frame_idx"):
        if col in df.columns:
            # Usamos int32 porque basta para indexar cuadros sin desperdiciar memoria.
            df[col] = df[col].astype(np.int32, copy=False)

    float_cols = [c for c in df.columns if c.startswith(("x", "y", "z", "v", "crop_"))]
    if "time_s" in df.columns:
        float_cols.append("time_s")
    for column in float_cols:
        df[column] = df[column].astype(np.float32, copy=False)
    return df


def infer_upright_quadrant_from_sequence(
    sequence: np.ndarray,
    *,
    sample: int = 25,
    min_valid: int = 6,
    dominance: float = 1.15,
) -> int:
    """Determinar cuántos grados (en múltiplos de 90) rotar para enderezar al sujeto."""

    total = len(sequence) if sequence is not None else 0
    if total == 0:
        return 0

    idxs = np.linspace(0, total - 1, num=min(total, sample), dtype=int)

    angles: list[float] = []
    votes: list[int] = []

    for idx in idxs:
        frame = sequence[idx]
        if frame is None:
            continue
        try:
            hx = (float(frame[23]["x"]) + float(frame[24]["x"])) * 0.5
            hy = (float(frame[23]["y"]) + float(frame[24]["y"])) * 0.5
            sx = (float(frame[11]["x"]) + float(frame[12]["x"])) * 0.5
            sy = (float(frame[11]["y"]) + float(frame[12]["y"])) * 0.5
        except Exception:
            continue
        if not (
            math.isfinite(hx)
            and math.isfinite(hy)
            and math.isfinite(sx)
            and math.isfinite(sy)
        ):
            continue

        dx = sx - hx
        dy = sy - hy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue

        angle = math.degrees(math.atan2(dx, -dy)) % 360.0
        angles.append(angle)

        ax, ay = abs(dx), abs(dy)
        # Votamos la rotación necesaria para enderezar al sujeto según dominancia del eje.
        if ay >= ax * dominance:
            votes.append(0 if dy < 0 else 180)
        elif ax >= ay * dominance:
            votes.append(270 if dx > 0 else 90)

    if len(angles) < min_valid:
        return 0

    if votes:
        counts = {v: votes.count(v) for v in (0, 90, 180, 270)}
        winner, runner_up = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        if winner[1] >= max(runner_up[1] * dominance, min_valid // 2):
            return normalize_rotation_deg(winner[0])

    median_ang = float(np.median(np.asarray(angles, dtype=float)))
    quant = int(round(median_ang / 90.0) * 90) % 360
    return normalize_rotation_deg((360 - quant) % 360)


def stream_pose_and_detection(
    frames: Iterable[np.ndarray | FrameInfo],
    cfg: "config.Config",
    *,
    detection_enabled: bool,
    detection_source_fps: float,
    debug_video_path: Optional[Path],
    debug_video_fps: float,
    preview_callback: Optional[Callable[[np.ndarray, int, float], None]] = None,
    preview_fps: Optional[float] = None,
    run_id: str = "default",
) -> StreamingPoseResult:
    """Recorrer los cuadros ejecutando pose, detección y vistas previas según la configuración."""

    rows: list[dict[str, float]] = []
    frames_processed = 0
    detection_result: Optional[DetectionResult] = None
    detection_error = False

    detection_target_fps = getattr(cfg.video, "detection_sample_fps", None)
    if not detection_target_fps or detection_target_fps <= 0:
        detection_target_fps = float(DEFAULT_DETECTION_SAMPLE_FPS)
    detection_source_fps = float(detection_source_fps or 0.0)
    detection_ts_fps = detection_source_fps if detection_source_fps > 0 else detection_target_fps

    # Resolver min_visibility antes de crear el detector incremental.
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
    configured_preview_fps = (
        float(preview_fps)
        if preview_fps and preview_fps > 0
        else float(getattr(getattr(cfg, "debug", object()), "preview_fps", 0.0) or 0.0)
    )
    if configured_preview_fps <= 0:
        configured_preview_fps = 10.0

    detection_stride_base = detection_ts_fps
    if detection_stride_base <= 0:
        detection_stride_base = (
            detection_target_fps if detection_target_fps > 0 else configured_preview_fps
        )
    stride_preview = 1
    if preview_active:
        try:
            stride_preview = max(1, int(round(detection_stride_base / configured_preview_fps)))
        except Exception:
            stride_preview = 1

    debug_writer: Optional[cv2.VideoWriter] = None
    debug_style = OverlayStyle()
    debug_path: Optional[Path] = None
    debug_fps = next(
        (
            fps_value
            for fps_value in (
                float(debug_video_fps or 0.0),
                float(detection_source_fps or 0.0),
                float(detection_ts_fps or 0.0),
                30.0,
            )
            if fps_value > 0
        ),
        30.0,
    )

    estimator_cls = (
        RoiPoseEstimator
        if (getattr(cfg.pose, "use_crop", False) and getattr(cfg.pose, "use_roi_tracking", False))
        else (CroppedPoseEstimator if getattr(cfg.pose, "use_crop", False) else PoseEstimator)
    )

    processed_size: Optional[tuple[int, int]] = None

    try:
        with estimator_cls(
            run_id=run_id,
            static_image_mode=False,
            model_complexity=int(getattr(cfg.pose, "model_complexity", MODEL_COMPLEXITY)),
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as estimator:
            prev_source_ts: float | None = None
            prev_overlay: Optional[np.ndarray] = None
            prev_overlay_ts: float = float("nan")
            prev_overlay_idx: int = 0

            for analysis_idx, frame_info in enumerate(frames):
                frames_processed += 1
                if isinstance(frame_info, FrameInfo):
                    frame = frame_info.array
                    source_frame_idx = int(getattr(frame_info, "index", analysis_idx))
                    ts_sec = float(getattr(frame_info, "timestamp_sec", float("nan")))
                else:
                    frame = frame_info
                    source_frame_idx = int(analysis_idx)
                    ts_sec = float("nan")

                height, width = frame.shape[:2]

                if processed_size is None:
                    processed_size = (int(width), int(height))

                if not np.isfinite(ts_sec):
                    if detection_ts_fps > 0:
                        ts_sec = source_frame_idx / float(detection_ts_fps)
                    else:
                        ts_sec = analysis_idx / float(max(detection_ts_fps, 1.0))

                if prev_source_ts is not None and np.isfinite(prev_source_ts):
                    if not np.isfinite(ts_sec) or ts_sec <= prev_source_ts:
                        ts_sec = prev_source_ts + max(1.0 / debug_fps, 0.001)

                prev_source_ts = ts_sec if np.isfinite(ts_sec) else prev_source_ts

                ts_ms = float(ts_sec * 1000.0)
                emit_preview = preview_active and (analysis_idx % stride_preview == 0)
                overlay_points_needed = emit_preview or debug_video_path is not None

                result = estimator.estimate(frame)
                landmarks = result.landmarks
                crop_box = result.crop_box
                row: dict[str, float] = {
                    "analysis_frame_idx": int(analysis_idx),
                    "frame_idx": int(analysis_idx),
                    "source_frame_idx": int(source_frame_idx),
                    "time_s": float(ts_sec),
                    "source_time_s": float(ts_sec),
                }
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
                            # Ignoramos puntos con visibilidad baja para evitar ruido en el análisis.
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
                    try:
                        if getattr(detection_extractor, "_done", False) or not detection_extractor.wants_frame(
                            analysis_idx
                        ):
                            pass
                        else:

                            def _coerce(value: object) -> float:
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

                            detection_extractor.add_landmarks(
                                analysis_idx,
                                arr,
                                width,
                                height,
                                ts_ms,
                            )
                    except Exception:  # pragma: no cover - protección ante fallos inesperados
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
                        else:
                            debug_path = debug_video_path
                    if debug_writer is not None:
                        overlay_to_write = ensure_overlay_frame()
                        if prev_overlay is not None:
                            repeat = compute_repeat(
                                prev_overlay_ts,
                                float(ts_sec),
                                prev_overlay_idx,
                                int(source_frame_idx),
                                debug_fps,
                            )
                            for _ in range(repeat):
                                debug_writer.write(prev_overlay)
                        prev_overlay = overlay_to_write
                        prev_overlay_ts = float(ts_sec)
                        prev_overlay_idx = int(source_frame_idx)

                if emit_preview and preview_active and preview_callback is not None:
                    try:
                        frame_for_preview = ensure_overlay_frame()
                        preview_callback(frame_for_preview, int(analysis_idx), float(ts_ms))
                    except Exception:  # pragma: no cover - mejor esfuerzo desde la UI
                        preview_active = False
                        logger.exception("Preview callback failed; disabling previews for this run")
            if debug_writer is not None and prev_overlay is not None:
                debug_writer.write(prev_overlay)
    finally:
        if debug_writer is not None:
            debug_writer.release()

    if detection_extractor is not None:
        try:
            detection_result = detection_extractor.finalize()
        except Exception:  # pragma: no cover - ruta de respaldo
            logger.exception("Automatic exercise detection failed during finalization")
            detection_result = DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

    df_landmarks = downcast_landmarks_df(pd.DataFrame.from_records(rows))
    logger.debug(
        "landmarks df dtypes: %s | shape=%s",
        dict(df_landmarks.dtypes.apply(lambda x: str(x))),
        df_landmarks.shape,
    )
    return StreamingPoseResult(
        df_landmarks=df_landmarks,
        frames_processed=frames_processed,
        detection=detection_result,
        debug_video_path=debug_path,
        processed_size=processed_size,
    )


def _open_debug_writer(output_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """Abrir un `VideoWriter` probando múltiples códecs hasta encontrar uno válido."""

    size = (int(width), int(height))
    fps_value = max(float(fps), 1.0)
    last_error = None
    for code in _DEBUG_VIDEO_CODECS:
        try:
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*code), fps_value, size)
            if writer.isOpened():
                logger.info("Debug video writer opened with codec=%s fps=%.2f size=%s", code, fps_value, size)
                return writer
            writer.release()
            logger.warning("Debug writer attempt failed with codec=%s fps=%.2f size=%s", code, fps_value, size)
        except Exception as exc:
            last_error = exc
            logger.warning("Debug writer exception with codec=%s: %s", code, exc)
    raise RuntimeError(f"Could not open debug VideoWriter for path={output_path}") from last_error
