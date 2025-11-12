"""Utilidades de extracción de fotogramas empleadas por el detector de ejercicios.

Este módulo delega la lectura de vídeo en la infraestructura de
``A_preprocessing.frame_extraction`` para no duplicar lógica de apertura,
rotación automática o muestreo temporal.  Aquí únicamente nos encargamos de
invocar MediaPipe y transformar los *landmarks* en las series numéricas que
consumen los clasificadores posteriores.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, MutableMapping

import cv2
import numpy as np

from src.A_preprocessing.frame_extraction import extract_frames_stream
from src.A_preprocessing.video_metadata import read_video_file_info
from src.config.settings import DEFAULT_LANDMARK_MIN_VISIBILITY

from .constants import DEFAULT_SAMPLING_RATE, FEATURE_NAMES
from .features import build_features_from_landmarks
from .types import FeatureSeries

logger = logging.getLogger(__name__)


def extract_features(video_path: str, max_frames: int = 300) -> FeatureSeries:
    """Extraer las series temporales de *landmarks* a partir de un vídeo local.

    La función calcula un *stride* aproximado para limitar la muestra a
    ``max_frames`` y deja que ``extract_frames_stream`` gestione aspectos como
    la rotación embebida en los metadatos.  Esto evita mantener dos códigos de
    lectura de vídeo divergentes dentro del repositorio.
    """

    info = read_video_file_info(video_path)
    fps = float(info.fps or 0.0)
    frame_count = int(info.frame_count or 0)
    rotation_deg = int(info.rotation or 0)

    max_frames = max(1, int(max_frames))
    if frame_count > 0:
        sample_count = min(frame_count, max_frames)
        stride = max(1, int(round(frame_count / sample_count)))
    else:
        sample_count = max_frames
        stride = 1

    feature_lists: Dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    total_processed = 0
    valid_frames = 0

    try:
        from mediapipe.python.solutions import pose as mp_pose_module
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MediaPipe is not available in the runtime environment") from exc

    pose_landmark = mp_pose_module.PoseLandmark
    pose_kwargs = dict(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_pose_module.Pose(**pose_kwargs) as pose:
        for frame_info in extract_frames_stream(
            video_path=video_path,
            sampling="index",
            every_n=stride,
            rotate=rotation_deg,
            max_frames=sample_count,
            prefetched_info=info,
        ):
            total_processed += 1
            valid = _process_frame(frame_info.array, pose, pose_landmark, feature_lists)
            if valid:
                valid_frames += 1

    while total_processed < sample_count:
        # Cuando el vídeo termina antes de la cuota prevista completamos con NaN
        # para conservar la longitud y no sesgar el filtrado estadístico.
        _append_nan(feature_lists)
        total_processed += 1

    data = {key: np.asarray(values, dtype=float) for key, values in feature_lists.items()}

    sampling_rate = _estimate_sampling_rate(fps, frame_count, total_processed)

    percent_valid = (valid_frames / total_processed * 100.0) if total_processed else 0.0
    logger.info(
        "Extracción para detección de ejercicio: frames=%d válidos=%d (%.1f%%) sample_rate=%.2f",
        total_processed,
        valid_frames,
        percent_valid,
        sampling_rate,
    )

    return FeatureSeries(
        data=data,
        sampling_rate=float(sampling_rate),
        valid_frames=int(valid_frames),
        total_frames=int(total_processed),
    )


def extract_features_from_frames(
    frames: Iterable[np.ndarray], *, fps: float, max_frames: int = 300
) -> FeatureSeries:
    """Generar las mismas características que ``extract_features`` pero desde un flujo.

    Este camino se utiliza cuando otra parte del sistema ya produjo los
    fotogramas (por ejemplo, durante la validación en vivo) y únicamente
    necesitamos aplicar MediaPipe y normalizar los resultados.
    """

    try:
        from mediapipe.python.solutions import pose as mp_pose_module
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MediaPipe is not available in the runtime environment") from exc

    pose_landmark = mp_pose_module.PoseLandmark
    pose_kwargs = dict(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    feature_lists: Dict[str, list[float]] = {name: [] for name in FEATURE_NAMES}

    total_processed = 0
    valid_frames = 0

    with mp_pose_module.Pose(**pose_kwargs) as pose:
        for frame in frames:
            if total_processed >= max_frames:
                break
            total_processed += 1
            valid = _process_frame(frame, pose, pose_landmark, feature_lists)
            if valid:
                valid_frames += 1

    data = {key: np.asarray(values, dtype=float) for key, values in feature_lists.items()}

    sampling_rate = float(fps) if (fps and fps > 0) else DEFAULT_SAMPLING_RATE

    logger.info(
        "Extracción (streaming) para detección de ejercicio: frames=%d válidos=%d (%.1f%%) sample_rate=%.2f",
        total_processed,
        valid_frames,
        (valid_frames / total_processed * 100.0) if total_processed else 0.0,
        sampling_rate,
    )

    return FeatureSeries(
        data=data,
        sampling_rate=sampling_rate,
        valid_frames=int(valid_frames),
        total_frames=int(total_processed),
    )


def _process_frame(
    frame: np.ndarray,
    pose: Any,
    pose_landmark: Any,
    feature_lists: MutableMapping[str, Any],
    *,
    min_visibility: float = DEFAULT_LANDMARK_MIN_VISIBILITY,
) -> bool:
    """Ejecutar MediaPipe sobre un fotograma y acumular las mediciones resultantes.

    El resultado booleano indica si al menos una métrica salió finita; esto nos
    ayuda a medir cuántos fotogramas fueron realmente útiles para las
    heurísticas.
    """

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        _append_nan(feature_lists)
        return False

    del pose_landmark

    mp_landmarks = results.pose_landmarks.landmark
    landmark_dicts: list[dict[str, float]] = []
    world_landmark_dicts: list[dict[str, float]] | None = None

    if getattr(results, "pose_world_landmarks", None):
        mp_world_landmarks = results.pose_world_landmarks.landmark
        world_landmark_dicts = []
    else:
        mp_world_landmarks = None

    def _coerce(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    for landmark in mp_landmarks:
        visibility = _coerce(getattr(landmark, "visibility", 0.0))
        x_val = _coerce(getattr(landmark, "x", float("nan")))
        y_val = _coerce(getattr(landmark, "y", float("nan")))
        z_val = _coerce(getattr(landmark, "z", float("nan")))

        if not np.isfinite(visibility) or visibility < min_visibility:
            x_val = float("nan")
            y_val = float("nan")
            z_val = float("nan")

        landmark_dicts.append({"x": x_val, "y": y_val, "z": z_val, "visibility": visibility})

    if world_landmark_dicts is not None and mp_world_landmarks is not None:
        for idx, landmark in enumerate(mp_world_landmarks):
            x_val = _coerce(getattr(landmark, "x", float("nan")))
            y_val = _coerce(getattr(landmark, "y", float("nan")))
            z_val = _coerce(getattr(landmark, "z", float("nan")))
            visibility = landmark_dicts[idx]["visibility"] if idx < len(landmark_dicts) else float("nan")
            if not np.isfinite(visibility) or visibility < min_visibility:
                x_val = float("nan")
                y_val = float("nan")
                z_val = float("nan")
            world_landmark_dicts.append({"x": x_val, "y": y_val, "z": z_val})

    feature_values = build_features_from_landmarks(landmark_dicts, world_landmarks=world_landmark_dicts)

    has_finite = False
    for name in FEATURE_NAMES:
        value = float(feature_values.get(name, float("nan")))
        feature_lists[name].append(value)
        if not has_finite and np.isfinite(value):
            has_finite = True

    return has_finite


def _append_nan(feature_lists: MutableMapping[str, Any]) -> None:
    for key in FEATURE_NAMES:
        feature_lists[key].append(float("nan"))


def _estimate_sampling_rate(fps: float, frame_count: int, samples: int) -> float:
    if samples <= 0:
        return DEFAULT_SAMPLING_RATE
    if fps > 0 and frame_count > 0:
        duration = frame_count / fps
        if duration > 0:
            return float(samples / duration)
    if fps > 0:
        return float(fps)
    return DEFAULT_SAMPLING_RATE
