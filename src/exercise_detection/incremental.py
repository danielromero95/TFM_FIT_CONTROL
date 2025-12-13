"""Extracción incremental de características para escenarios de streaming.

El objetivo es replicar la funcionalidad previa (muestreo adaptativo y
clasificación en caliente) manteniendo el código separado del flujo offline.
La clase principal se puede integrar con capturas en vivo o con pipelines de
landmarks precalculados.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from src.config.constants import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from src.config.settings import (
    DEFAULT_LANDMARK_MIN_VISIBILITY,
    MODEL_COMPLEXITY,
    build_pose_kwargs,
)
from src.core.types import ExerciseType, ViewType

from .classification import classify_features
from .constants import DEFAULT_SAMPLING_RATE, FEATURE_NAMES, MIN_VALID_FRAMES
from .extraction import _append_nan, _process_frame
from .features import build_features_from_landmark_array
from .types import DetectionResult, FeatureSeries, make_detection_result

logger = logging.getLogger(__name__)


class _FeatureBuffer:
    """Arreglo dinámico ligero empleado por el extractor incremental."""

    __slots__ = ("array", "size", "dtype")

    def __init__(self, initial_capacity: int = 512, *, dtype: Any = float) -> None:
        dtype_obj = np.dtype(dtype)
        self.array = np.empty(int(initial_capacity), dtype=dtype_obj)
        self.size = 0
        self.dtype = dtype_obj

    def append(self, value: float) -> None:
        if self.size >= self.array.size:
            self._grow()
        self.array[self.size] = value
        self.size += 1

    def to_array(self) -> np.ndarray:
        if self.size == 0:
            return np.empty(0, dtype=self.dtype)
        return np.asarray(self.array[: self.size], dtype=float)

    def _grow(self) -> None:
        new_capacity = max(self.array.size * 2, 1024)
        new_array = np.empty(new_capacity, dtype=self.dtype)
        new_array[: self.size] = self.array[: self.size]
        self.array = new_array


class IncrementalExerciseFeatureExtractor:
    """Submuestrea fotogramas en vivo y clasifica el ejercicio en una pasada."""

    def __init__(
        self,
        *,
        target_fps: float,
        source_fps: float,
        max_frames: int = 300,
        min_visibility: Optional[float] = None,
    ) -> None:
        """Configurar el extractor incremental para *landmarks* en streaming."""

        self.target_fps = max(float(target_fps or 0.0), 0.1)
        self.source_fps = float(source_fps or 0.0)
        self.max_frames = max(1, int(max_frames))
        self._stride = self._compute_stride()
        self._min_visibility = float(
            min_visibility if (min_visibility is not None) else DEFAULT_LANDMARK_MIN_VISIBILITY
        )
        self._pose = None
        self._pose_landmark = None
        self._feature_buffers: Dict[str, _FeatureBuffer] = {
            name: _FeatureBuffer() for name in FEATURE_NAMES
        }
        self._samples = 0
        self._valid_samples = 0
        self._frames_considered = 0
        self._frames_accepted = 0
        self._initialised = False
        self._done = False

    def wants_frame(self, frame_idx: int) -> bool:
        """Indicar si el fotograma cumple el stride y aún cabe en el buffer."""

        if self._done or self._samples >= self.max_frames:
            return False
        stride = self._stride if self._stride and self._stride > 0 else 1
        try:
            idx = int(frame_idx)
        except (TypeError, ValueError):
            return False
        stride_int = int(stride) if stride else 1
        if stride_int <= 0:
            stride_int = 1
        return (idx % stride_int) == 0

    def _compute_stride(self) -> int:
        if self.source_fps <= 0.0:
            approx_source = max(self.target_fps, DEFAULT_SAMPLING_RATE)
        else:
            approx_source = self.source_fps
        stride = int(round(approx_source / self.target_fps)) if self.target_fps > 0 else 1
        return max(1, stride)

    def _ensure_pose(self) -> None:
        if self._initialised:
            return
        try:
            from mediapipe.python.solutions import pose as mp_pose_module
        except ImportError as exc:  # pragma: no cover - guardia defensiva
            raise RuntimeError("MediaPipe is not available for exercise detection") from exc

        pose_kwargs = build_pose_kwargs(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._pose = mp_pose_module.Pose(**pose_kwargs)
        self._pose_landmark = mp_pose_module.PoseLandmark
        self._initialised = True

    def add_frame(self, frame_idx: int, frame: np.ndarray, ts_ms: float) -> None:
        """Evaluar un fotograma bruto y añadirlo al buffer si toca muestrear."""

        del ts_ms
        self._frames_considered += 1
        if self._samples >= self.max_frames:
            self._done = True
            return
        if not self.wants_frame(frame_idx):
            return

        self._ensure_pose()
        assert self._pose is not None and self._pose_landmark is not None

        feature_lists: Dict[str, _FeatureBuffer] = self._feature_buffers
        valid = _process_frame(
            frame,
            self._pose,
            self._pose_landmark,
            feature_lists,
            min_visibility=self._min_visibility,
        )
        self._samples += 1
        if valid:
            self._valid_samples += 1
            self._frames_accepted += 1

    def add_landmarks(
        self,
        frame_idx: int,
        landmarks: list[dict[str, float]] | "np.ndarray",
        width: int,
        height: int,
        ts_ms: float,
    ) -> None:
        """Ingerir *landmarks* precalculados respetando el stride configurado."""

        del width, height, ts_ms

        self._frames_considered += 1
        if self._samples >= self.max_frames:
            self._done = True
            return
        if not self.wants_frame(frame_idx):
            return

        feature_lists: Dict[str, _FeatureBuffer] = self._feature_buffers
        arr: "np.ndarray | None" = None
        if isinstance(landmarks, np.ndarray):
            a = np.asarray(landmarks, dtype=float)
            if a.ndim == 2 and a.shape[0] >= 33 and a.shape[1] >= 3:
                if a.shape[1] >= 4:
                    arr = np.array(a[:33, :4], dtype=float, copy=True)
                else:
                    coords = np.array(a[:33, :3], dtype=float, copy=True)
                    vis = np.full((33, 1), np.nan, dtype=float)
                    arr = np.concatenate([coords, vis], axis=1)

        if arr is None:
            def _coerce(value: Any, default: float = float("nan")) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return float(default)

            xs = np.empty(33, dtype=float)
            ys = np.empty(33, dtype=float)
            zs = np.empty(33, dtype=float)
            vs = np.empty(33, dtype=float)
            for i in range(33):
                entry = landmarks[i] if (0 <= i < len(landmarks)) and isinstance(landmarks[i], dict) else None
                xs[i] = _coerce(entry.get("x") if entry else float("nan"))
                ys[i] = _coerce(entry.get("y") if entry else float("nan"))
                zs[i] = _coerce(entry.get("z") if entry else float("nan"))
                vs[i] = _coerce(entry.get("visibility") if entry else 0.0, 0.0)
            arr = np.column_stack((xs, ys, zs, vs))

        vis = arr[:, 3]
        mask = ~(np.isfinite(vis) & (vis >= self._min_visibility))
        arr[:, :3][mask] = np.nan

        try:
            feature_values = build_features_from_landmark_array(arr)
        except ValueError:
            _append_nan(feature_lists)
            self._samples += 1
            return

        for name in FEATURE_NAMES:
            feature_lists[name].append(float(feature_values.get(name, float("nan"))))

        self._samples += 1
        has_finite = any(np.isfinite(feature_values.get(name, float("nan"))) for name in FEATURE_NAMES)
        if has_finite:
            self._valid_samples += 1
            self._frames_accepted += 1

    def finalize(self) -> DetectionResult:
        """Cerrar recursos y clasificar la serie agregada de características."""

        try:
            logger.info(
                "detector.incremental: muestras=%d aceptadas=%d sr=%.2f",
                self._samples,
                self._valid_samples,
                self._effective_sampling_rate(),
            )
            if self._samples == 0 or self._valid_samples < MIN_VALID_FRAMES:
                return DetectionResult(ExerciseType.UNKNOWN, ViewType.UNKNOWN, 0.0)

            data = {name: buf.to_array() for name, buf in self._feature_buffers.items()}
            sampling_rate = self._effective_sampling_rate()
            features = FeatureSeries(
                data=data,
                sampling_rate=float(sampling_rate),
                valid_frames=int(self._valid_samples),
                total_frames=int(self._samples),
            )
            label, view, confidence = classify_features(features)
            return make_detection_result(label, view, confidence)
        finally:
            if self._initialised and self._pose is not None:
                try:
                    self._pose.close()
                except Exception:  # pragma: no cover
                    logger.debug("No se pudo cerrar la instancia de MediaPipe Pose", exc_info=True)
            self._pose = None
            self._pose_landmark = None
            self._initialised = False

    def _effective_sampling_rate(self) -> float:
        if self.source_fps > 0 and self._stride > 0:
            return float(self.source_fps / self._stride)
        if self.target_fps > 0:
            return float(self.target_fps)
        return DEFAULT_SAMPLING_RATE


__all__ = ["IncrementalExerciseFeatureExtractor"]
