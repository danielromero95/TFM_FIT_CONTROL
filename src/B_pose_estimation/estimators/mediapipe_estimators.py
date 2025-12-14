"""Estimadores de pose basados en Mediapipe listos para usar."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.config import POSE_CONNECTIONS
from src.config.constants import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from src.config.settings import MODEL_COMPLEXITY

from ..constants import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER
from ..geometry import (
    bounding_box_from_landmarks,
    expand_and_clip_box,
    landmarks_from_proto,
    smooth_bounding_box,
)
from ..types import Landmark, PoseResult
from .base import PoseEstimatorBase
from .mediapipe_pool import PoseGraphPool


REQUIRED_RELIABILITY_LANDMARKS: Tuple[int, ...] = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)


def pose_reliable(
    landmarks: Optional[Sequence[Landmark]],
    *,
    visibility_threshold: float = 0.5,
    required_indices: Sequence[int] = REQUIRED_RELIABILITY_LANDMARKS,
) -> bool:
    """Comprueba que existan *landmarks* clave con visibilidad suficiente."""

    if not landmarks:
        return False
    if max(required_indices, default=-1) >= len(landmarks):
        return False

    for idx in required_indices:
        lm = landmarks[idx]
        visibility = float(getattr(lm, "visibility", np.nan))
        if not np.isfinite(visibility) or visibility < visibility_threshold:
            return False
    return True


def reliability_score(
    landmarks: Optional[Sequence[Landmark]],
    *,
    required_indices: Sequence[int] = REQUIRED_RELIABILITY_LANDMARKS,
) -> float:
    """Media de visibilidad para los marcadores requeridos."""

    if not landmarks or max(required_indices, default=-1) >= len(landmarks):
        return float("nan")

    visibilities = [
        float(getattr(landmarks[idx], "visibility", np.nan)) for idx in required_indices
    ]
    if not all(np.isfinite(v) for v in visibilities):
        return float("nan")
    return float(np.mean(visibilities))


class LandmarkSmoother:
    """Suaviza marcadores con una media exponencial simple por fotograma."""

    def __init__(self, alpha: float = 0.5, visibility_threshold: float = 0.5) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.visibility_threshold = float(visibility_threshold)
        self._state: Optional[List[Landmark]] = None

    def reset(self) -> None:
        self._state = None

    def __call__(self, landmarks: Optional[List[Landmark]]) -> Optional[List[Landmark]]:
        if not landmarks:
            self._state = None
            return landmarks

        if self._state is None or len(self._state) != len(landmarks):
            self._state = [Landmark(**lm.to_dict()) for lm in landmarks]
            return self._state

        smoothed: List[Landmark] = []
        for previous, current in zip(self._state, landmarks):
            visibility = float(current.visibility)
            if not np.isfinite(visibility) or visibility < self.visibility_threshold:
                blended = previous
            else:
                coords = (float(current.x), float(current.y), float(current.z))
                if not all(np.isfinite(value) for value in coords):
                    blended = previous
                else:
                    blended = Landmark(
                        x=self.alpha * coords[0] + (1.0 - self.alpha) * float(previous.x),
                        y=self.alpha * coords[1] + (1.0 - self.alpha) * float(previous.y),
                        z=self.alpha * coords[2] + (1.0 - self.alpha) * float(previous.z),
                        visibility=visibility,
                    )
            smoothed.append(blended)

        self._state = smoothed
        return smoothed


def _rescale_landmarks_from_crop(
    mp_landmarks: Iterable[object],
    crop_box: Sequence[int],
    image_width: int,
    image_height: int,
    landmark_pb2_module,
) -> tuple[list[Landmark], object]:
    """Reescala *landmarks* normalizados del recorte al fotograma completo.

    Mantiene las coordenadas normalizadas en ``[0, 1]`` para que los renderizadores
    de vídeo no sufran desplazamientos cuando el recorte o el *resize* cambian el
    sistema de referencia.  Devuelve tanto la lista de ``Landmark`` propia como
    una instancia de ``NormalizedLandmarkList`` lista para dibujarse con MediaPipe.
    """

    if len(crop_box) != 4:
        raise ValueError("crop_box must contain four integers (x1, y1, x2, y2)")

    x1, y1, x2, y2 = map(int, crop_box)
    scale_x = x2 - x1
    scale_y = y2 - y1

    landmarks_full: list[Landmark] = []
    mp_landmarks_full = []

    for lm in mp_landmarks:
        x_full = (float(lm.x) * scale_x + x1) / image_width if scale_x > 0 else 0.0
        y_full = (float(lm.y) * scale_y + y1) / image_height if scale_y > 0 else 0.0
        x_clipped = float(np.clip(x_full, 0.0, 1.0))
        y_clipped = float(np.clip(y_full, 0.0, 1.0))
        landmark = Landmark(
            x=x_clipped,
            y=y_clipped,
            z=float(getattr(lm, "z", 0.0)),
            visibility=float(getattr(lm, "visibility", 0.0)),
        )
        landmarks_full.append(landmark)
        mp_landmarks_full.append(
            landmark_pb2_module.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
            )
        )

    return landmarks_full, landmark_pb2_module.NormalizedLandmarkList(
        landmark=mp_landmarks_full
    )


def _process_with_recovery(
    pose_graph: object,
    rgb_image: np.ndarray,
    *,
    static_image_mode: bool,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    smooth_landmarks: bool | None,
    enable_segmentation: bool | None,
    enable_recovery_pass: bool,
    recovery_miss_threshold: int,
    consecutive_misses: int,
):
    """Ejecuta el grafo de pose con un segundo intento para casos difíciles.

    Cuando el rastreador pierde el esqueleto en vídeo, relanzamos una pasada en
    modo estático con el modelo más completo disponible. Esto mejora la
    recuperación tras oclusiones o cambios bruscos de cámara sin depender de
    heurísticas externas.
    """

    results = pose_graph.process(rgb_image)
    if (
        results.pose_landmarks
        or static_image_mode
        or not enable_recovery_pass
        or (consecutive_misses + 1) < recovery_miss_threshold
    ):
        return results

    fallback_pose, key = PoseGraphPool.acquire(
        static_image_mode=True,
        model_complexity=max(model_complexity, 2),
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        smooth_landmarks=smooth_landmarks,
        enable_segmentation=enable_segmentation,
    )
    try:
        return fallback_pose.process(rgb_image)
    finally:
        PoseGraphPool.release(fallback_pose, key)


class PoseEstimator(PoseEstimatorBase):
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = MODEL_COMPLEXITY,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        smooth_landmarks: bool | None = None,
        enable_segmentation: bool | None = None,
        reliability_min_visibility: float = 0.5,
        landmark_smoothing_alpha: float | None = None,
        enable_recovery_pass: bool = False,
        recovery_miss_threshold: int = 2,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.enable_recovery_pass = enable_recovery_pass
        self.recovery_miss_threshold = max(1, int(recovery_miss_threshold))
        self.reliability_min_visibility = reliability_min_visibility
        self._smoother = (
            LandmarkSmoother(landmark_smoothing_alpha, visibility_threshold=reliability_min_visibility)
            if landmark_smoothing_alpha is not None
            else None
        )
        self.pose = None
        self._key: Optional[Tuple[bool, int, float, float, bool, bool]] = None
        self._misses = 0
        PoseGraphPool._ensure_imports()
        self.mp_pose = PoseGraphPool.mp_pose
        self.mp_drawing = PoseGraphPool.mp_drawing

    def _ensure_pose(self) -> None:
        if self.pose is None:
            self.pose, self._key = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
            )

    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        self._ensure_pose()
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = _process_with_recovery(
            self.pose,
            rgb_image,
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            enable_recovery_pass=self.enable_recovery_pass,
            recovery_miss_threshold=self.recovery_miss_threshold,
            consecutive_misses=self._misses,
        )
        if results.pose_landmarks:
            self._misses = 0
        else:
            self._misses += 1
            if self._smoother is not None:
                self._smoother.reset()
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)
        raw_landmarks = landmarks_from_proto(results.pose_landmarks.landmark)
        pose_ok = pose_reliable(raw_landmarks, visibility_threshold=self.reliability_min_visibility)
        landmarks = (
            self._smoother(raw_landmarks) if self._smoother is not None else raw_landmarks
        )
        annotated_image = image_bgr.copy()
        self.mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, POSE_CONNECTIONS)
        return PoseResult(
            landmarks=landmarks,
            annotated_image=annotated_image,
            crop_box=None,
            pose_ok=pose_ok,
        )

    def close(self) -> None:
        if self.pose is not None and self._key is not None:
            PoseGraphPool.release(self.pose, self._key)
            self.pose, self._key = None, None


class CroppedPoseEstimator(PoseEstimatorBase):
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = MODEL_COMPLEXITY,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        crop_margin: float = 0.15,
        target_size: Tuple[int, int] = (256, 256),
        smooth_landmarks: bool | None = None,
        enable_segmentation: bool | None = None,
        reliability_min_visibility: float = 0.5,
        landmark_smoothing_alpha: float | None = None,
        enable_recovery_pass: bool = False,
        recovery_miss_threshold: int = 2,
    ) -> None:
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.enable_recovery_pass = enable_recovery_pass
        self.recovery_miss_threshold = max(1, int(recovery_miss_threshold))
        self.reliability_min_visibility = reliability_min_visibility
        self._smoother = (
            LandmarkSmoother(landmark_smoothing_alpha, visibility_threshold=reliability_min_visibility)
            if landmark_smoothing_alpha is not None
            else None
        )
        self.pose_full = None
        self.pose_crop = None
        self._key_full: Optional[Tuple[bool, int, float, float, bool, bool]] = None
        self._key_crop: Optional[Tuple[bool, int, float, float, bool, bool]] = None
        self.smooth_factor = 0.65
        self._smoothed_bbox: Optional[Tuple[float, float, float, float]] = None
        self._misses_full = 0
        self._misses_crop = 0
        PoseGraphPool._ensure_imports()
        self.mp_pose = PoseGraphPool.mp_pose
        self.mp_drawing = PoseGraphPool.mp_drawing
        try:
            from mediapipe.framework.formats import landmark_pb2
        except ImportError as exc:  # pragma: no cover - entorno sin MediaPipe
            raise RuntimeError("MediaPipe landmark protobufs are not available") from exc

        self.landmark_pb2 = landmark_pb2

    def _ensure_graphs(self) -> None:
        if self.pose_full is None:
            self.pose_full, self._key_full = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
            )
        if self.pose_crop is None:
            self.pose_crop, self._key_crop = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
            )

    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        self._ensure_graphs()
        height, width = image_bgr.shape[:2]
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results_full = _process_with_recovery(
            self.pose_full,
            rgb_image,
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            enable_recovery_pass=self.enable_recovery_pass,
            recovery_miss_threshold=self.recovery_miss_threshold,
            consecutive_misses=self._misses_full,
        )
        if not results_full.pose_landmarks:
            self._misses_full += 1
            self._smoothed_bbox = None
            if self._smoother is not None:
                self._smoother.reset()
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)

        self._misses_full = 0
        landmarks_full = landmarks_from_proto(results_full.pose_landmarks.landmark)
        bbox = bounding_box_from_landmarks(landmarks_full, width, height)
        if bbox is None:
            self._smoothed_bbox = None
            if self._smoother is not None:
                self._smoother.reset()
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)

        smoothed_bbox = smooth_bounding_box(
            self._smoothed_bbox,
            bbox,
            factor=self.smooth_factor,
            width=width,
            height=height,
        )
        self._smoothed_bbox = smoothed_bbox
        crop_box = expand_and_clip_box(smoothed_bbox, width, height, self.crop_margin)
        x1, y1, x2, y2 = crop_box
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            self._smoothed_bbox = None
            if self._smoother is not None:
                self._smoother.reset()
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)

        crop_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        results_crop = _process_with_recovery(
            self.pose_crop,
            crop_rgb,
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            enable_recovery_pass=self.enable_recovery_pass,
            recovery_miss_threshold=self.recovery_miss_threshold,
            consecutive_misses=self._misses_crop,
        )
        annotated_image = image_bgr.copy()

        if results_crop.pose_landmarks:
            self._misses_crop = 0
            landmarks_crop = landmarks_from_proto(results_crop.pose_landmarks.landmark)
            pose_ok = pose_reliable(landmarks_crop, visibility_threshold=self.reliability_min_visibility)
            landmarks_crop = (
                self._smoother(landmarks_crop) if self._smoother is not None else landmarks_crop
            )
            landmarks_full, landmark_list = _rescale_landmarks_from_crop(
                results_crop.pose_landmarks.landmark,
                crop_box,
                width,
                height,
                self.landmark_pb2,
            )
            self.mp_drawing.draw_landmarks(annotated_image, landmark_list, POSE_CONNECTIONS)

            # Mantenemos los *landmarks* normalizados al recorte para no alterar el
            # contrato de "extract_landmarks_from_frames", pero usamos la versión
            # reescalada al fotograma completo para la visualización.
            del landmarks_full
        else:
            if self._smoother is not None:
                self._smoother.reset()
            self._misses_crop += 1
            pose_ok = False
            landmarks_crop = None
        return PoseResult(
            landmarks=landmarks_crop,
            annotated_image=annotated_image,
            crop_box=crop_box,
            pose_ok=pose_ok,
        )

    def close(self) -> None:
        if self.pose_full is not None and self._key_full is not None:
            PoseGraphPool.release(self.pose_full, self._key_full)
            self.pose_full, self._key_full = None, None
        if self.pose_crop is not None and self._key_crop is not None:
            PoseGraphPool.release(self.pose_crop, self._key_crop)
            self.pose_crop, self._key_crop = None, None
        self._smoothed_bbox = None


class RoiPoseEstimator(PoseEstimatorBase):
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = MODEL_COMPLEXITY,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        crop_margin: float = 0.15,
        target_size: Tuple[int, int] = (256, 256),
        refresh_period: int = 10,
        max_misses: int = 2,
        smooth_landmarks: bool | None = None,
        enable_segmentation: bool | None = None,
        reliability_min_visibility: float = 0.5,
        landmark_smoothing_alpha: float | None = None,
        enable_recovery_pass: bool = False,
        recovery_miss_threshold: int = 2,
    ) -> None:
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.refresh_period = max(1, refresh_period)
        self.max_misses = max(1, max_misses)
        self.last_box: Optional[List[int]] = None
        self.misses = 0
        self.frame_idx = 0
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.enable_recovery_pass = enable_recovery_pass
        self.recovery_miss_threshold = max(1, int(recovery_miss_threshold))
        self.reliability_min_visibility = reliability_min_visibility
        self._smoother = (
            LandmarkSmoother(landmark_smoothing_alpha, visibility_threshold=reliability_min_visibility)
            if landmark_smoothing_alpha is not None
            else None
        )
        self.smooth_factor = 0.65
        self._smoothed_bbox: Optional[Tuple[float, float, float, float]] = None

        PoseGraphPool._ensure_imports()
        self.mp_pose = PoseGraphPool.mp_pose
        self.mp_drawing = PoseGraphPool.mp_drawing
        try:
            from mediapipe.framework.formats import landmark_pb2
        except ImportError as exc:  # pragma: no cover - entorno sin MediaPipe
            raise RuntimeError("MediaPipe landmark protobufs are not available") from exc

        self.landmark_pb2 = landmark_pb2
        self.pose_full = None
        self.pose_crop = None
        self._key_full: Optional[Tuple[bool, int, float, float, bool, bool]] = None
        self._key_crop: Optional[Tuple[bool, int, float, float, bool, bool]] = None
        self._recovery_misses_full = 0
        self._recovery_misses_crop = 0

    def _ensure_graphs(self) -> None:
        if self.pose_full is None:
            self.pose_full, self._key_full = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
            )
        if self.pose_crop is None:
            self.pose_crop, self._key_crop = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
            )

    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        self._ensure_graphs()
        height, width = image_bgr.shape[:2]
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        run_full = (
            self.last_box is None
            or (self.frame_idx % self.refresh_period == 0)
            or (self.misses >= self.max_misses)
        )

        if not run_full and self.last_box is not None:
            x1, y1, x2, y2 = self.last_box
            if x2 <= x1 or y2 <= y1:
                run_full = True

        if run_full:
            results_full = _process_with_recovery(
                self.pose_full,
                rgb_image,
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=self.enable_segmentation,
                enable_recovery_pass=self.enable_recovery_pass,
                recovery_miss_threshold=self.recovery_miss_threshold,
                consecutive_misses=self._recovery_misses_full,
            )
            if not results_full.pose_landmarks:
                self._recovery_misses_full += 1
                self.misses += 1
                self.last_box = None
                self._smoothed_bbox = None
                if self._smoother is not None:
                    self._smoother.reset()
                output = PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)
            else:
                self._recovery_misses_full = 0
                landmarks_raw = landmarks_from_proto(results_full.pose_landmarks.landmark)
                pose_ok = pose_reliable(landmarks_raw, visibility_threshold=self.reliability_min_visibility)
                landmarks = (
                    self._smoother(landmarks_raw) if self._smoother is not None else landmarks_raw
                )
                annotated_image = image_bgr.copy()
                self.mp_drawing.draw_landmarks(annotated_image, results_full.pose_landmarks, POSE_CONNECTIONS)
                bbox = bounding_box_from_landmarks(landmarks, width, height)
                if bbox is None:
                    self.last_box = None
                    self._smoothed_bbox = None
                else:
                    smoothed_bbox = smooth_bounding_box(
                        self._smoothed_bbox,
                        bbox,
                        factor=self.smooth_factor,
                        width=width,
                        height=height,
                    )
                    self._smoothed_bbox = smoothed_bbox
                    self.last_box = expand_and_clip_box(smoothed_bbox, width, height, self.crop_margin)
                self.misses = 0
                output = PoseResult(
                    landmarks=landmarks,
                    annotated_image=annotated_image,
                    crop_box=self.last_box,
                    pose_ok=pose_ok,
                )
        else:
            x1, y1, x2, y2 = self.last_box  # type: ignore[misc]
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                self.misses += 1
                self.last_box = None
                self._smoothed_bbox = None
                if self._smoother is not None:
                    self._smoother.reset()
                output = PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)
            else:
                crop_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                results_crop = _process_with_recovery(
                    self.pose_crop,
                    crop_rgb,
                    static_image_mode=self.static_image_mode,
                    model_complexity=self.model_complexity,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    smooth_landmarks=self.smooth_landmarks,
                    enable_segmentation=self.enable_segmentation,
                    enable_recovery_pass=self.enable_recovery_pass,
                    recovery_miss_threshold=self.recovery_miss_threshold,
                    consecutive_misses=self._recovery_misses_crop,
                )
                if not results_crop.pose_landmarks:
                    self._recovery_misses_crop += 1
                    self.misses += 1
                    if self.misses >= self.max_misses:
                        self.last_box = None
                        self._smoothed_bbox = None
                    if self._smoother is not None:
                        self._smoother.reset()
                    output = PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)
                else:
                    self._recovery_misses_crop = 0
                    landmarks_full, landmark_list = _rescale_landmarks_from_crop(
                        results_crop.pose_landmarks.landmark,
                        (x1, y1, x2, y2),
                        width,
                        height,
                        self.landmark_pb2,
                    )

                    pose_ok = pose_reliable(landmarks_full, visibility_threshold=self.reliability_min_visibility)
                    landmarks_full = (
                        self._smoother(landmarks_full) if self._smoother is not None else landmarks_full
                    )

                    annotated_image = image_bgr.copy()
                    self.mp_drawing.draw_landmarks(annotated_image, landmark_list, POSE_CONNECTIONS)
                    bbox = bounding_box_from_landmarks(landmarks_full, width, height)
                    if bbox is None:
                        self.last_box = None
                        self._smoothed_bbox = None
                    else:
                        smoothed_bbox = smooth_bounding_box(
                            self._smoothed_bbox,
                            bbox,
                            factor=self.smooth_factor,
                            width=width,
                            height=height,
                        )
                        self._smoothed_bbox = smoothed_bbox
                        self.last_box = expand_and_clip_box(smoothed_bbox, width, height, self.crop_margin)
                    self.misses = 0
                    output = PoseResult(
                        landmarks=landmarks_full,
                        annotated_image=annotated_image,
                        crop_box=self.last_box,
                        pose_ok=pose_ok,
                    )

        self.frame_idx += 1
        return output

    def close(self) -> None:
        if self.pose_full is not None and self._key_full is not None:
            PoseGraphPool.release(self.pose_full, self._key_full)
            self.pose_full, self._key_full = None, None
        if self.pose_crop is not None and self._key_crop is not None:
            PoseGraphPool.release(self.pose_crop, self._key_crop)
            self.pose_crop, self._key_crop = None, None
        self._smoothed_bbox = None


__all__ = [
    "PoseEstimator",
    "CroppedPoseEstimator",
    "RoiPoseEstimator",
    "LandmarkSmoother",
    "pose_reliable",
    "reliability_score",
]
