"""Estimadores de pose basados en Mediapipe listos para usar."""

from __future__ import annotations

from dataclasses import dataclass
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
from ..roi_state import RoiDebugRecorder, RoiState
from ..types import Landmark, PoseResult
from .base import PoseEstimatorBase
from .mediapipe_pool import PoseGraphPool


REQUIRED_RELIABILITY_LANDMARKS: Tuple[int, ...] = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)


@dataclass
class LetterboxTransform:
    target_width: int
    target_height: int
    scale: float
    pad_x: int
    pad_y: int


def _resize_with_letterbox(
    image: np.ndarray, target_size: Tuple[int, int]
) -> tuple[np.ndarray, LetterboxTransform | None]:
    target_width, target_height = target_size
    height, width = image.shape[:2]
    if width == target_width and height == target_height:
        return image, None

    scale = min(target_width / max(width, 1), target_height / max(height, 1))
    new_w = max(int(round(width * scale)), 1)
    new_h = max(int(round(height * scale)), 1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if resized.shape[:2] == image.shape[:2] and (new_w != width or new_h != height):
        # Fallback para stubs de cv2 que devuelven la entrada sin redimensionar
        ys = np.linspace(0, height - 1, new_h).astype(int)
        xs = np.linspace(0, width - 1, new_w).astype(int)
        resized = image[ys[:, None], xs]
    h_resized, w_resized = resized.shape[:2]

    x_offset = max((target_width - w_resized) // 2, 0)
    y_offset = max((target_height - h_resized) // 2, 0)

    canvas = np.zeros((target_height, target_width, *image.shape[2:]), dtype=image.dtype)
    x_start = max(x_offset, 0)
    y_start = max(y_offset, 0)
    x_end = min(x_start + w_resized, target_width)
    y_end = min(y_start + h_resized, target_height)

    src_x_end = x_end - x_start
    src_y_end = y_end - y_start
    canvas[y_start:y_end, x_start:x_end] = resized[:src_y_end, :src_x_end]

    transform = LetterboxTransform(
        target_width=target_width,
        target_height=target_height,
        scale=scale,
        pad_x=x_offset,
        pad_y=y_offset,
    )
    return canvas, transform


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
    letterbox_transform: LetterboxTransform | None = None,
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
        if letterbox_transform is None:
            x_crop = float(lm.x) * scale_x if scale_x > 0 else 0.0
            y_crop = float(lm.y) * scale_y if scale_y > 0 else 0.0
            x_crop_clipped = x_crop
            y_crop_clipped = y_crop
        else:
            x_lb = float(lm.x) * letterbox_transform.target_width
            y_lb = float(lm.y) * letterbox_transform.target_height
            x_crop = (x_lb - letterbox_transform.pad_x) / max(letterbox_transform.scale, 1e-6)
            y_crop = (y_lb - letterbox_transform.pad_y) / max(letterbox_transform.scale, 1e-6)

            x_crop_clipped = float(np.clip(x_crop, 0.0, scale_x)) if scale_x > 0 else 0.0
            y_crop_clipped = float(np.clip(y_crop, 0.0, scale_y)) if scale_y > 0 else 0.0

        x_full = (x_crop_clipped + x1) / image_width if image_width > 0 else 0.0
        y_full = (y_crop_clipped + y1) / image_height if image_height > 0 else 0.0
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
        debug_recorder: RoiDebugRecorder | None = None,
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
        self._debug_recorder = debug_recorder
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
        if self._debug_recorder:
            self._debug_recorder.finalize()


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
        debug_recorder: RoiDebugRecorder | None = None,
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
        self._frame_idx = 0
        self._debug_recorder = debug_recorder
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
        used_crop = False
        letterbox_transform: Optional[LetterboxTransform] = None
        frame_idx = self._frame_idx

        def _record_debug(pose_ok_value: bool, crop_box_value: Optional[Sequence[int]], *, roi_used: Optional[Sequence[int]]) -> None:
            if self._debug_recorder:
                self._debug_recorder.record(
                    {
                        "frame_idx": int(frame_idx),
                        "input_size": [int(width), int(height)],
                        "output_size": list(self.target_size),
                        "crop_used": bool(used_crop),
                        "roi": [int(v) for v in (roi_used or (0, 0, width, height))],
                        "pose_ok": bool(pose_ok_value),
                        "fail_streak": int(self._misses_crop),
                        "warmup_active": False,
                        "fallback_to_full": not used_crop,
                    }
                )
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
            _record_debug(False, None, roi_used=None)
            self._frame_idx += 1
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None, pose_ok=False)

        self._misses_full = 0
        landmarks_full = landmarks_from_proto(results_full.pose_landmarks.landmark)
        pose_ok_full = pose_reliable(landmarks_full, visibility_threshold=self.reliability_min_visibility)
        bbox = bounding_box_from_landmarks(landmarks_full, width, height)
        if bbox is None:
            self._smoothed_bbox = None
            if self._smoother is not None:
                self._smoother.reset()
            _record_debug(False, None, roi_used=None)
            self._frame_idx += 1
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
            pose_result = PoseResult(
                landmarks=landmarks_full,
                annotated_image=image_bgr,
                crop_box=(0, 0, width, height),
                pose_ok=pose_ok_full,
            )
            _record_debug(pose_ok_full, pose_result.crop_box, roi_used=crop_box)
            self._frame_idx += 1
            return pose_result

        crop_resized, letterbox_transform = _resize_with_letterbox(crop, self.target_size)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        used_crop = True
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
        pose_roi = crop_box

        if results_crop.pose_landmarks:
            self._misses_crop = 0
            landmarks_full_raw, landmark_list = _rescale_landmarks_from_crop(
                results_crop.pose_landmarks.landmark,
                crop_box,
                width,
                height,
                self.landmark_pb2,
                letterbox_transform=letterbox_transform,
            )
            pose_ok = pose_reliable(
                landmarks_full_raw, visibility_threshold=self.reliability_min_visibility
            )
            landmarks_full = (
                self._smoother(landmarks_full_raw)
                if self._smoother is not None
                else landmarks_full_raw
            )
            landmark_list_to_draw = landmark_list
            if self._smoother is not None:
                landmark_list_to_draw = self.landmark_pb2.NormalizedLandmarkList(
                    landmark=[
                        self.landmark_pb2.NormalizedLandmark(
                            x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
                        )
                        for lm in landmarks_full
                    ]
                )
            self.mp_drawing.draw_landmarks(
                annotated_image, landmark_list_to_draw, POSE_CONNECTIONS
            )
        else:
            if self._smoother is not None:
                self._smoother.reset()
            self._misses_crop += 1
            pose_ok = pose_ok_full
        pose_result = PoseResult(
            landmarks=landmarks_full,
            annotated_image=annotated_image,
            crop_box=(0, 0, width, height),
            pose_ok=pose_ok,
        )
        _record_debug(pose_ok, pose_result.crop_box, roi_used=pose_roi)
        self._frame_idx += 1
        return pose_result

    def close(self) -> None:
        if self.pose_full is not None and self._key_full is not None:
            PoseGraphPool.release(self.pose_full, self._key_full)
            self.pose_full, self._key_full = None, None
        if self.pose_crop is not None and self._key_crop is not None:
            PoseGraphPool.release(self.pose_crop, self._key_crop)
            self.pose_crop, self._key_crop = None, None
        self._smoothed_bbox = None
        if self._debug_recorder:
            self._debug_recorder.finalize()


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
        warmup_frames: int = 10,
        expansion_factor: float = 1.8,
        smooth_landmarks: bool | None = None,
        enable_segmentation: bool | None = None,
        reliability_min_visibility: float = 0.5,
        landmark_smoothing_alpha: float | None = None,
        enable_recovery_pass: bool = False,
        recovery_miss_threshold: int = 2,
        debug_recorder: RoiDebugRecorder | None = None,
    ) -> None:
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.refresh_period = max(1, refresh_period)
        self.max_misses = max(1, max_misses)
        self.last_box: Optional[List[int]] = None
        self.misses = 0
        self.frame_idx = 0
        self.roi_state = RoiState(
            warmup_frames=warmup_frames,
            fallback_misses=max_misses,
            expansion_factor=expansion_factor,
            recorder=debug_recorder,
        )
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
        self._debug_recorder = debug_recorder

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
        annotated_image: Optional[np.ndarray] = None

        roi, fallback_to_full = self.roi_state.next_roi(width, height)
        refresh_now = (self.frame_idx % self.refresh_period) == 0
        use_full_frame = fallback_to_full or refresh_now
        pose_ok = False
        landmarks_output: Optional[List[Landmark]] = None
        crop_box: Optional[Sequence[int]] = None
        letterbox_transform: Optional[LetterboxTransform] = None

        if use_full_frame:
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
            if results_full.pose_landmarks:
                self._recovery_misses_full = 0
                landmarks_raw = landmarks_from_proto(results_full.pose_landmarks.landmark)
                pose_ok = pose_reliable(landmarks_raw, visibility_threshold=self.reliability_min_visibility)
                landmarks_output = (
                    self._smoother(landmarks_raw) if self._smoother is not None else landmarks_raw
                )
                annotated_image = image_bgr.copy()
                self.mp_drawing.draw_landmarks(annotated_image, results_full.pose_landmarks, POSE_CONNECTIONS)
                bbox = bounding_box_from_landmarks(landmarks_output, width, height)
                if bbox is not None:
                    smoothed_bbox = smooth_bounding_box(
                        self._smoothed_bbox,
                        bbox,
                        factor=self.smooth_factor,
                        width=width,
                        height=height,
                    )
                    self._smoothed_bbox = smoothed_bbox
                    crop_box = expand_and_clip_box(smoothed_bbox, width, height, self.crop_margin)
                    self.roi_state.update_success(smoothed_bbox, width, height)
                    self.last_box = list(crop_box)
                else:
                    self._smoothed_bbox = None
                    self.last_box = None
                    self.roi_state.update_failure(width, height)
            else:
                self._recovery_misses_full += 1
                self._smoothed_bbox = None
                self.last_box = None
                if self._smoother is not None:
                    self._smoother.reset()
                self.roi_state.update_failure(width, height)
        else:
            x1, y1, x2, y2 = roi
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                self._smoothed_bbox = None
                self.last_box = None
                if self._smoother is not None:
                    self._smoother.reset()
                self.roi_state.update_failure(width, height)
            else:
                crop_resized, letterbox_transform = _resize_with_letterbox(crop, self.target_size)
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
                if results_crop.pose_landmarks:
                    self._recovery_misses_crop = 0
                    landmarks_full, landmark_list = _rescale_landmarks_from_crop(
                        results_crop.pose_landmarks.landmark,
                        roi,
                        width,
                        height,
                        self.landmark_pb2,
                        letterbox_transform=letterbox_transform,
                    )
                    pose_ok = pose_reliable(landmarks_full, visibility_threshold=self.reliability_min_visibility)
                    landmarks_output = (
                        self._smoother(landmarks_full) if self._smoother is not None else landmarks_full
                    )
                    annotated_image = image_bgr.copy()
                    self.mp_drawing.draw_landmarks(annotated_image, landmark_list, POSE_CONNECTIONS)
                    bbox = bounding_box_from_landmarks(landmarks_output, width, height)
                    if bbox is not None:
                        smoothed_bbox = smooth_bounding_box(
                            self._smoothed_bbox,
                            bbox,
                            factor=self.smooth_factor,
                            width=width,
                            height=height,
                        )
                        self._smoothed_bbox = smoothed_bbox
                        crop_box = expand_and_clip_box(smoothed_bbox, width, height, self.crop_margin)
                        self.roi_state.update_success(smoothed_bbox, width, height)
                        self.last_box = list(crop_box)
                    else:
                        self._smoothed_bbox = None
                        self.last_box = None
                        self.roi_state.update_failure(width, height)
                else:
                    self._recovery_misses_crop += 1
                    if self._smoother is not None:
                        self._smoother.reset()
                    self._smoothed_bbox = None
                    self.last_box = None
                    self.roi_state.update_failure(width, height)

        crop_box = self.last_box if crop_box is None else crop_box
        roi_for_debug = crop_box if crop_box is not None else [0, 0, width, height]
        result_crop_box = (0, 0, width, height) if landmarks_output is not None else crop_box
        pose_result = PoseResult(
            landmarks=landmarks_output,
            annotated_image=image_bgr if annotated_image is None else annotated_image,
            crop_box=result_crop_box,
            pose_ok=pose_ok,
        )

        warmup_active = self.frame_idx < self.roi_state.warmup_frames or not self.roi_state.has_pose
        self.roi_state.emit_debug(
            frame_idx=self.frame_idx,
            input_size=(width, height),
            output_size=self.target_size,
            crop_used=not use_full_frame,
            roi=roi_for_debug,
            pose_ok=pose_ok,
            warmup_active=warmup_active,
            fallback_to_full=use_full_frame,
        )

        self.frame_idx += 1
        return pose_result

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
