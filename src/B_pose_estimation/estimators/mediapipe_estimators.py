"""Estimadores de pose basados en Mediapipe listos para usar."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import POSE_CONNECTIONS
from src.config.constants import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE
from src.config.settings import MODEL_COMPLEXITY

from ..geometry import (
    bounding_box_from_landmarks,
    expand_and_clip_box,
    landmarks_from_proto,
    smooth_bounding_box,
)
from ..types import Landmark, PoseResult
from .base import PoseEstimatorBase
from .mediapipe_pool import PoseGraphPool


def _resize_with_padding(
    image: np.ndarray, target_size: Tuple[int, int]
) -> tuple[np.ndarray, float, int, int]:
    """Resize ``image`` preserving aspect ratio and pad to ``target_size``.

    Returns the padded image along with ``scale`` (applied to the original
    frame) and the ``pad_x``/``pad_y`` offsets used to center the resized crop.
    """

    target_w, target_h = target_size
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return (
            np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype),
            0.0,
            0,
            0,
        )

    scale = min(target_w / float(width), target_h / float(height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    channels = image.shape[2] if image.ndim == 3 else 1
    if resized.shape[:2] != (resized_h, resized_w):
        resized = cv2.resize(resized, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    if resized.shape[:2] != (resized_h, resized_w):
        resized = np.zeros((resized_h, resized_w, channels), dtype=image.dtype)
    else:
        if resized.ndim == 2 and channels == 3:
            resized = np.repeat(resized[:, :, np.newaxis], channels, axis=2)
    resized_h, resized_w = resized.shape[:2]
    padded = np.zeros((target_h, target_w, channels), dtype=image.dtype)

    pad_x = max(0, int((target_w - resized_w) // 2))
    pad_y = max(0, int((target_h - resized_h) // 2))
    end_x = min(target_w, pad_x + resized_w)
    end_y = min(target_h, pad_y + resized_h)
    padded[pad_y:end_y, pad_x:end_x] = resized[: end_y - pad_y, : end_x - pad_x]
    return padded, float(scale), pad_x, pad_y


def _remap_landmarks_from_padded_result(
    landmarks: List[object],
    *,
    crop_box: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
    pad_x: int,
    pad_y: int,
    scale: float,
    frame_width: int,
    frame_height: int,
    to_frame: bool,
) -> list[Landmark]:
    """Undo padding/letterboxing and remap landmarks to crop or frame space."""

    target_w, target_h = target_size
    x1, y1, x2, y2 = crop_box
    crop_w = max(1.0, float(x2 - x1))
    crop_h = max(1.0, float(y2 - y1))
    safe_scale = float(scale) if scale > 0 else 1.0

    mapped: list[Landmark] = []
    for lm in landmarks:
        try:
            x_norm = float(getattr(lm, "x", np.nan))
            y_norm = float(getattr(lm, "y", np.nan))
        except Exception:
            continue
        if not np.isfinite(x_norm) or not np.isfinite(y_norm):
            continue

        x_target = x_norm * float(target_w) - float(pad_x)
        y_target = y_norm * float(target_h) - float(pad_y)

        x_crop = x_target / safe_scale
        y_crop = y_target / safe_scale

        if to_frame:
            x_final = (x_crop + float(x1)) / float(max(frame_width, 1))
            y_final = (y_crop + float(y1)) / float(max(frame_height, 1))
        else:
            x_final = x_crop / crop_w
            y_final = y_crop / crop_h

        z_val = float(getattr(lm, "z", np.nan))
        if np.isfinite(z_val):
            z_val /= safe_scale

        mapped.append(
            Landmark(
                x=float(np.clip(x_final, 0.0, 1.0)),
                y=float(np.clip(y_final, 0.0, 1.0)),
                z=z_val,
                visibility=float(getattr(lm, "visibility", np.nan)),
            )
        )

    return mapped


class PoseEstimator(PoseEstimatorBase):
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = MODEL_COMPLEXITY,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose = None
        self._key: Optional[Tuple[bool, int, float, float]] = None
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
            )

    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        self._ensure_pose()
        results = self.pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None)
        landmarks = landmarks_from_proto(results.pose_landmarks.landmark)
        annotated_image = image_bgr.copy()
        self.mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, POSE_CONNECTIONS)
        return PoseResult(landmarks=landmarks, annotated_image=annotated_image, crop_box=None)

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
    ) -> None:
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose_full = None
        self.pose_crop = None
        self._key_full: Optional[Tuple[bool, int, float, float]] = None
        self._key_crop: Optional[Tuple[bool, int, float, float]] = None
        self.smooth_factor = 0.65
        self._smoothed_bbox: Optional[Tuple[float, float, float, float]] = None
        PoseGraphPool._ensure_imports()
        self.mp_pose = PoseGraphPool.mp_pose
        self.mp_drawing = PoseGraphPool.mp_drawing
        from mediapipe.framework.formats import landmark_pb2

        self.landmark_pb2 = landmark_pb2

    def _ensure_graphs(self) -> None:
        if self.pose_full is None:
            self.pose_full, self._key_full = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        if self.pose_crop is None:
            self.pose_crop, self._key_crop = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        self._ensure_graphs()
        height, width = image_bgr.shape[:2]
        results_full = self.pose_full.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results_full.pose_landmarks:
            self._smoothed_bbox = None
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None)

        landmarks_full = landmarks_from_proto(results_full.pose_landmarks.landmark)
        annotated_full = image_bgr.copy()
        self.mp_drawing.draw_landmarks(annotated_full, results_full.pose_landmarks, POSE_CONNECTIONS)
        bbox = bounding_box_from_landmarks(landmarks_full, width, height)
        if bbox is None:
            self._smoothed_bbox = None
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None)

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
            return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None)

        crop_resized, scale, pad_x, pad_y = _resize_with_padding(crop, self.target_size)
        results_crop = self.pose_crop.process(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        annotated_crop = crop_resized.copy()
        if results_crop.pose_landmarks:
            landmarks_crop = _remap_landmarks_from_padded_result(
                list(results_crop.pose_landmarks.landmark),
                crop_box=tuple(crop_box),
                target_size=self.target_size,
                pad_x=pad_x,
                pad_y=pad_y,
                scale=scale,
                frame_width=width,
                frame_height=height,
                to_frame=False,
            )
            landmark_list = self.mp_pose.NormalizedLandmarkList(landmark=results_crop.pose_landmarks.landmark)
            self.mp_drawing.draw_landmarks(annotated_crop, landmark_list, POSE_CONNECTIONS)
        else:
            # Si el recorte falla, usamos el resultado del frame completo para no perder la detección.
            return PoseResult(landmarks=landmarks_full, annotated_image=annotated_full, crop_box=None)
        return PoseResult(landmarks=landmarks_crop, annotated_image=annotated_crop, crop_box=crop_box)

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
        self.smooth_factor = 0.65
        self._smoothed_bbox: Optional[Tuple[float, float, float, float]] = None

        PoseGraphPool._ensure_imports()
        self.mp_pose = PoseGraphPool.mp_pose
        self.mp_drawing = PoseGraphPool.mp_drawing
        from mediapipe.framework.formats import landmark_pb2

        self.landmark_pb2 = landmark_pb2
        self.pose_full = None
        self.pose_crop = None
        self._key_full: Optional[Tuple[bool, int, float, float]] = None
        self._key_crop: Optional[Tuple[bool, int, float, float]] = None
        self.prev_landmarks: Optional[list[Landmark]] = None
        self.landmark_smooth = 0.65

    def _ensure_graphs(self) -> None:
        if self.pose_full is None:
            self.pose_full, self._key_full = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        if self.pose_crop is None:
            self.pose_crop, self._key_crop = PoseGraphPool.acquire(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

    def _smooth_landmarks(self, current: list[Landmark]) -> list[Landmark]:
        if not current:
            return current
        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current):
            self.prev_landmarks = current
            return current

        alpha = float(min(0.99, max(0.0, self.landmark_smooth)))
        smoothed: list[Landmark] = []
        for prev, cur in zip(self.prev_landmarks, current):
            smoothed.append(
                Landmark(
                    x=float(prev.x * alpha + cur.x * (1.0 - alpha)),
                    y=float(prev.y * alpha + cur.y * (1.0 - alpha)),
                    z=float(prev.z * alpha + cur.z * (1.0 - alpha)),
                    visibility=float(prev.visibility * alpha + cur.visibility * (1.0 - alpha)),
                )
            )
        self.prev_landmarks = smoothed
        return smoothed

    def estimate(self, image_bgr: np.ndarray) -> PoseResult:
        self._ensure_graphs()
        height, width = image_bgr.shape[:2]

        def _detect_full_frame() -> PoseResult:
            results_full = self.pose_full.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if not results_full.pose_landmarks:
                self.misses += 1
                self.last_box = None
                self._smoothed_bbox = None
                return PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None)

            landmarks = landmarks_from_proto(results_full.pose_landmarks.landmark)
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
            return PoseResult(landmarks=landmarks, annotated_image=annotated_image, crop_box=self.last_box)

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
            output = _detect_full_frame()
        else:
            x1, y1, x2, y2 = self.last_box  # type: ignore[misc]
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                self.misses += 1
                # Si el recorte está vacío, intentamos una detección completa para recuperar la caja.
                output = _detect_full_frame()
            else:
                crop_resized, scale, pad_x, pad_y = _resize_with_padding(crop, self.target_size)
                results_crop = self.pose_crop.process(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                if not results_crop.pose_landmarks:
                    self.misses += 1
                    # Recaemos en el detector completo para recuperarnos rápido cuando el seguimiento falla.
                    if self.misses >= self.max_misses or self.misses == 1:
                        output = _detect_full_frame()
                    else:
                        output = PoseResult(landmarks=None, annotated_image=image_bgr, crop_box=None)
                else:
                    landmarks_full = _remap_landmarks_from_padded_result(
                        list(results_crop.pose_landmarks.landmark),
                        crop_box=(x1, y1, x2, y2),
                        target_size=self.target_size,
                        pad_x=pad_x,
                        pad_y=pad_y,
                        scale=scale,
                        frame_width=width,
                        frame_height=height,
                        to_frame=True,
                    )

                    smoothed = self._smooth_landmarks(landmarks_full)
                    annotated_image = image_bgr.copy()
                    landmark_list = self.landmark_pb2.NormalizedLandmarkList(
                        landmark=[
                            self.landmark_pb2.NormalizedLandmark(
                                x=float(lm.x), y=float(lm.y), z=float(lm.z), visibility=float(lm.visibility)
                            )
                            for lm in smoothed
                        ]
                    )
                    self.mp_drawing.draw_landmarks(annotated_image, landmark_list, POSE_CONNECTIONS)
                    bbox = bounding_box_from_landmarks(smoothed, width, height)
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
                    output = PoseResult(landmarks=smoothed, annotated_image=annotated_image, crop_box=self.last_box)

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
        self.prev_landmarks = None


__all__ = ["PoseEstimator", "CroppedPoseEstimator", "RoiPoseEstimator"]
