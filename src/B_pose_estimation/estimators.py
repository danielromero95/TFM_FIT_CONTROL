# src/B_pose_estimation/estimators.py
import logging
from typing import Iterable, List, Dict, Optional, Tuple

import cv2
import numpy as np
from src.config import POSE_CONNECTIONS
from src.config.constants import MIN_DETECTION_CONFIDENCE
from src.config.settings import MODEL_COMPLEXITY

logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, static_image_mode=True, model_complexity=MODEL_COMPLEXITY, min_detection_confidence=MIN_DETECTION_CONFIDENCE):
        try:
            from mediapipe.python.solutions import pose as mp_pose_module, drawing_utils as mp_drawing
        except ImportError: raise
        self.mp_pose, self.mp_drawing = mp_pose_module, mp_drawing
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity, min_detection_confidence=min_detection_confidence)
    def estimate(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks: return None, image, None
        landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results.pose_landmarks.landmark]
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, POSE_CONNECTIONS)
        return landmarks, annotated_image, None
    def close(self):
        self.pose.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class CroppedPoseEstimator:
    def __init__(self, static_image_mode=True, model_complexity=MODEL_COMPLEXITY, min_detection_confidence=MIN_DETECTION_CONFIDENCE, crop_margin=0.15, target_size=(256, 256)):
        self.crop_margin, self.target_size = crop_margin, target_size
        try:
            from mediapipe.python.solutions import pose as mp_pose_module, drawing_utils as mp_drawing
        except ImportError: raise
        self.mp_pose, self.mp_drawing = mp_pose_module, mp_drawing
        self.pose_full = self.mp_pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity, min_detection_confidence=min_detection_confidence)
        self.pose_crop = self.mp_pose.Pose(static_image_mode=static_image_mode, model_complexity=model_complexity, min_detection_confidence=min_detection_confidence)
    def estimate(self, image_bgr):
        h0, w0 = image_bgr.shape[:2]
        results_full = self.pose_full.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results_full.pose_landmarks: return None, image_bgr, None
        xy_full = np.array([[lm.x * w0, lm.y * h0] for lm in results_full.pose_landmarks.landmark])
        x_min, y_min, x_max, y_max = xy_full[:, 0].min(), xy_full[:, 1].min(), xy_full[:, 0].max(), xy_full[:, 1].max()
        dx, dy = (x_max - x_min) * self.crop_margin, (y_max - y_min) * self.crop_margin
        x1, y1 = max(int(x_min - dx), 0), max(int(y_min - dy), 0)
        x2, y2 = min(int(x_max + dx), w0), min(int(y_max + dy), h0)
        crop_box = [x1, y1, x2, y2]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0: return None, image_bgr, None
        crop_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
        results_crop = self.pose_crop.process(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        annotated_crop = crop_resized.copy()
        if results_crop.pose_landmarks:
            landmarks_crop = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in results_crop.pose_landmarks.landmark]
            self.mp_drawing.draw_landmarks(annotated_crop, results_crop.pose_landmarks, POSE_CONNECTIONS)
        else: landmarks_crop = None
        return landmarks_crop, annotated_crop, crop_box
    def close(self):
        self.pose_full.close(); self.pose_crop.close()
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RoiPoseEstimator:
    def __init__(
        self,
        static_image_mode: bool = True,
        model_complexity: int = MODEL_COMPLEXITY,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        crop_margin: float = 0.15,
        target_size: Tuple[int, int] = (256, 256),
        refresh_period: int = 10,
        max_misses: int = 2,
    ):
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.refresh_period = max(1, refresh_period)
        self.max_misses = max(1, max_misses)
        self.last_box: Optional[List[int]] = None
        self.misses = 0
        self.frame_idx = 0

        try:
            from mediapipe.python.solutions import pose as mp_pose_module, drawing_utils as mp_drawing
            from mediapipe.framework.formats import landmark_pb2
        except ImportError:
            raise

        self.mp_pose = mp_pose_module
        self.mp_drawing = mp_drawing
        self.landmark_pb2 = landmark_pb2
        self.pose_full = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        self.pose_crop = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )

    def _expand_and_clip_box(self, bbox: Tuple[float, float, float, float], width: int, height: int) -> List[int]:
        x_min, y_min, x_max, y_max = bbox
        dx = (x_max - x_min) * self.crop_margin
        dy = (y_max - y_min) * self.crop_margin
        x1 = max(int(np.floor(x_min - dx)), 0)
        y1 = max(int(np.floor(y_min - dy)), 0)
        x2 = min(int(np.ceil(x_max + dx)), width)
        y2 = min(int(np.ceil(y_max + dy)), height)

        if x2 <= x1:
            if x1 >= width:
                x1 = max(width - 1, 0)
            x2 = min(width, x1 + 1)
            x1 = max(0, x2 - 1)
        if y2 <= y1:
            if y1 >= height:
                y1 = max(height - 1, 0)
            y2 = min(height, y1 + 1)
            y1 = max(0, y2 - 1)

        return [x1, y1, x2, y2]

    def _landmarks_to_dict(self, landmarks: Iterable) -> List[Dict[str, float]]:
        return [
            {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
            }
            for lm in landmarks
        ]

    def estimate(self, image_bgr):
        height, width = image_bgr.shape[:2]
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
            results_full = self.pose_full.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if not results_full.pose_landmarks:
                self.misses += 1
                self.last_box = None
                output = (None, image_bgr, None)
            else:
                landmarks = self._landmarks_to_dict(results_full.pose_landmarks.landmark)
                annotated_image = image_bgr.copy()
                self.mp_drawing.draw_landmarks(annotated_image, results_full.pose_landmarks, POSE_CONNECTIONS)
                xy_full = np.array(
                    [
                        [lm.x * width, lm.y * height]
                        for lm in results_full.pose_landmarks.landmark
                    ]
                )
                bbox = (
                    float(xy_full[:, 0].min()),
                    float(xy_full[:, 1].min()),
                    float(xy_full[:, 0].max()),
                    float(xy_full[:, 1].max()),
                )
                self.last_box = self._expand_and_clip_box(bbox, width, height)
                self.misses = 0
                output = (landmarks, annotated_image, self.last_box)
        else:
            x1, y1, x2, y2 = self.last_box  # type: ignore[misc]
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                self.misses += 1
                self.last_box = None
                output = (None, image_bgr, None)
            else:
                crop_resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)
                results_crop = self.pose_crop.process(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                if not results_crop.pose_landmarks:
                    self.misses += 1
                    if self.misses >= self.max_misses:
                        self.last_box = None
                    output = (None, image_bgr, None)
                else:
                    landmarks_full: List[Dict[str, float]] = []
                    mp_landmarks = []
                    scale_x = x2 - x1
                    scale_y = y2 - y1
                    for lm in results_crop.pose_landmarks.landmark:
                        x_full = (lm.x * scale_x + x1) / width if scale_x > 0 else 0.0
                        y_full = (lm.y * scale_y + y1) / height if scale_y > 0 else 0.0
                        x_clipped = float(np.clip(x_full, 0.0, 1.0))
                        y_clipped = float(np.clip(y_full, 0.0, 1.0))
                        landmarks_full.append(
                            {
                                "x": float(x_clipped),
                                "y": float(y_clipped),
                                "z": float(lm.z),
                                "visibility": float(lm.visibility),
                            }
                        )
                        mp_landmarks.append(
                            self.landmark_pb2.NormalizedLandmark(
                                x=x_clipped,
                                y=y_clipped,
                                z=float(lm.z),
                                visibility=float(lm.visibility),
                            )
                        )

                    annotated_image = image_bgr.copy()
                    landmark_list = self.landmark_pb2.NormalizedLandmarkList(landmark=mp_landmarks)
                    self.mp_drawing.draw_landmarks(annotated_image, landmark_list, POSE_CONNECTIONS)

                    xy_full = np.array(
                        [
                            [lm["x"] * width, lm["y"] * height]
                            for lm in landmarks_full
                        ]
                    )
                    bbox = (
                        float(xy_full[:, 0].min()),
                        float(xy_full[:, 1].min()),
                        float(xy_full[:, 0].max()),
                        float(xy_full[:, 1].max()),
                    )
                    self.last_box = self._expand_and_clip_box(bbox, width, height)
                    self.misses = 0
                    output = (landmarks_full, annotated_image, self.last_box)

        self.frame_idx += 1
        return output

    def close(self):
        self.pose_full.close()
        self.pose_crop.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
