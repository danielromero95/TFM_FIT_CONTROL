"""Utility script to validate the cropped pose estimator on a single frame."""

from __future__ import annotations

import os

import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils as mp_drawing_utils
from mediapipe.python.solutions import pose as mp_pose_module

IMG_PATH = "data/processed/images/1-Squat_Own/frame_0020.jpg"
CROP_MARGIN = 0.15
TARGET_SIZE = (256, 256)


class CroppedPoseEstimator:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        crop_margin: float = 0.15,
        target_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.crop_margin = crop_margin
        self.target_size = target_size
        self.mp_pose_module = mp_pose_module
        self.mp_drawing_utils = mp_drawing_utils

        self.pose_full = self.mp_pose_module.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        self.pose_crop = self.mp_pose_module.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )

    def estimate_and_crop(self, image_bgr):
        height, width = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results_full = self.pose_full.process(image_rgb)
        if not results_full.pose_landmarks:
            return None, image_bgr

        landmarks = results_full.pose_landmarks.landmark
        coords = np.array([[lm.x * width, lm.y * height] for lm in landmarks])

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        dx = (x_max - x_min) * self.crop_margin
        dy = (y_max - y_min) * self.crop_margin
        x1 = max(int(x_min - dx), 0)
        y1 = max(int(y_min - dy), 0)
        x2 = min(int(x_max + dx), width - 1)
        y2 = min(int(y_max + dy), height - 1)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, image_bgr

        target_width, target_height = self.target_size
        crop_resized = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        results_crop = self.pose_crop.process(crop_rgb)

        landmarks_crop = []
        annotated_crop = crop_resized.copy()
        if results_crop.pose_landmarks:
            for landmark in results_crop.pose_landmarks.landmark:
                landmarks_crop.append(
                    {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                )
            self.mp_drawing_utils.draw_landmarks(
                annotated_crop,
                results_crop.pose_landmarks,
                self.mp_pose_module.POSE_CONNECTIONS,
            )
        else:
            landmarks_crop = None

        return landmarks_crop, annotated_crop

    def close(self) -> None:
        self.pose_full.close()
        self.pose_crop.close()


if __name__ == "__main__":
    image = cv2.imread(IMG_PATH)
    if image is None:
        print(f"[ERROR] Could not read {IMG_PATH}")
        raise SystemExit(1)

    estimator = CroppedPoseEstimator(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        crop_margin=CROP_MARGIN,
        target_size=TARGET_SIZE,
    )

    landmarks_crop, annotated_crop = estimator.estimate_and_crop(image)
    estimator.close()

    os.makedirs("debug_outputs", exist_ok=True)
    cv2.imwrite("debug_outputs/debug_full_for_crop.jpg", image)
    if landmarks_crop is None:
        print("[DEBUG] CroppedPoseEstimator did not detect anything in the final crop.")
        cv2.imwrite("debug_outputs/debug_crop_empty.jpg", annotated_crop)
    else:
        print(
            f"[DEBUG] CroppedPoseEstimator detected {len(landmarks_crop)} landmarks in the final crop."
        )
        cv2.imwrite("debug_outputs/debug_crop_annotated.jpg", annotated_crop)
