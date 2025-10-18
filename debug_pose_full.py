"""Utility script to visualise MediaPipe pose detections on a single frame."""

import os

import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing_utils
from mediapipe.python.solutions import pose as mp_pose_module

IMG_PATH = "data/processed/images/1-Squat_Own/frame_0020.jpg"

image = cv2.imread(IMG_PATH)
if image is None:
    print(f"[ERROR] Could not read image at {IMG_PATH}")
    raise SystemExit(1)

pose = mp_pose_module.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)
pose.close()

if not results.pose_landmarks:
    print("[DEBUG] MediaPipe did not detect a pose in frame 20 (no crop).")
    cv2.imwrite("debug_full_no_pose.jpg", image)
    raise SystemExit(0)

annotated = image.copy()
mp_drawing_utils.draw_landmarks(annotated, results.pose_landmarks, mp_pose_module.POSE_CONNECTIONS)
os.makedirs("debug_outputs", exist_ok=True)
cv2.imwrite("debug_outputs/debug_full_annotated.jpg", annotated)
print("[DEBUG] MediaPipe detected a pose. Annotated image saved to debug_outputs/debug_full_annotated.jpg")
