# FIT CONTROL

**Repository:** `TFM_FIT_CONTROL`  
**Author:** Daniel Romero  
**Version:** 2.2 (20-10-2025)

---

## Description

This project aims to build a exercise analysis system based on computer vision.
From strength training videos (squat, bench press, deadlift), the app (desktop app and streamlit app) should:

- Detect the exercise type (squat/bench/deadlift) and camera view (front/side) with confidence.
- Automatically count repetitions.
- Calculate key joint angles (knee, hip, shoulder).
- Detect potential technique errors with ≥90% target accuracy.
- Offer a minimal web demo to view results.


The general workflow includes:
- Video preprocessing: frame extraction, resizing, normalization, filtering (Gauss, CLAHE), ROI cropping.
- Pose estimation: MediaPipe Pose to extract landmarks and calculate angles, angular velocities, symmetry, etc.


## What’s new in 1.1

**Parity achieved between Streamlit and Desktop UI.** Both front-ends now call the same unified pipeline and config:

- Unified `Config` (dataclasses) with SHA1 **fingerprint** for reproducibility
- Single entry point `run_pipeline(video_path, cfg)` returning a rich `Report`
- Robust FPS detection with fallbacks + guardrails (`min_frames`, `min_fps`)
- One *single* resize stage (documented in config) before pose estimation
- Rep counting based on valleys with **prominence / min distance / refractory**
- Streamlit & GUI surface **run stats**, warnings and `skip_reason`
- CLI ↔ Streamlit parity test added
- Pinned OpenCV/SciPy in `environment.yml` for stability

## What’s new in 2.0
**Automatic Exercise & View Detection (MVP)**

New module: src/detect/exercise_detector.py (and src/detect/__init__.py).

- Detects squat / bench / deadlift + front / side with a confidence score.

- Uses MediaPipe landmarks with heuristics (ROMs, pelvis displacement, wrist travel) and new view cues:

- Shoulder yaw (3D angle from left/right shoulder depth).

- Shoulder z-depth delta (asymmetry).

- Normalized shoulder width (width/torso-length) with stability check.

**Streamlit integration**

“Detectar ejercicio (beta)” button now runs detection, caches results, and updates the selectbox.

Detection results (label, view, confidence) are shown in the UI.

Pipeline integration

run_pipeline(..., prefetched_detection=...) accepts the cached UI result to skip duplicate detector runs.


**RunStats now includes:**

- exercise_selected (UI choice)

- exercise_detected, view_detected, detection_confidence

- Results table shows both selected vs detected for transparency.


**Debugging & tests**

Lightweight debug logging prints a single DET DEBUG line with kinematic and view stats (ROMs, width mean/std, yaw median, z-depth median).

Unit tests:

Synthetic fixtures for squat/bench/deadlift and front vs side view checks.

## Installation

```bash
# 1) Create conda env
conda env create -f environment.yml
conda activate gym_env