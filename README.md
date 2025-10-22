# FIT CONTROL

**Repository:** `TFM_FIT_CONTROL`
**Author:** Daniel Romero
**Version:** 2.4 (20-10-2025)

---

## Description

This project delivers an exercise-analysis system powered by computer vision. From strength-training videos (squat, bench press, deadlift), the application—available both as a desktop client and a Streamlit experience—aims to:

- Detect the exercise type (squat / bench press / deadlift) and camera view (front / side) with confidence scores.
- Automatically count repetitions.
- Calculate key joint angles (knee, hip, shoulder).
- Detect potential technique errors with ≥90% target accuracy.
- Provide a streamlined web demo to visualize the outcomes.

The core workflow includes:

- **Video preprocessing:** frame extraction, resizing, normalization, filtering (Gaussian, CLAHE), and ROI cropping.
- **Pose estimation:** MediaPipe Pose for landmark extraction and angle / angular-velocity / symmetry calculations.
- **Analysis:** repetition detection, metric aggregation, and alerting on technique deviations.

---

## Installation

The project ships with `pyproject.toml`. Use your preferred toolchain.

### Option A — uv (recommended)

~~~
# From repo root
uv venv
uv pip install -e .
uv run streamlit run src/app.py
~~~

### Option B — venv + pip

~~~
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -e .
streamlit run src/app.py
~~~

### (Legacy) Conda

A legacy `environment.yml` exists under `OLD/`, but it’s no longer the primary path.

## Quick Start

~~~
streamlit run src/app.py
~~~

## UI Steps

1. Upload a training video.
2. Detect exercise & view (auto or manual).
3. Configure parameters (FPS, rotation, debug video, etc.).
4. Run the analysis (asynchronous; progress shown).
5. Results: repetition count, metrics, optional debug video, downloads.

---

## What’s new in 2.3

**Documentation and project structure refresh.** This release focuses on improving discoverability and aligning the repository layout with the refactored modules:

- Updated README with the latest release information and reorganized change-log sections.
- Added a high-level overview of the current project structure to accelerate onboarding.
- Clarified the purpose of legacy assets stored under `OLD/` for historical reference.

## What’s new in 2.0

**Automatic Exercise & View Detection (MVP)**

- Added `src/exercise_detection/exercise_detector.py` (and `src/exercise_detection/__init__.py`).
- Detects squat / bench press / deadlift + front / side with a confidence score.
- Uses MediaPipe landmarks with heuristics (ROMs, pelvis displacement, wrist travel) and new view cues:
  - Shoulder yaw (3D angle from left/right shoulder depth).
  - Shoulder z-depth delta (asymmetry).
  - Normalized shoulder width (width/torso-length) with stability check.

**Streamlit integration**

- “Detectar ejercicio (beta)” button now runs detection, caches results, and updates the selectbox.
- Detection results (label, view, confidence) are shown in the UI.
- `run_pipeline(..., prefetched_detection=...)` accepts cached UI results to skip duplicate detector runs.

**RunStats now includes:**

- `exercise_selected` (UI choice)
- `exercise_detected`, `view_detected`, `detection_confidence`
- Results table shows both selected vs detected for transparency.

**Debugging & tests**

- Lightweight debug logging prints a single `DET DEBUG` line with kinematic and view stats (ROMs, width mean/std, yaw median, z-depth median).
- Synthetic fixtures for squat/bench press/deadlift and front vs side view checks.

## What’s new in 1.1

**Parity achieved between Streamlit and Desktop UI.** Both front-ends now call the same unified pipeline and config:

- Unified `Config` (dataclasses) with SHA1 **fingerprint** for reproducibility.
- Single entry point `run_pipeline(video_path, cfg)` returning a rich `Report`.
- Robust FPS detection with fallbacks + guardrails (`min_frames`, `min_fps`).
- One *single* resize stage (documented in config) before pose estimation.
- Rep counting based on valleys with **prominence / min distance / refractory**.
- Streamlit & GUI surface **run stats**, warnings and `skip_reason`.
- Pinned OpenCV/SciPy in `environment.yml` for stability.

---

## Project structure (v2.3)

| Path | Description |
| --- | --- |
| `src/app.py` | Streamlit entry point that wires UI controls to the exercise-analysis pipeline. |
| `src/pipeline/` | Preprocessing, pose-estimation, and repetition-analysis building blocks. |
| `src/exercise_detection/` | Heuristics and models that infer exercise type and camera view. |
| `src/config/` | Dataclasses, defaults, and validation utilities governing pipeline execution. |
| `src/utils/` | Shared helpers for I/O, logging, math utilities, and MediaPipe integrations. |
| `tests/` | Automated tests (unit + integration) that validate the detection pipeline and utilities. |
| `docs/` | Design notes, research references, and extended documentation. |
| `OLD/` | Historical experiments, legacy scripts, and the deprecated `environment.yml`. |
| `pyproject.toml` | Project metadata, dependency declarations, and entry-points for tooling. |
| `uv.lock` | Locked dependency versions for reproducible uv-based environments. |
| `project_tree.cmd` | Helper script to regenerate a summarized repository tree. |

---
