"""Constantes tunables específicas de la detección de ejercicio.

Separar estos valores aclara que no forman parte de la configuración global de
la aplicación sino de las heurísticas concretas.  Aquí agrupamos los umbrales
kinemáticos y geométricos que alimentan al clasificador.
"""

from __future__ import annotations

from typing import Tuple

MIN_VALID_FRAMES = 20

# --- Segmentación temporal -------------------------------------------------
KNEE_DOWN_THRESHOLD_DEG = 115.0
KNEE_UP_THRESHOLD_DEG = 150.0
BAR_DROP_MIN_NORM = 0.06
EVENT_MIN_GAP_SECONDS = 0.30

# --- Heurísticas para press banca -----------------------------------------
BENCH_TORSO_HORIZONTAL_DEG = 60.0
BENCH_ELBOW_ROM_MIN_DEG = 35.0
BENCH_KNEE_ROM_MAX_DEG = 30.0
BENCH_HIP_ROM_MAX_DEG = 30.0
BENCH_BAR_RANGE_MIN_NORM = 0.15
BENCH_BAR_HORIZONTAL_STD_MAX = 0.05
BENCH_HIP_RANGE_MAX_NORM = 0.08

# --- Heurísticas para sentadilla ------------------------------------------
SQUAT_KNEE_BOTTOM_MAX_DEG = 110.0
SQUAT_HIP_BOTTOM_MAX_DEG = 140.0
SQUAT_TORSO_TILT_MAX_DEG = 45.0
SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM = 0.18
SQUAT_ELBOW_BOTTOM_MIN_DEG = 60.0
SQUAT_ELBOW_BOTTOM_MAX_DEG = 130.0
SQUAT_KNEE_FORWARD_MIN_NORM = 0.05
SQUAT_MIN_ROM_DEG = 35.0
SQUAT_TIBIA_MAX_DEG = 32.0

# --- Heurísticas para peso muerto -----------------------------------------
DEADLIFT_KNEE_BOTTOM_MIN_DEG = 120.0
DEADLIFT_TORSO_TILT_MIN_DEG = 35.0
DEADLIFT_WRIST_HIP_DIFF_MIN_NORM = 0.20
DEADLIFT_ELBOW_MIN_DEG = 160.0
DEADLIFT_KNEE_FORWARD_MAX_NORM = 0.05
DEADLIFT_BAR_ANKLE_MAX_NORM = 0.08
DEADLIFT_HIP_ROM_MIN_DEG = 30.0
DEADLIFT_BAR_RANGE_MIN_NORM = 0.10

# --- Clasificación de vista ------------------------------------------------
VIEW_FRONT_WIDTH_THRESHOLD = 0.55      # anchura de hombros / torso normalizada
VIEW_WIDTH_STD_THRESHOLD = 0.12
YAW_FRONT_MAX_DEG = 20.0
YAW_SIDE_MIN_DEG = 25.0
Z_DELTA_FRONT_MAX = 0.08               # unidades normalizadas de MediaPipe; pequeño implica vista frontal
SIDE_WIDTH_MAX = 0.50                  # si la anchura normalizada ≤ este valor, asumimos vista lateral
ANKLE_FRONT_WIDTH_THRESHOLD = 0.50
ANKLE_SIDE_WIDTH_MAX = 0.40
ANKLE_WIDTH_STD_THRESHOLD = 0.12

VIEW_SCORE_PER_EVIDENCE_THRESHOLD = 0.22
VIEW_MARGIN_PER_EVIDENCE_THRESHOLD = 0.10
VIEW_FRONT_FALLBACK_YAW_DEG = 24.0
VIEW_SIDE_FALLBACK_YAW_DEG = 27.0

CLASSIFICATION_MARGIN = 0.12
MIN_CONFIDENCE_SCORE = 0.45

DEFAULT_SAMPLING_RATE = 30.0

FEATURE_NAMES: Tuple[str, ...] = (
    "knee_angle_left",
    "knee_angle_right",
    "hip_angle_left",
    "hip_angle_right",
    "elbow_angle_left",
    "elbow_angle_right",
    "shoulder_angle_left",
    "shoulder_angle_right",
    "pelvis_y",
    "torso_length",
    "torso_length_world",
    "wrist_left_x",
    "wrist_left_y",
    "wrist_right_x",
    "wrist_right_y",
    "shoulder_width_norm",
    "shoulder_yaw_deg",
    "shoulder_z_delta_abs",
    "torso_tilt_deg",
    "ankle_width_norm",
    "shoulder_left_x",
    "shoulder_left_y",
    "shoulder_right_x",
    "shoulder_right_y",
    "hip_left_x",
    "hip_left_y",
    "hip_right_x",
    "hip_right_y",
    "knee_left_x",
    "knee_left_y",
    "knee_right_x",
    "knee_right_y",
    "ankle_left_x",
    "ankle_left_y",
    "ankle_right_x",
    "ankle_right_y",
    "elbow_left_x",
    "elbow_left_y",
    "elbow_right_x",
    "elbow_right_y",
)
