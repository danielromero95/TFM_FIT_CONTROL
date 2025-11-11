"""Constantes tunables específicas de la detección de ejercicio.

Separar estos valores aclara que no forman parte de la configuración global de
la aplicación sino de las heurísticas concretas.  Aquí agrupamos los umbrales
kinemáticos y geométricos que alimentan al clasificador.
"""

from __future__ import annotations

from typing import Tuple

MIN_VALID_FRAMES = 20
SQUAT_KNEE_ROM_THRESHOLD_DEG = 40.0
SQUAT_HIP_ROM_THRESHOLD_DEG = 30.0
SQUAT_PELVIS_DISPLACEMENT_THRESHOLD = 0.20  # normalizado por longitud del torso
SQUAT_KNEE_MIN_ANGLE_DEG = 130.0
SQUAT_HIP_MIN_ANGLE_DEG = 150.0
SQUAT_SIDE_SYMMETRY_TOLERANCE_DEG = 18.0

BENCH_ELBOW_ROM_THRESHOLD_DEG = 55.0
BENCH_WRIST_HORIZONTAL_THRESHOLD = 0.08
BENCH_TORSO_TILT_THRESHOLD_DEG = 40.0  # grados de desviación respecto a la vertical
BENCH_TORSO_MIN_DEG = 55.0
BENCH_PELVIS_DROP_NORM_MAX = 0.12
BENCH_LOWER_ROM_MAX_DEG = 25.0

DEADLIFT_HIP_ROM_THRESHOLD_DEG = 40.0
DEADLIFT_WRIST_VERTICAL_THRESHOLD = 0.12
DEADLIFT_TORSO_UPRIGHT_TARGET_DEG = 35.0

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
    "wrist_left_x",
    "wrist_left_y",
    "wrist_right_x",
    "wrist_right_y",
    "shoulder_width_norm",
    "shoulder_yaw_deg",
    "shoulder_z_delta_abs",
    "torso_tilt_deg",
    "ankle_width_norm",
)
