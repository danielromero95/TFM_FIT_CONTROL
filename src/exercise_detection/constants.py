"""Umbrales numéricos centralizados para el clasificador de ejercicios.

Los valores definidos aquí se mantienen intencionalmente fáciles de ajustar
cuando se calibra el detector con nuevas grabaciones. Cada módulo debe importar
los umbrales desde este archivo en lugar de fijar literales, lo que mantiene
los experimentos reproducibles y auditables.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Muestreo y validación
# ---------------------------------------------------------------------------

# Frecuencia esperada (fotogramas por segundo) para los *landmarks* de MediaPipe
DEFAULT_SAMPLING_RATE = 30.0
# Clip más corto (en fotogramas válidos) que intentamos puntuar
MIN_VALID_FRAMES = 20

# Suavizado Savitzky–Golay (expresado en segundos para facilitar el ajuste)
# Ventana adaptable al FPS efectivo
SMOOTHING_WINDOW_SECONDS = 0.35
SMOOTHING_POLY_ORDER = 2

# ---------------------------------------------------------------------------
# Umbrales de segmentación
# ---------------------------------------------------------------------------

KNEE_DOWN_THRESHOLD_DEG = 115.0
KNEE_UP_THRESHOLD_DEG = 150.0
BAR_DROP_MIN_NORM = 0.06
EVENT_MIN_GAP_SECONDS = 0.30

# ---------------------------------------------------------------------------
# Heurísticas para press de banca
# ---------------------------------------------------------------------------

BENCH_TORSO_HORIZONTAL_DEG = 60.0
BENCH_ELBOW_ROM_MIN_DEG = 35.0
BENCH_KNEE_ROM_MAX_DEG = 30.0
BENCH_HIP_ROM_MAX_DEG = 30.0
BENCH_BAR_RANGE_MIN_NORM = 0.15
BENCH_BAR_HORIZONTAL_STD_MAX = 0.05
BENCH_HIP_RANGE_MAX_NORM = 0.08
BENCH_GATE_BONUS = 1.5
BENCH_ELBOW_ROM_GATE_FACTOR = 1.4

BENCH_POSTURE_WEIGHT = 0.8
BENCH_ELBOW_ROM_WEIGHT = 0.6
BENCH_BAR_RANGE_WEIGHT = 0.5
BENCH_ROM_PENALTY_WEIGHT = 0.4
BENCH_BAR_HORIZONTAL_PENALTY_WEIGHT = 0.7
BENCH_LOWER_BODY_PENALTY_WEIGHT = 0.6

# ---------------------------------------------------------------------------
# Heurísticas para sentadilla
# ---------------------------------------------------------------------------

SQUAT_KNEE_BOTTOM_MAX_DEG = 110.0
SQUAT_HIP_BOTTOM_MAX_DEG = 140.0
SQUAT_TORSO_TILT_MAX_DEG = 45.0
SQUAT_WRIST_SHOULDER_DIFF_MAX_NORM = 0.18
SQUAT_ELBOW_BOTTOM_MIN_DEG = 60.0
SQUAT_ELBOW_BOTTOM_MAX_DEG = 130.0
SQUAT_KNEE_FORWARD_MIN_NORM = 0.06
SQUAT_TIBIA_MAX_DEG = 32.0
SQUAT_MIN_ROM_DEG = 35.0
SQUAT_BAR_ABOVE_HIP_MIN_NORM = 0.06
SQUAT_BAR_BELOW_HIP_MAX_NORM = -0.06
SQUAT_ARM_HIGH_FRACTION_STRONG = 0.65
SQUAT_ARM_HIGH_FRACTION_LOW = 0.20

SQUAT_DEPTH_WEIGHT = 0.9
SQUAT_TORSO_WEIGHT = 0.6
SQUAT_ARM_BONUS_WEIGHT = 0.7
SQUAT_ARM_HEIGHT_WEIGHT = 0.9
SQUAT_KNEE_FORWARD_WEIGHT = 0.5
SQUAT_TIBIA_PENALTY_WEIGHT = 0.6
SQUAT_ROM_WEIGHT = 0.5
SQUAT_ARM_PENALTY_FACTOR = 0.65
SQUAT_ARM_HIGH_BONUS_WEIGHT = 0.8
SQUAT_ARM_LOW_PENALTY_WEIGHT = 0.6
SQUAT_HINGE_PENALTY_WEIGHT = 0.5
SQUAT_BAR_BELOW_HIP_PENALTY_WEIGHT = 0.75
ARM_ABOVE_HIP_HIGH_FRAC = 0.60
ARM_ABOVE_HIP_LOW_FRAC = 0.25
ARM_HIGH_SQUAT_BONUS = 0.8
ARM_LOW_DEADLIFT_BONUS = 0.8
ARM_HIGH_DEADLIFT_CLAMP = 0.15
ARM_LOW_SQUAT_CLAMP = 0.15

# ---------------------------------------------------------------------------
# Heurísticas para peso muerto
# ---------------------------------------------------------------------------

DEADLIFT_TORSO_TILT_MIN_DEG = 35.0
DEADLIFT_WRIST_HIP_DIFF_MIN_NORM = 0.20
DEADLIFT_ELBOW_MIN_DEG = 160.0
DEADLIFT_KNEE_BOTTOM_MIN_DEG = 120.0
DEADLIFT_KNEE_FORWARD_MAX_NORM = 0.03
DEADLIFT_BAR_ANKLE_MAX_NORM = 0.08
DEADLIFT_HIP_ROM_MIN_DEG = 30.0
DEADLIFT_BAR_RANGE_MIN_NORM = 0.10
DEADLIFT_BAR_HORIZONTAL_STD_MAX = 0.06
DEADLIFT_BENCH_PENALTY_WEIGHT = 1.5
DEADLIFT_BAR_SHOULDER_MIN_NORM = 0.25
DEADLIFT_BAR_SHOULDER_PENALTY_WEIGHT = 0.9
DEADLIFT_BAR_ABOVE_HIP_MAX_NORM = 0.05
DEADLIFT_ARM_CLAMP = 0.2
DEADLIFT_ARM_HIGH_CLAMP = 0.1

DEADLIFT_TORSO_WEIGHT = 0.9
DEADLIFT_WRIST_HIP_WEIGHT = 0.8
DEADLIFT_ELBOW_WEIGHT = 0.5
DEADLIFT_KNEE_PENALTY_WEIGHT = 0.7
DEADLIFT_BAR_ANKLE_WEIGHT = 0.8
DEADLIFT_BAR_HORIZONTAL_WEIGHT = 0.4
DEADLIFT_BAR_RANGE_WEIGHT = 0.6
DEADLIFT_ROM_WEIGHT = 0.4
DEADLIFT_LOW_MOVEMENT_CAP = 1.2
DEADLIFT_SQUAT_PENALTY_WEIGHT = 0.6
DEADLIFT_SQUAT_BAR_CLAMP = 0.55

DEADLIFT_VETO_SCORE_CLAMP = 0.8
DEADLIFT_VETO_MOVEMENT_MIN = 0.08

# Viabilidad de barra alta (sentadilla) vs barra baja (peso muerto)
SQUAT_BAR_SHOULDER_MAX_NORM = 0.22
SQUAT_BAR_HIGH_BONUS_WEIGHT = 0.7

# Umbrales agregados de brazos/barra
ARM_ABOVE_HIP_THRESH = 0.12
BAR_NEAR_SHOULDER_THRESH = 0.16
BAR_PROXY_MIN_FINITE_RATIO = 0.35
SIDE_VISIBILITY_SCORE_MARGIN = 0.08

# ---------------------------------------------------------------------------
# Heurísticas para la clasificación de la vista de cámara
# ---------------------------------------------------------------------------

VIEW_FRONT_WIDTH_THRESHOLD = 0.55
VIEW_WIDTH_STD_THRESHOLD = 0.12
YAW_FRONT_MAX_DEG = 20.0
YAW_SIDE_MIN_DEG = 25.0
Z_DELTA_FRONT_MAX = 0.08
SIDE_WIDTH_MAX = 0.50
ANKLE_FRONT_WIDTH_THRESHOLD = 0.50
ANKLE_SIDE_WIDTH_MAX = 0.40
ANKLE_WIDTH_STD_THRESHOLD = 0.12

VIEW_FRONT_YAW_WEIGHT = 1.0
VIEW_FRONT_Z_WEIGHT = 0.6
VIEW_FRONT_WIDTH_WEIGHT = 0.8
VIEW_FRONT_WIDTH_STD_WEIGHT = 0.5
VIEW_FRONT_ANKLE_WIDTH_WEIGHT = 0.6
VIEW_FRONT_ANKLE_STD_WEIGHT = 0.4

VIEW_SIDE_YAW_WEIGHT = 1.0
VIEW_SIDE_Z_WEIGHT = 0.7
VIEW_SIDE_WIDTH_WEIGHT = 0.8
VIEW_SIDE_WIDTH_STD_WEIGHT = 0.5
VIEW_SIDE_ANKLE_WIDTH_WEIGHT = 0.6
VIEW_SIDE_ANKLE_STD_WEIGHT = 0.5

VIEW_SCORE_MIN = 0.8
VIEW_SCORE_MARGIN = 0.3
VIEW_SCORE_PER_EVIDENCE_THRESHOLD = 0.22
VIEW_MARGIN_PER_EVIDENCE_THRESHOLD = 0.10
VIEW_FRONT_FALLBACK_YAW_DEG = 24.0
VIEW_SIDE_FALLBACK_YAW_DEG = 27.0
VIEW_STRONG_CONTRADICTION_YAW_DEG = 35.0
VIEW_TH_LO = 0.42
VIEW_TH_HI = 0.58
VIEW_MIN_RELIABLE_FRAMES = 4
RELIABILITY_VIS_THRESHOLD = 0.55
W_ZDELTA = 0.5
W_WIDTHNORM = 0.35
W_VISDELTA = 0.15
EPS = 1e-6

# ---------------------------------------------------------------------------
# Gestión de la confianza
# ---------------------------------------------------------------------------

CLASSIFICATION_MARGIN = 0.12
MIN_CONFIDENCE_SCORE = 0.45

# ---------------------------------------------------------------------------
# Miscelánea
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    "knee_angle_left",
    "knee_angle_right",
    "hip_angle_left",
    "hip_angle_right",
    "elbow_angle_left",
    "elbow_angle_right",
    "torso_tilt_deg",
    "torso_length",
    "torso_length_world",
    "wrist_left_x",
    "wrist_left_y",
    "wrist_right_x",
    "wrist_right_y",
    "shoulder_width_norm",
    "shoulder_yaw_deg",
    "shoulder_z_delta_abs",
    "ankle_width_norm",
    "shoulder_left_y",
    "shoulder_right_y",
    "hip_left_y",
    "hip_right_y",
    "knee_left_x",
    "knee_left_y",
    "knee_right_x",
    "knee_right_y",
    "ankle_left_x",
    "ankle_left_y",
    "ankle_right_x",
    "ankle_right_y",
    "elbow_left_y",
    "elbow_right_y",
)
