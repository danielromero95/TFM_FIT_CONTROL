"""Parámetros por defecto y utilidades de configuración para el pipeline y la GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .constants import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE


def configure_environment() -> None:
    """Ajusta variables de entorno y rutas necesarias antes de lanzar la aplicación."""

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["GLOG_minloglevel"] = "2"

    try:
        from absl import logging as absl_logging  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        # Forzamos a ``absl`` a emitir solo errores para no saturar la consola.
        absl_logging.set_verbosity(absl_logging.ERROR)

    if sys.platform.startswith("win"):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            dll_dir = Path(conda_prefix) / "Library" / "bin"
            if dll_dir.exists():
                # Añadimos las DLL de Conda al ``PATH`` para evitar errores de carga.
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


# --- PARÁMETROS DEL PIPELINE ---
MODEL_COMPLEXITY = 2
POSE_MODEL_COMPLEXITY = MODEL_COMPLEXITY
POSE_ENABLE_SEGMENTATION = False
POSE_SMOOTH_SEGMENTATION = False
POSE_SMOOTH_LANDMARKS = True
POSE_STATIC_IMAGE_MODE = False
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_HEIGHT = 256
DETECTION_SAMPLE_FPS = 4.0
DEFAULT_PREVIEW_FPS = 10.0
DEFAULT_LANDMARK_MIN_VISIBILITY = 0.5

# --- PARÁMETROS DE CONTEO (LEGADO) ---
SQUAT_HIGH_THRESH = 160.0
SQUAT_LOW_THRESH = 100.0
PEAK_PROMINENCE = 10  # Prominencia usada por el detector de picos
PEAK_DISTANCE = 15    # Distancia mínima en fotogramas entre repeticiones

# --- VALORES PREDETERMINADOS PARA LA UI ---
DEFAULT_USE_CROP = True
DEFAULT_GENERATE_VIDEO = True
DEFAULT_DEBUG_MODE = True
DEFAULT_WARMUP_SECONDS = 0.5

# --- PROTECCIONES PARA MEDIOS PESADOS ---
OVERLAY_MAX_LONG_SIDE = 1280          # píxeles (p.ej. 720p/1080p-lite)
OVERLAY_DISABLE_OVER_BYTES = 40 * 1024 * 1024  # 40 MB
PREVIEW_DISABLE_OVER_MP = 2.5         # desactiva preview si megapíxeles > 2.5
PREVIEW_MAX_FPS_HEAVY = 5.0           # reduce preview FPS cuando es pesado
OVERLAY_FPS_CAP = 15.0                # límite superior para FPS del overlay


def build_pose_kwargs(
    *,
    static_image_mode: bool | None = None,
    model_complexity: int | None = None,
    min_detection_confidence: float | None = None,
    min_tracking_confidence: float | None = None,
    smooth_landmarks: bool | None = None,
    enable_segmentation: bool | None = None,
) -> dict[str, object]:
    """Configuración estándar para el grafo ``Pose`` de MediaPipe.

    Centralizar estos parámetros garantiza que todos los puntos del pipeline
    utilicen el mismo modelo "heavy" (``model_complexity=2``) y suavizado de
    *landmarks*, priorizando la calidad sobre la velocidad para vídeos
    desafiantes.
    """

    return {
        "static_image_mode": POSE_STATIC_IMAGE_MODE if static_image_mode is None else static_image_mode,
        "model_complexity": POSE_MODEL_COMPLEXITY if model_complexity is None else model_complexity,
        "smooth_landmarks": POSE_SMOOTH_LANDMARKS if smooth_landmarks is None else bool(smooth_landmarks),
        "enable_segmentation": POSE_ENABLE_SEGMENTATION
        if enable_segmentation is None
        else bool(enable_segmentation),
        "smooth_segmentation": POSE_SMOOTH_SEGMENTATION,
        "min_detection_confidence": (
            MIN_DETECTION_CONFIDENCE if min_detection_confidence is None else float(min_detection_confidence)
        ),
        "min_tracking_confidence": (
            MIN_TRACKING_CONFIDENCE if min_tracking_confidence is None else float(min_tracking_confidence)
        ),
    }
