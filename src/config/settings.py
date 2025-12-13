"""Parámetros por defecto y utilidades de configuración para el pipeline y la GUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path


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
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_HEIGHT = 256
DETECTION_SAMPLE_FPS = 4.0
DEFAULT_PREVIEW_FPS = 10.0
# Bajamos el umbral de visibilidad para no descartar landmarks útiles en vistas
# laterales o con oclusiones parciales.
DEFAULT_LANDMARK_MIN_VISIBILITY = 0.2

# --- PARÁMETROS DE CONTEO (LEGADO) ---
SQUAT_HIGH_THRESH = 160.0
SQUAT_LOW_THRESH = 100.0
PEAK_PROMINENCE = 10  # Prominencia usada por el detector de picos
PEAK_DISTANCE = 15    # Distancia mínima en fotogramas entre repeticiones

# --- VALORES PREDETERMINADOS PARA LA UI ---
DEFAULT_USE_CROP = True
DEFAULT_GENERATE_VIDEO = True
DEFAULT_DEBUG_MODE = True

# --- PROTECCIONES PARA MEDIOS PESADOS ---
OVERLAY_MAX_LONG_SIDE = 1280          # píxeles (p.ej. 720p/1080p-lite)
OVERLAY_DISABLE_OVER_BYTES = 40 * 1024 * 1024  # 40 MB
PREVIEW_DISABLE_OVER_MP = 2.5         # desactiva preview si megapíxeles > 2.5
PREVIEW_MAX_FPS_HEAVY = 5.0           # reduce preview FPS cuando es pesado
OVERLAY_FPS_CAP = 15.0                # límite superior para FPS del overlay
