"""Constantes globales de la aplicación, extensiones y rutas de vídeo."""
from pathlib import Path

# --- CONFIGURACIÓN GENERAL ---
APP_NAME = "FIT CONTROL"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}

# --- RUTAS DE ARCHIVOS ---
# NOTA: usamos ``parents[2]`` porque este archivo vive en ``src/config/``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# --- CONSTANTES DEL PIPELINE ---
# Apostamos por umbrales ligeramente más permisivos para maximizar la detección
# en vídeos con iluminación pobre u oclusiones parciales sin sacrificar la
# estabilidad de MediaPipe.
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# --- FILTRADO Y SUAVIZADO DE ANÁLISIS ---
ANALYSIS_MAX_GAP_FRAMES = 3
ANALYSIS_SMOOTH_METHOD = "savgol"
ANALYSIS_SAVGOL_WINDOW_SEC = 0.25
ANALYSIS_SAVGOL_POLYORDER = 2
ANALYSIS_DERIV_SMOOTH = False
