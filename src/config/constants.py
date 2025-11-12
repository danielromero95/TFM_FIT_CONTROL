"""Constantes globales de la aplicación, extensiones y rutas de vídeo."""
from pathlib import Path

# --- CONFIGURACIÓN GENERAL ---
APP_NAME = "Gym Performance Analyzer"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}

# --- RUTAS DE ARCHIVOS ---
# NOTA: usamos ``parents[2]`` porque este archivo vive en ``src/config/``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_COUNTS_DIR = DEFAULT_OUTPUT_DIR / "counts"
DEFAULT_POSES_DIR = DEFAULT_OUTPUT_DIR / "poses"

# --- CONSTANTES DEL PIPELINE ---
MIN_DETECTION_CONFIDENCE = 0.5
