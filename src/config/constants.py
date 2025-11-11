"""
Global application constants, video extensions, and filesystem paths.
"""
from pathlib import Path

# --- GENERAL CONFIGURATION ---
APP_NAME = "Gym Performance Analyzer"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}

# --- FILE PATHS ---
# NOTE: We use parents[2] here because this file is in src/config/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_COUNTS_DIR = DEFAULT_OUTPUT_DIR / "counts"
DEFAULT_POSES_DIR = DEFAULT_OUTPUT_DIR / "poses"

# --- PIPELINE CONSTANTS ---
MIN_DETECTION_CONFIDENCE = 0.5
