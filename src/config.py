# src/config.py
"""
Archivo de Configuración Centralizado
"""

# --- CONFIGURACIÓN GENERAL ---
APP_NAME = "Gym Performance Analyzer"
ORGANIZATION_NAME = "GymPerformance"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}

# --- PARÁMETROS DEL PIPELINE ---
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_HEIGHT = 256

# --- PARÁMETROS DE CONTEO ---
SQUAT_HIGH_THRESH = 160.0
SQUAT_LOW_THRESH = 100.0
PEAK_PROMINENCE = 10  # Prominencia para el detector de picos
PEAK_DISTANCE = 15    # Distancia mínima en frames entre repeticiones

# --- CONFIGURACIÓN DE VISUALIZACIÓN ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (12, 14), 
    (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24), (23, 24), 
    (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30), 
    (28, 32), (29, 31), (30, 32)
]
LANDMARK_COLOR = (0, 255, 0)  # Verde
CONNECTION_COLOR = (0, 0, 255) # Rojo

# --- VALORES POR DEFECTO DE LA GUI ---
DEFAULT_SAMPLE_RATE = 3
DEFAULT_ROTATION = "0"
DEFAULT_USE_CROP = True
DEFAULT_GENERATE_VIDEO = True
DEFAULT_DEBUG_MODE = True
DEFAULT_DARK_MODE = False