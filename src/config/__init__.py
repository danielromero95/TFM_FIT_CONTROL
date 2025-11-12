"""Reexportaciones para mantener compatibilidad con ``from src import config``."""

from __future__ import annotations

# Dataclasses principales de configuración --------------------------------------
try:  # pragma: no cover - salvaguarda por dependencias opcionales
    from .models import (
        Config,
        PoseConfig,
        VideoConfig,
        CountingConfig,
        FaultConfig,
        DebugConfig,
        OutputConfig,
    )
except Exception:  # pragma: no cover
    Config = None  # type: ignore
    PoseConfig = None  # type: ignore
    VideoConfig = None  # type: ignore
    CountingConfig = None  # type: ignore
    FaultConfig = None  # type: ignore
    DebugConfig = None  # type: ignore
    OutputConfig = None  # type: ignore

# Funciones auxiliares de carga --------------------------------------------------
try:  # pragma: no cover - salvaguarda defensiva
    from .utils import load_default, from_yaml
except Exception:  # pragma: no cover
    def load_default(*_args, **_kwargs):  # type: ignore
        raise RuntimeError("config.utils.load_default no está disponible")

    def from_yaml(*_args, **_kwargs):  # type: ignore
        raise RuntimeError("config.utils.from_yaml no está disponible")

# Constantes compartidas ---------------------------------------------------------
try:
    from .constants import (
        APP_NAME,
        PROJECT_ROOT,
        VIDEO_EXTENSIONS,
        MIN_DETECTION_CONFIDENCE,
    )
except Exception:  # pragma: no cover
    APP_NAME = None  # type: ignore
    PROJECT_ROOT = None  # type: ignore
    VIDEO_EXTENSIONS = None  # type: ignore
    MIN_DETECTION_CONFIDENCE = None  # type: ignore

# Utilidades de visualización ----------------------------------------------------
try:  # pragma: no cover - la visualización puede ser opcional en algunas instalaciones
    from .video_landmarks_visualization import (
        POSE_CONNECTIONS,
        LANDMARK_COLOR,
        CONNECTION_COLOR,
        THICKNESS_DEFAULT,
        RADIUS_DEFAULT,
    )
except Exception:  # pragma: no cover
    try:
        from .models import POSE_CONNECTIONS  # type: ignore
    except Exception:  # pragma: no cover
        POSE_CONNECTIONS = None  # type: ignore
    LANDMARK_COLOR = None  # type: ignore
    CONNECTION_COLOR = None  # type: ignore
    THICKNESS_DEFAULT = None  # type: ignore
    RADIUS_DEFAULT = None  # type: ignore

__all__ = [
    # Models
    "Config",
    "PoseConfig",
    "VideoConfig",
    "CountingConfig",
    "FaultConfig",
    "DebugConfig",
    "OutputConfig",

    # Utilities
    "load_default",
    "from_yaml",

    # Constants
    "APP_NAME",
    "PROJECT_ROOT",
    "VIDEO_EXTENSIONS",
    "MIN_DETECTION_CONFIDENCE",

    # Visualization
    "POSE_CONNECTIONS",
    "LANDMARK_COLOR",
    "CONNECTION_COLOR",
    "THICKNESS_DEFAULT",
    "RADIUS_DEFAULT",
]
