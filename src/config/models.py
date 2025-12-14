"""Modelos ``dataclass`` que describen la configuración del pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import hashlib
import json

# Importa valores por defecto definidos en los módulos de configuración central.
from .constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_COUNTS_DIR,
    DEFAULT_POSES_DIR,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from .settings import (
    DEFAULT_USE_CROP,
    DEFAULT_TARGET_WIDTH,
    DEFAULT_TARGET_HEIGHT,
    MODEL_COMPLEXITY,
    PEAK_PROMINENCE,
    SQUAT_LOW_THRESH,
    SQUAT_HIGH_THRESH,
    DEFAULT_GENERATE_VIDEO,
    DEFAULT_DEBUG_MODE,
    DEFAULT_PREVIEW_FPS,
    DEFAULT_LANDMARK_MIN_VISIBILITY,
    POSE_ENABLE_SEGMENTATION,
    POSE_SMOOTH_LANDMARKS,
    OVERLAY_MAX_LONG_SIDE,
    OVERLAY_DISABLE_OVER_BYTES,
    PREVIEW_DISABLE_OVER_MP,
    PREVIEW_MAX_FPS_HEAVY,
    OVERLAY_FPS_CAP,
)


@dataclass
class PoseConfig:
    """Interruptores para el preprocesado y la estimación de pose."""
    rotate: Optional[int] = None
    use_crop: bool = DEFAULT_USE_CROP
    target_width: int = DEFAULT_TARGET_WIDTH
    target_height: int = DEFAULT_TARGET_HEIGHT
    model_complexity: int = MODEL_COMPLEXITY
    min_detection_confidence: float = float(MIN_DETECTION_CONFIDENCE)
    min_tracking_confidence: float = float(MIN_TRACKING_CONFIDENCE)
    smooth_landmarks: bool = POSE_SMOOTH_LANDMARKS
    enable_segmentation: bool = POSE_ENABLE_SEGMENTATION


@dataclass
class VideoConfig:
    """Parámetros de decodificación y muestreo del vídeo."""
    target_fps: Optional[float] = 10.0
    min_frames: int = 15
    min_fps: float = 5.0
    manual_sample_rate: Optional[int] = None
    detection_sample_fps: Optional[float] = None


@dataclass
class CountingConfig:
    """Parámetros empleados para el conteo de repeticiones."""
    exercise: str = "squat"
    primary_angle: str = "left_knee"
    min_prominence: float = float(PEAK_PROMINENCE)
    min_distance_sec: float = 0.5
    refractory_sec: float = 0.4
    min_angle_excursion_deg: float = 15.0


@dataclass
class FaultConfig:
    """Umbrales para detectar fallos y evaluar la profundidad de la sentadilla."""
    low_thresh: float = SQUAT_LOW_THRESH
    high_thresh: float = SQUAT_HIGH_THRESH


@dataclass
class DebugConfig:
    """Ajustes de depuración y diagnósticos del pipeline."""
    generate_debug_video: bool = DEFAULT_GENERATE_VIDEO
    debug_mode: bool = DEFAULT_DEBUG_MODE
    preview_fps: float = DEFAULT_PREVIEW_FPS
    min_visibility: float = DEFAULT_LANDMARK_MIN_VISIBILITY
    overlay_max_long_side: int = OVERLAY_MAX_LONG_SIDE
    overlay_disable_over_bytes: int = OVERLAY_DISABLE_OVER_BYTES
    preview_disable_over_mp: float = PREVIEW_DISABLE_OVER_MP
    preview_fps_heavy: float = PREVIEW_MAX_FPS_HEAVY
    overlay_fps_cap: float = OVERLAY_FPS_CAP


@dataclass
class OutputConfig:
    """Estructura de carpetas donde se guardan los artefactos generados."""
    base_dir: Path = DEFAULT_OUTPUT_DIR
    counts_dir: Path = DEFAULT_COUNTS_DIR
    poses_dir: Path = DEFAULT_POSES_DIR


@dataclass
class Config:
    """Configuración de alto nivel consumida por el pipeline completo."""
    pose: PoseConfig = field(default_factory=PoseConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    faults: FaultConfig = field(default_factory=FaultConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def copy(self) -> "Config":
        """Devuelve una copia profunda del objeto de configuración."""
        return copy.deepcopy(self)

    # --- Serialisation helpers -------------------------------------------------
    def _to_dict(self, convert_paths: bool = False) -> Dict[str, Any]:
        return _dataclass_to_dict(self, convert_paths=convert_paths)

    def to_dict(self) -> Dict[str, Any]:
        """Entrega la configuración como diccionario de Python."""
        return self._to_dict(convert_paths=False)

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Genera una representación serializable en JSON."""
        return self._to_dict(convert_paths=True)

    # --- Fingerprint -----------------------------------------------------------
    def fingerprint(self) -> str:
        """Calcula un hash SHA1 de los parámetros críticos del conteo y la pose."""
        payload = {
            "pose": _dataclass_to_dict(self.pose, convert_paths=True),
            "counting": _dataclass_to_dict(self.counting, convert_paths=True),
            "faults": _dataclass_to_dict(self.faults, convert_paths=True),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()


# --- Internal utilities -------------------------------------------------------

def _dataclass_to_dict(obj: Any, *, convert_paths: bool = False) -> Any:
    """Convierte recursivamente ``dataclasses`` (y anidados) en diccionarios."""
    if is_dataclass(obj):
        return {key: _dataclass_to_dict(value, convert_paths=convert_paths) for key, value in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {key: _dataclass_to_dict(value, convert_paths=convert_paths) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(value, convert_paths=convert_paths) for value in obj]
    if isinstance(obj, Path):
        return str(obj) if convert_paths else obj
    return obj


def _update_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    """Actualiza recursivamente ``instance`` respetando los límites de cada ``dataclass``."""
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance
