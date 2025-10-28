"""
Dataclass models for pipeline configuration.
"""
from __future__ import annotations
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import hashlib
import json

# Import defaults from our new settings and constants files
from .constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_COUNTS_DIR,
    DEFAULT_POSES_DIR
)
from .settings import (
    DEFAULT_USE_CROP,
    DEFAULT_TARGET_WIDTH,
    DEFAULT_TARGET_HEIGHT,
    PEAK_PROMINENCE,
    SQUAT_LOW_THRESH,
    SQUAT_HIGH_THRESH,
    DEFAULT_GENERATE_VIDEO,
    DEFAULT_DEBUG_MODE,
    DEFAULT_PREVIEW_FPS,
    DEFAULT_LANDMARK_MIN_VISIBILITY,
)


@dataclass
class PoseConfig:
    """Pose estimation and preprocessing toggles."""
    rotate: Optional[int] = None
    use_crop: bool = DEFAULT_USE_CROP
    target_width: int = DEFAULT_TARGET_WIDTH
    target_height: int = DEFAULT_TARGET_HEIGHT


@dataclass
class VideoConfig:
    """Video decoding and sampling configuration."""
    target_fps: Optional[float] = 10.0
    min_frames: int = 15
    min_fps: float = 5.0
    manual_sample_rate: Optional[int] = None
    detection_sample_fps: Optional[float] = None


@dataclass
class CountingConfig:
    """Parameters used for repetition counting."""
    exercise: str = "squat"
    primary_angle: str = "left_knee"
    min_prominence: float = float(PEAK_PROMINENCE)
    min_distance_sec: float = 0.5
    refractory_sec: float = 0.4
    min_angle_excursion_deg: float = 15.0


@dataclass
class FaultConfig:
    """Thresholds for fault detection / squat depth evaluation."""
    low_thresh: float = SQUAT_LOW_THRESH
    high_thresh: float = SQUAT_HIGH_THRESH


@dataclass
class DebugConfig:
    """Debug and diagnostics toggles."""
    generate_debug_video: bool = DEFAULT_GENERATE_VIDEO
    debug_mode: bool = DEFAULT_DEBUG_MODE
    preview_fps: float = DEFAULT_PREVIEW_FPS
    min_visibility: float = DEFAULT_LANDMARK_MIN_VISIBILITY


@dataclass
class OutputConfig:
    """Filesystem layout used to persist artefacts."""
    base_dir: Path = DEFAULT_OUTPUT_DIR
    counts_dir: Path = DEFAULT_COUNTS_DIR
    poses_dir: Path = DEFAULT_POSES_DIR


@dataclass
class Config:
    """High level configuration object consumed by the pipeline."""
    pose: PoseConfig = field(default_factory=PoseConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    faults: FaultConfig = field(default_factory=FaultConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def copy(self) -> "Config":
        """Return a deep copy of the configuration object."""
        return copy.deepcopy(self)

    # --- Serialisation helpers -------------------------------------------------
    def _to_dict(self, convert_paths: bool = False) -> Dict[str, Any]:
        return _dataclass_to_dict(self, convert_paths=convert_paths)

    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a Python dictionary."""
        return self._to_dict(convert_paths=False)

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return self._to_dict(convert_paths=True)

    # --- Fingerprint -----------------------------------------------------------
    def fingerprint(self) -> str:
        """Return a SHA1 hash of the parameters relevant to counting/faults/pose."""
        payload = {
            "pose": _dataclass_to_dict(self.pose, convert_paths=True),
            "counting": _dataclass_to_dict(self.counting, convert_paths=True),
            "faults": _dataclass_to_dict(self.faults, convert_paths=True),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()


# --- Internal utilities -------------------------------------------------------

def _dataclass_to_dict(obj: Any, *, convert_paths: bool = False) -> Any:
    """Recursively convert dataclasses (and nested objects) to dictionaries."""
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
    """Recursively update ``instance`` with ``updates`` respecting dataclass boundaries."""
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)
    return instance
