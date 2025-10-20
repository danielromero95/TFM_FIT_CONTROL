from __future__ import annotations

"""Centralised configuration helpers and defaults."""

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import copy
import hashlib
import json

try:  # Optional dependency – only needed when loading from YAML files.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is optional at runtime
    yaml = None  # type: ignore

# --- GENERAL CONFIGURATION ---
APP_NAME = "Gym Performance Analyzer"
ORGANIZATION_NAME = "GymPerformance"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv"}

# --- PIPELINE PARAMETERS ---
MODEL_COMPLEXITY = 1
MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_HEIGHT = 256

# --- COUNTING PARAMETERS (legacy) ---
SQUAT_HIGH_THRESH = 160.0
SQUAT_LOW_THRESH = 100.0
PEAK_PROMINENCE = 10  # Prominence used by the peak detector
PEAK_DISTANCE = 15    # Minimum distance in frames between repetitions

# --- VISUALISATION CONFIGURATION ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (12, 14),
    (14, 16), (16, 18), (16, 20), (16, 22), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (24, 26), (26, 28), (28, 30),
    (28, 32), (29, 31), (30, 32)
]
LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (0, 0, 255)  # Red

# --- DEFAULT GUI VALUES ---
DEFAULT_SAMPLE_RATE = 3
DEFAULT_ROTATION = "0"
DEFAULT_USE_CROP = True
DEFAULT_GENERATE_VIDEO = True
DEFAULT_DEBUG_MODE = True
DEFAULT_DARK_MODE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_COUNTS_DIR = DEFAULT_OUTPUT_DIR / "counts"
DEFAULT_POSES_DIR = DEFAULT_OUTPUT_DIR / "poses"


@dataclass
class PoseConfig:
    """Pose estimation and preprocessing toggles.

    ``target_width``/``target_height`` define the single resizing stage applied in the
    pipeline prior to pose estimation. ``extract_and_preprocess_frames`` returns the
    frames at their native resolution so there is a single, well-defined resize point.
    """

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


# --- Public helpers ------------------------------------------------------------

def load_default() -> Config:
    """Return the default configuration used by the Streamlit application."""
    return Config()


def from_yaml(path: str | Path) -> Config:
    """Load a configuration from a YAML file and merge it with defaults."""
    if yaml is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("PyYAML is not available. Install it to load YAML files.")

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = load_default()
    _update_dataclass(cfg, data)
    return cfg


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
