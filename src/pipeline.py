"""Data structures shared across the analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src import config
from src.core.types import ExerciseType, ViewType


@dataclass
class OutputPaths:
    """Resolved output directories for a run."""

    base_dir: Path
    counts_dir: Path
    poses_dir: Path
    session_dir: Path


@dataclass
class RunStats:
    """Execution statistics shared with the UIs."""

    config_sha1: str
    fps_original: float
    fps_effective: float
    frames: int
    exercise_selected: Optional[ExerciseType | str]
    exercise_detected: ExerciseType | str
    view_detected: ViewType | str
    detection_confidence: float
    primary_angle: Optional[str]
    angle_range_deg: float
    min_prominence: float
    min_distance_sec: float
    refractory_sec: float
    warnings: list[str] = field(default_factory=list)
    skip_reason: Optional[str] = None
    config_path: Optional[Path] = None
    # Stage timings (milliseconds). Optional for backward compatibility.
    t_extract_ms: Optional[float] = None
    t_pose_ms: Optional[float] = None
    t_filter_ms: Optional[float] = None
    t_metrics_ms: Optional[float] = None
    t_count_ms: Optional[float] = None
    t_total_ms: Optional[float] = None


@dataclass
class Report:
    """Pipeline outcome."""

    repetitions: int
    metrics: Optional[pd.DataFrame]
    debug_video_path: Optional[Path]
    stats: RunStats
    config_used: config.Config

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Preserve the dictionary-based API used by the existing front-ends."""
        stats_dict = asdict(self.stats)
        for key in ("exercise_selected", "exercise_detected", "view_detected"):
            val = stats_dict.get(key)
            if isinstance(val, Enum):
                stats_dict[key] = val.value

        legacy: Dict[str, Any] = {
            "repetition_count": self.repetitions,
            "metrics_dataframe": self.metrics,
            "debug_video_path": str(self.debug_video_path) if self.debug_video_path else None,
            "stats": stats_dict,
            "config_sha1": self.stats.config_sha1,
            "warnings": list(self.stats.warnings),
            "config_path": str(self.stats.config_path) if self.stats.config_path else None,
        }
        return legacy
