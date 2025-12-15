"""Estructuras de datos compartidas por todo el pipeline de análisis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - importado únicamente para tipado
    import pandas as pd

from src import config
from src.core.types import ExerciseType, ViewType


@dataclass
class OutputPaths:
    """Rutas de salida calculadas para una ejecución del pipeline."""

    base_dir: Path
    counts_dir: Path
    poses_dir: Path
    session_dir: Path


@dataclass
class RunStats:
    """Estadísticas de ejecución que se comparten con las interfaces de usuario."""

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
    # Tiempos por etapa (milisegundos). Opcional por compatibilidad retroactiva.
    t_extract_ms: Optional[float] = None
    t_pose_ms: Optional[float] = None
    t_filter_ms: Optional[float] = None
    t_metrics_ms: Optional[float] = None
    t_count_ms: Optional[float] = None
    t_total_ms: Optional[float] = None


@dataclass
class Report:
    """Resultado completo de una ejecución del pipeline.

    Attributes:
        debug_summary: contenedor opcional y serializable con información de
            depuración, incluyendo etiqueta/vista detectada, parámetros de
            conteo ajustados y métricas de calidad básicas.
    """

    repetitions: int
    metrics: Optional["pd.DataFrame"]
    stats: RunStats
    config_used: config.Config
    debug_video_path: Optional[Path] = None
    overlay_video_path: Optional[Path] = None
    overlay_video_stream_path: Optional[Path] = None
    effective_config_path: Optional[Path] = None
    debug_summary: Dict[str, Any] | None = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convierte el reporte en el diccionario esperado por los clientes heredados."""
        stats_dict = asdict(self.stats)
        for key in ("exercise_selected", "exercise_detected", "view_detected"):
            val = stats_dict.get(key)
            if isinstance(val, Enum):
                stats_dict[key] = val.value

        legacy: Dict[str, Any] = {
            "repetition_count": self.repetitions,
            "metrics_dataframe": self.metrics,
            "debug_video_path": str(self.debug_video_path) if self.debug_video_path else None,
            "overlay_video_path": str(self.overlay_video_path)
            if self.overlay_video_path
            else None,
            "overlay_video_stream_path": str(self.overlay_video_stream_path)
            if self.overlay_video_stream_path
            else None,
            "stats": stats_dict,
            "config_sha1": self.stats.config_sha1,
            "warnings": list(self.stats.warnings),
            "config_path": str(self.stats.config_path) if self.stats.config_path else None,
            "effective_config_path": str(self.effective_config_path)
            if self.effective_config_path
            else None,
        }
        # Se devuelve un ``dict`` explícito para mantener compatibilidad con UIs antiguas.
        return legacy
