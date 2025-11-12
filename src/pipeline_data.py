"""Estructuras de datos compartidas en toda la *pipeline* de análisis.

El módulo centraliza los objetos que permiten intercambiar información entre las
distintas etapas (decodificación de video, estimación de pose, detección y
visualización). Al documentar cada clase se busca dejar claro qué pieza del
proceso la consume y qué atributos expone para otros componentes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    import pandas as pd

from src import config
from src.core.types import ExerciseType, ViewType


@dataclass
class OutputPaths:
    """Rutas finales donde se almacenarán los artefactos de una ejecución.

    Las *pipelines* y las interfaces gráficas utilizan esta estructura para
    conocer el directorio base y las subcarpetas destinadas a los conteos y a
    las poses sin necesidad de recalcular las rutas absolutas en cada paso."""

    base_dir: Path
    counts_dir: Path
    poses_dir: Path
    session_dir: Path


@dataclass
class RunStats:
    """Estadísticas de ejecución que se comparten con las interfaces de usuario.

    Los atributos resumen tanto la configuración efectiva como los valores
    calculados durante el análisis (FPS, ángulos detectados, advertencias,
    tiempos por etapa, etc.), lo que permite mostrar diagnósticos o depurar
    problemas sin volver a procesar el video."""

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
    """Resultado final de la *pipeline*.

    Contiene el conteo de repeticiones, las métricas calculadas y una copia de
    la configuración utilizada para que la capa de presentación pueda
    persistirlos o renderizarlos sin depender de otros módulos."""

    repetitions: int
    metrics: Optional["pd.DataFrame"]
    stats: RunStats
    config_used: config.Config
    debug_video_path: Optional[Path] = None
    overlay_video_path: Optional[Path] = None
    effective_config_path: Optional[Path] = None

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Convertir el reporte a un diccionario compatible con interfaces previas.

        La aplicación original consumía estructuras dinámicas; este método
        traduce las *dataclasses* a dicho formato para mantener compatibilidad
        sin duplicar lógica."""
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
            "stats": stats_dict,
            "config_sha1": self.stats.config_sha1,
            "warnings": list(self.stats.warnings),
            "config_path": str(self.stats.config_path) if self.stats.config_path else None,
            "effective_config_path": str(self.effective_config_path)
            if self.effective_config_path
            else None,
        }
        return legacy
