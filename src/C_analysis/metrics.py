"""Cálculo de métricas biomecánicas y ajuste automático de parámetros de conteo."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config
from src.B_pose_estimation.pipeline import calculate_metrics_from_sequence, filter_and_interpolate_landmarks
from .repetition_counter import count_repetitions_with_config
from src.core.types import ExerciseType, ViewType, as_exercise, as_view

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoTuneResult:
    """Valores ajustados dinámicamente para los parámetros de conteo."""

    min_prominence: float
    min_distance_sec: float
    refractory_sec: float
    cadence_period_sec: float | None = None
    iqr_deg: float | None = None
    multipliers: Dict[str, float] = field(default_factory=dict)


def filter_landmarks(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, object, pd.Series]:
    """Aplicar filtrado e interpolación a los *landmarks* detectados."""

    sequence, crops, quality = filter_and_interpolate_landmarks(df_raw)
    return sequence, crops, pd.Series(quality)


def choose_primary_angle(
    exercise: str | ExerciseType,
    view: str | ViewType,
    df: pd.DataFrame,
) -> str | None:
    """Seleccionar de forma heurística el ángulo principal según ejercicio y vista."""

    ex = as_exercise(exercise).value
    vw = as_view(view).value
    candidates_map = {
        "squat": ["left_knee", "right_knee"],
        "bench_press": ["left_elbow", "right_elbow"],
        "deadlift": ["left_hip", "right_hip"],
    }
    candidates = [c for c in candidates_map.get(ex, []) if c in df.columns]
    if not candidates:
        return None

    def series_for(col: str):
        raw_name = f"raw_{col}"
        base = df[raw_name] if raw_name in df.columns else df[col]
        return pd.to_numeric(base, errors="coerce")

    def quality(col: str) -> tuple[float, float]:
        series = series_for(col)
        finite = series.dropna()
        if finite.empty:
            return (0.0, 0.0)
        coverage = float(finite.size) / float(max(1, series.size))
        rom = float(finite.max() - finite.min())
        return (coverage, rom)

    candidates.sort(key=lambda col: quality(col), reverse=True)
    chosen = candidates[0] if candidates else None
    if chosen:
        logger.info(
            "PRIMARY AUTO-SELECT: exercise=%s view=%s candidates=%s → chosen=%s (coverage,ROM)=%s",
            ex,
            vw,
            candidates,
            chosen,
            quality(chosen),
        )
    return chosen


def _trimmed_rom_deg(df: pd.DataFrame, col: str) -> float:
    """Calcular un rango de movimiento robusto utilizando percentiles 10 y 90."""

    if df is None or not col:
        return 0.0
    raw = f"raw_{col}"
    base = df[raw] if raw in df.columns else df[col] if col in df.columns else None
    if base is None:
        return 0.0
    series = pd.to_numeric(base, errors="coerce").dropna()
    if series.empty:
        return 0.0
    values = series.to_numpy(dtype=float)
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    return max(0.0, p90 - p10)


def _iqr_deg(df: pd.DataFrame, col: str) -> float | None:
    """Obtener el rango intercuartílico del ángulo especificado."""

    if df is None or not col:
        return None
    raw = f"raw_{col}"
    base = df[raw] if raw in df.columns else df[col] if col in df.columns else None
    if base is None:
        return None
    series = pd.to_numeric(base, errors="coerce").dropna()
    if series.empty:
        return None
    values = series.to_numpy(dtype=float)
    q75, q25 = np.percentile(values, [75, 25])
    return max(0.0, float(q75 - q25))


def _simple_peak_indices(values: np.ndarray, min_gap: int) -> list[int]:
    """Detección simple de picos para cuando SciPy no está disponible."""

    indices: list[int] = []
    for idx in range(1, len(values) - 1):
        v_prev, value, v_next = values[idx - 1], values[idx], values[idx + 1]
        if not (np.isfinite(v_prev) and np.isfinite(value) and np.isfinite(v_next)):
            continue
        if value > v_prev and value >= v_next:
            if indices and idx - indices[-1] < min_gap:
                if value > values[indices[-1]]:
                    indices[-1] = idx
            else:
                indices.append(idx)
    return indices


def _estimate_cadence_period(
    df: pd.DataFrame,
    fps: float,
    columns: Optional[list[str]] = None,
) -> float | None:
    """Estimar el periodo de cadencia considerando columnas candidatas."""

    if df is None or fps is None or fps <= 0:
        return None

    if columns is None:
        columns = []
        for base in ("left_knee", "right_knee"):
            raw = f"raw_{base}"
            if raw in df.columns:
                columns.append(raw)
            elif base in df.columns:
                columns.append(base)
    else:
        columns = [col for col in columns if col in df.columns]

    if not columns:
        return None

    diffs: list[float] = []
    min_distance = int(max(1, round(float(fps) * 0.25)))

    try:  # pragma: no cover - SciPy opcional
        from scipy.signal import find_peaks as _find_peaks
    except Exception:  # pragma: no cover - SciPy opcional
        _find_peaks = None

    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        filled = series.interpolate(method="linear", limit_direction="both")
        values = filled.to_numpy(dtype=float)
        if values.size < 3:
            continue

        if _find_peaks is not None:
            peaks, _ = _find_peaks(values, distance=min_distance)
            peak_indices = peaks.tolist()
        else:
            peak_indices = _simple_peak_indices(values, min_distance)

        if len(peak_indices) >= 2:
            diffs.extend(np.diff(peak_indices))

    if not diffs:
        return None

    median_frames = float(np.median(np.asarray(diffs, dtype=float)))
    if not np.isfinite(median_frames) or median_frames <= 0:
        return None
    return median_frames / float(fps)


def auto_counting_params(
    exercise: ExerciseType | str,
    df_metrics: pd.DataFrame,
    primary: str | None,
    fps: float,
    counting_cfg: config.CountingConfig,
    *,
    view: ViewType | str = ViewType.UNKNOWN,
) -> AutoTuneResult:
    """Inferir parámetros de conteo basados en la cadencia y robustez del movimiento."""

    prom_default = float(getattr(counting_cfg, "min_prominence", 0.0) or 0.0)
    dist_default = float(getattr(counting_cfg, "min_distance_sec", 0.5) or 0.5)
    refractory_default = float(getattr(counting_cfg, "refractory_sec", 0.4) or 0.4)

    rom = _trimmed_rom_deg(df_metrics, primary or "")
    prom_candidates = [max(5.0, 0.20 * rom)] if rom > 0 else [5.0]
    iqr = _iqr_deg(df_metrics, primary or "")
    if iqr is not None and iqr > 0:
        prom_candidates.append(max(9.0, 0.35 * iqr))
    prom_candidates.append(prom_default)
    prom = max(prom_candidates)

    cadence_period = _estimate_cadence_period(df_metrics, fps)
    if cadence_period is not None and np.isfinite(cadence_period):
        min_distance = float(np.clip(0.35 * cadence_period, 0.35, 1.2))
        refractory = float(np.clip(0.25 * cadence_period, 0.25, 0.8))
    else:
        if dist_default:
            min_distance = dist_default
        else:
            min_distance = 0.6
        if refractory_default:
            refractory = refractory_default
        else:
            refractory = 0.4

    multipliers: Dict[str, float] = {"prominence": 1.0, "distance": 1.0, "refractory": 1.0}
    view_value = as_view(view).value
    if view_value == "frontal":
        multipliers["prominence"] = 1.1
        multipliers["distance"] = 1.05
        multipliers["refractory"] = 1.1

    prom *= multipliers["prominence"]
    min_distance *= multipliers["distance"]
    refractory *= multipliers["refractory"]

    if not np.isfinite(prom) or prom <= 0:
        prom = max(5.0, prom_default or 5.0)
    prom = float(np.clip(prom, 5.0, 60.0))

    if not np.isfinite(min_distance) or min_distance <= 0:
        min_distance = dist_default
    min_distance = float(np.clip(min_distance, 0.25, 2.0))

    if not np.isfinite(refractory) or refractory <= 0:
        refractory = refractory_default
    refractory = float(np.clip(refractory, 0.20, 1.5))
    if refractory > min_distance:
        refractory = max(0.20, min_distance * 0.9)

    return AutoTuneResult(
        min_prominence=prom,
        min_distance_sec=min_distance,
        refractory_sec=refractory,
        cadence_period_sec=cadence_period,
        iqr_deg=iqr,
        multipliers=multipliers,
    )


def compute_metrics_and_angle(
    df_seq: pd.DataFrame,
    primary_angle: str,
    fps_effective: float,
    *,
    exercise: ExerciseType | str = ExerciseType.UNKNOWN,
    view: ViewType | str = ViewType.UNKNOWN,
    quality_mask: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, float, list[str], Optional[str], Optional[str]]:
    """Calcular métricas biomecánicas y derivar la excursión del ángulo principal."""

    warnings: list[str] = []
    skip_reason: Optional[str] = None

    df_metrics = calculate_metrics_from_sequence(df_seq, fps_effective, quality_mask=quality_mask)
    angle_range = 0.0

    chosen_primary = primary_angle
    if (not chosen_primary) or (chosen_primary == "auto") or (
        chosen_primary not in df_metrics.columns
    ):
        chosen = choose_primary_angle(exercise, view, df_metrics)
        if chosen:
            chosen_primary = chosen

    if chosen_primary and chosen_primary in df_metrics.columns:
        series = df_metrics[chosen_primary].dropna()
        if not series.empty:
            angle_range = float(series.max() - series.min())
        else:
            warnings.append(
                f"No valid values could be obtained for the primary angle '{chosen_primary}'."
            )
            skip_reason = "The primary angle column does not contain valid data."
    else:
        missing_column = chosen_primary or primary_angle
        if missing_column:
            warnings.append(
                f"The primary angle column '{missing_column}' is not present in the metrics."
            )
        else:
            warnings.append("No valid primary angle column could be determined from the metrics.")
        skip_reason = "The primary angle column was not found in the metrics."

    return df_metrics, angle_range, warnings, skip_reason, chosen_primary


def maybe_count_reps(
    df_metrics: pd.DataFrame,
    cfg: config.Config,
    fps_effective: float,
    skip_reason: Optional[str],
    *,
    overrides: Optional[dict[str, float]] = None,
) -> tuple[int, list[str]]:
    """Contar repeticiones a menos que exista una causa para omitir el conteo."""

    if skip_reason is not None:
        return 0, []

    result, debug_info = count_repetitions_with_config(
        df_metrics, cfg.counting, fps_effective, overrides=overrides
    )
    stage_warnings: list[str] = []
    if result == 0 and not debug_info.valley_indices:
        stage_warnings.append("No repetitions were detected with the current parameters.")
    return result, stage_warnings
