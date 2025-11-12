"""Conteo de repeticiones mediante detección de valles y consolidación por periodo refractario."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src import config

logger = logging.getLogger(__name__)


@dataclass
class CountingDebugInfo:
    """Carga de depuración que devuelve el contador de repeticiones."""
    valley_indices: List[int]
    prominences: List[float]


def _count_reps_by_valleys(
    angle_sequence: Sequence[float],
    *,
    prominence: float,
    distance: int,
) -> Tuple[int, CountingDebugInfo]:
    """Detecta valles en la secuencia de ángulos buscando picos en la señal invertida."""
    if not angle_sequence:
        return 0, CountingDebugInfo([], [])

    inverted = -np.asarray(angle_sequence, dtype=float)
    valleys, properties = find_peaks(
        inverted,
        prominence=float(prominence),
        distance=int(max(1, distance)),
    )

    prominences = properties.get("prominences", np.array([], dtype=float))
    debug = CountingDebugInfo(
        valley_indices=[int(i) for i in valleys.tolist()],
        prominences=[float(p) for p in prominences.tolist()],
    )
    logger.debug("Valley detection found %d candidates.", len(debug.valley_indices))
    return len(debug.valley_indices), debug


def _apply_refractory_filter(
    indices: List[int],
    prominences: List[float],
    refractory_frames: int,
) -> Tuple[List[int], List[float]]:
    """Agrupa valles separados por menos de ``refractory_frames`` y conserva el más prominente de cada grupo."""
    if refractory_frames <= 0 or len(indices) <= 1:
        return indices, prominences

    # Trabajamos con arreglos de NumPy; se espera que los índices lleguen en orden ascendente.
    idx = np.asarray(indices, dtype=np.int64)
    prom = np.asarray(prominences, dtype=np.float64)

    # Identifica el inicio de cada grupo cuando la separación es >= ``refractory_frames``
    diffs = np.diff(idx)
    starts = np.r_[0, np.flatnonzero(diffs >= int(refractory_frames)) + 1]
    ends = np.r_[starts[1:], idx.size]

    kept_idx: list[int] = []
    kept_prom: list[float] = []

    # Itera por grupo (el número de grupos suele ser mucho menor que el número de puntos)
    for s, e in zip(starts, ends):
        # ``argmax`` devuelve la primera ocurrencia en empates → índice determinista más temprano del grupo
        local = s + int(np.argmax(prom[s:e]))
        kept_idx.append(int(idx[local]))
        kept_prom.append(float(prom[local]))

    return kept_idx, kept_prom


def count_repetitions_with_config(
    df_metrics: pd.DataFrame,
    counting_cfg: config.CountingConfig,
    fps: float,
    *,
    overrides: dict[str, float] | None = None,
) -> Tuple[int, CountingDebugInfo]:
    """
    Cuenta repeticiones empleando los parámetros definidos en la configuración.

    Args:
        df_metrics: DataFrame con las métricas biomecánicas; debe contener ``counting_cfg.primary_angle``.
        counting_cfg: instancia de ``src.config.models.CountingConfig`` con los umbrales vigentes.
        fps: fotogramas por segundo efectivos de la serie sobre la que se cuenta.

    Returns:
        Tupla ``(repetition_count, CountingDebugInfo)`` con el total y la información de depuración.

    Keyword Args:
        overrides: diccionario opcional con claves ``min_prominence``, ``min_distance_sec`` y
            ``refractory_sec`` para ajustar temporalmente los umbrales sin mutar la configuración original.
    """
    angle_column = counting_cfg.primary_angle

    if df_metrics.empty or angle_column not in df_metrics.columns:
        logger.warning(
            "Column '%s' was not found or the DataFrame is empty. Returning 0 repetitions.",
            angle_column,
        )
        return 0, CountingDebugInfo([], [])

    # Rellena hacia adelante y hacia atrás para mitigar periodos breves de NaN y elimina los restantes
    angles = df_metrics[angle_column].ffill().bfill().dropna().tolist()
    if not angles:
        return 0, CountingDebugInfo([], [])

    overrides = overrides or {}

    fps_safe = float(fps) if fps and fps > 0 else 1.0
    min_distance_sec = float(
        overrides.get("min_distance_sec", counting_cfg.min_distance_sec)
    )
    prominence_thr = float(overrides.get("min_prominence", counting_cfg.min_prominence))

    distance_frames = max(1, int(round(min_distance_sec * fps_safe)))

    reps, debug = _count_reps_by_valleys(
        angle_sequence=angles,
        prominence=prominence_thr,
        distance=distance_frames,
    )

    refractory_sec = float(overrides.get("refractory_sec", counting_cfg.refractory_sec))
    refractory_frames = max(0, int(round(refractory_sec * fps_safe)))
    if refractory_frames > 0 and debug.valley_indices:
        filtered_idx, filtered_prom = _apply_refractory_filter(
            debug.valley_indices, debug.prominences, refractory_frames
        )
        debug = CountingDebugInfo(valley_indices=filtered_idx, prominences=filtered_prom)
        reps = len(filtered_idx)

    logger.debug(
        "Repetition count=%d (distance=%d frames ≈ %.2fs, prominence>=%.3f, refractory=%d frames ≈ %.2fs).",
        reps,
        distance_frames,
        distance_frames / fps_safe,
        prominence_thr,
        refractory_frames,
        refractory_frames / fps_safe,
    )
    return reps, debug
