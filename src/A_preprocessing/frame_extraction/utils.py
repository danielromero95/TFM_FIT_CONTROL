"""Utilidades compartidas para la extracción de fotogramas de vídeo."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from ..video_metadata import VideoInfo, read_video_file_info

# Rotaciones soportadas por OpenCV (múltiplos de 90°).
_ROTATE_CANDIDATES = (0, 90, 180, 270)


def normalize_rotation_deg(value: int) -> int:
    """Ajusta ``value`` al múltiplo de 90° más cercano admitido por OpenCV.

    Los metadatos de rotación pueden contener valores inesperados (por ejemplo,
    450 o 89). Para garantizar que los fotogramas y los *landmarks* se dibujen
    siempre con la misma orientación, reducimos el ángulo a uno de los
    cuadrantes compatibles.
    """

    value = int(value) % 360
    if value in _ROTATE_CANDIDATES:
        return value
    return min(
        _ROTATE_CANDIDATES,
        key=lambda cand: min((value - cand) % 360, (cand - value) % 360),
    )


def _seek_to_msec(cap: cv2.VideoCapture, msec: float) -> None:
    """Intenta desplazar el cursor del vídeo hasta el instante indicado en ms."""

    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(msec))
    except Exception:
        # Silencioso: algunos backends no soportan seek fiable.
        pass


def _open_video_capture(
    video_path: str | Path, cap: Optional[cv2.VideoCapture]
) -> tuple[cv2.VideoCapture, bool]:
    """Abre (o reutiliza) un ``VideoCapture`` y señala si somos dueños de él."""
    cap_obj = cap if cap is not None else cv2.VideoCapture(str(video_path))
    own_cap = cap is None

    if not cap_obj.isOpened():
        if own_cap:
            cap_obj.release()
        raise IOError(f"Could not open the video: {video_path}")
    return cap_obj, own_cap


def _load_video_info(
    path_obj: Path, cap_obj: cv2.VideoCapture, prefetched_info: Optional[VideoInfo]
) -> VideoInfo:
    """Recupera metadatos, reutilizando ``prefetched_info`` si ya existe."""
    info = prefetched_info
    if info is None:
        info = read_video_file_info(path_obj, cap=cap_obj)
    return info


def _resolve_rotation(rotate: int | str | None, info: VideoInfo) -> int:
    """Determina la rotación final a aplicar según la configuración y el vídeo."""
    if rotate == "auto" or rotate is None:
        return normalize_rotation_deg(int(info.rotation or 0))
    return normalize_rotation_deg(int(rotate))
