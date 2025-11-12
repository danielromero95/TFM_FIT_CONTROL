"""Herramientas de vídeo orientadas a depurar renderizados de marcadores.
Gestiona códecs y transcodificaciones para que los clips resultantes se reproduzcan en
equipos donde quizá no esté instalado el mismo stack multimedia que en desarrollo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple

import cv2

from src.config.constants import DEFAULT_VIDEO_CODEC_PREFERENCE

logger = logging.getLogger(__name__)

# Exponemos las utilidades de vídeo utilizadas por los renderizadores de anotaciones.
__all__ = ["_open_writer", "transcode_video"]


def _open_writer(path: str, fps: float, size: Tuple[int, int], prefs: Sequence[str]) -> tuple[cv2.VideoWriter, str]:
    """Intenta abrir un ``VideoWriter`` probando múltiples preferencias de códec.
    Esta estrategia reduce los fallos en entornos heterogéneos donde los códecs
    disponibles cambian entre máquinas o sistemas operativos."""

    for code in prefs:
        # Probamos cada ``fourcc`` en orden de preferencia hasta encontrar uno utilizable.
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), max(fps, 1.0), size)
        if writer.isOpened():
            logger.info(
                "Opened VideoWriter fourcc=%s size=%s fps=%.2f path=%s",
                code,
                size,
                max(fps, 1.0),
                path,
            )
            return writer, code
        logger.info("VideoWriter open failed with fourcc=%s; trying next...", code)
        # Liberamos el recurso antes de intentar con el siguiente códec para evitar fugas.
        writer.release()
    raise RuntimeError(f"Could not open VideoWriter for path={path} with prefs={prefs}")


def transcode_video(
    src_path: str,
    dst_path: str,
    *,
    fps: float,
    codec_preference: Sequence[str] = DEFAULT_VIDEO_CODEC_PREFERENCE,
) -> tuple[bool, str]:
    """Re-encode ``src_path`` utilizando las preferencias de códec deseadas.
    Convertimos el archivo a un códec conocido para poder compartirlo con clientes o
    revisarlo en herramientas estándar aunque la captura original use formatos raros."""

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False, ""

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return False, ""

    try:
        writer, used_code = _open_writer(
            dst_path,
            fps if fps > 0 else 1.0,
            (width, height),
            codec_preference,
        )
    except Exception:
        cap.release()
        return False, ""

    frames_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames_written += 1
    finally:
        writer.release()
        cap.release()

    if frames_written <= 0:
        try:
            # Si la transcodificación no generó frames, eliminamos el archivo vacío.
            Path(dst_path).unlink(missing_ok=True)
        except Exception:
            pass
        return False, ""

    return True, used_code
