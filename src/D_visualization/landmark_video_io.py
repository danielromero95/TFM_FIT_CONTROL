"""Herramientas de vídeo orientadas a depurar renderizados de marcadores.
Gestiona códecs y transcodificaciones para que los clips resultantes se reproduzcan en
equipos donde quizá no esté instalado el mismo stack multimedia que en desarrollo."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import cv2

logger = logging.getLogger(__name__)

DEFAULT_CODEC_PREFERENCE: tuple[str, ...] = ("avc1", "H264", "mp4v")
"""Preferencia por códecs al abrir ``VideoWriter`` en distintos entornos."""

WEB_SAFE_SUFFIX = "_stream"
"""Sufijo aplicado a los vídeos convertidos a H.264 para streaming web."""


@dataclass(slots=True)
class WebSafeVideoResult:
    """Representa el resultado de intentar generar un MP4 reproducible en navegadores."""

    output_path: Path | None
    ok: bool
    used_ffmpeg: bool
    error: str | None = None


# Exponemos las utilidades de vídeo utilizadas por los renderizadores de anotaciones.
__all__ = [
    "DEFAULT_CODEC_PREFERENCE",
    "WEB_SAFE_SUFFIX",
    "WebSafeVideoResult",
    "_open_writer",
    "transcode_video",
    "make_web_safe_h264",
]


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
    codec_preference: Sequence[str] = ("avc1", "H264"),
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


def make_web_safe_h264(
    src_path: str | Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    suffix: str = WEB_SAFE_SUFFIX,
) -> WebSafeVideoResult:
    """Genera una copia H.264 + yuv420p + faststart reproducible en navegadores."""

    source = Path(str(src_path)).expanduser()
    if not source.exists():
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=False, error="missing")

    try:
        src_stat = source.stat()
    except OSError as exc:  # pragma: no cover - fallo poco común
        logger.warning("Could not stat source video for web-safe conversion: %s", exc)
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=False, error="stat_failed")

    destination = source.with_name(f"{source.stem}{suffix}{source.suffix}")

    try:
        if destination.exists():
            dst_stat = destination.stat()
            if dst_stat.st_size > 0 and dst_stat.st_mtime >= src_stat.st_mtime:
                logger.info(
                    "Reusing existing web-safe clip at %s (size=%d)",
                    destination,
                    dst_stat.st_size,
                )
                return WebSafeVideoResult(output_path=destination, ok=True, used_ffmpeg=False)
    except OSError as exc:  # pragma: no cover - se ignora fallo puntual
        logger.debug("Could not reuse previous web-safe clip: %s", exc)

    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    try:
        tmp_destination.unlink(missing_ok=True)
    except OSError:
        pass

    command = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(tmp_destination),
    ]

    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError:
        logger.warning(
            "ffmpeg binary not found when generating web-safe clip. Using original overlay.",
        )
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=False, error="ffmpeg_missing")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "ignore") if isinstance(exc.stderr, bytes) else exc.stderr
        logger.warning(
            "ffmpeg failed to convert overlay to web-safe mp4: returncode=%s stderr=%s",
            exc.returncode,
            stderr,
        )
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=True, error="ffmpeg_failed")

    if completed.returncode != 0:  # pragma: no cover - ``check=True`` debería impedirlo
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=True, error="ffmpeg_failed")

    if not tmp_destination.exists():
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=True, error="tmp_missing")

    try:
        size = tmp_destination.stat().st_size
    except OSError as exc:
        logger.warning("Could not stat ffmpeg output: %s", exc)
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=True, error="tmp_stat_failed")

    if size <= 0:
        logger.warning("Web-safe conversion produced empty file at %s", tmp_destination)
        tmp_destination.unlink(missing_ok=True)
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=True, error="empty")

    try:
        tmp_destination.replace(destination)
    except OSError as exc:
        logger.warning("Could not finalize web-safe clip rename: %s", exc)
        return WebSafeVideoResult(output_path=None, ok=False, used_ffmpeg=True, error="rename_failed")

    logger.info(
        "Generated web-safe overlay clip at %s (size=%d)",
        destination,
        size,
    )
    return WebSafeVideoResult(output_path=destination, ok=True, used_ffmpeg=True)
