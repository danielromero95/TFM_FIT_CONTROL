"""Lector unificado de metadatos de vídeo con soporte opcional de ffprobe."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2

logger = logging.getLogger(__name__)

__all__ = ["VideoInfo", "get_video_metadata", "read_video_file_info"]


@dataclass(slots=True)
class VideoInfo:
    """Metadatos estructurados derivados de un archivo de vídeo."""
    path: Path
    width: int | None
    height: int | None
    fps: float | None              # None si no hay un valor válido
    frame_count: int | None        # None si no se puede determinar
    duration_sec: float | None     # None si no hay duración fiable
    rotation: int | None           # 0/90/180/270 o None
    codec: str | None              # mejor intento a partir del FOURCC
    fps_source: str | None         # "metadata" | "estimated" | "reader" | None


def _normalize_rotation(value: int) -> int:
    """Devuelve el valor más cercano entre 0/90/180/270 para ``value``."""
    value = int(value) % 360
    candidates = [0, 90, 180, 270]
    return min(candidates, key=lambda c: abs(c - value))


def _read_rotation_ffprobe(path: Path) -> Optional[int]:
    """Lee la rotación desde los tags de ffprobe; devuelve None si no existe."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate",
            "-of", "default=nw=1:nk=1",
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = (res.stdout or "").strip()
        if output:
            return _normalize_rotation(int(float(output)))
    except Exception as exc:  # pragma: no cover
        logger.warning("ffprobe rotation check failed: %s", exc)
    return None


def _estimate_duration_seconds(cap: cv2.VideoCapture, frame_count: int) -> float:
    """Calcula la duración aproximada usando la marca temporal del último frame."""
    if frame_count <= 1:
        return 0.0

    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    try:
        last_frame_index = max(frame_count - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame_index)
        if cap.grab():
            duration_msec = cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0
            return float(duration_msec) / 1000.0 if duration_msec > 0 else 0.0
    except Exception:  # pragma: no cover - dependiente del backend (ruta de respaldo)
        return 0.0
    finally:
        try:
            if original_pos is not None and original_pos >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        except Exception:  # pragma: no cover - dependiente del backend (ruta de respaldo)
            pass
    return 0.0


def _safe_fraction_to_float(value: str | None) -> float | None:
    """Convert fraction-like strings (e.g., ``"30000/1001"``) to floats."""

    if not value:
        return None
    try:
        if "/" in value:
            num, den = value.split("/", maxsplit=1)
            den_val = float(den)
            if den_val != 0:
                return float(num) / den_val
        return float(value)
    except Exception:  # pragma: no cover - defensive
        return None


def read_video_file_info(path: str | Path, cap: cv2.VideoCapture | None = None) -> VideoInfo:
    """
    Obtiene metadatos completos usando OpenCV y, si existe, ffprobe.

    Reglas:
    - Si el FPS reportado por OpenCV es inválido (<=1 o no finito), intentamos estimarlo con ``frame_count / duration``.
    - Si la estimación no es posible, dejamos ``fps=None`` y marcamos ``fps_source="reader"``.
    - La rotación se consulta vía ffprobe si está disponible; de lo contrario se asume 0.

    Cuando se entrega ``cap``, se reutiliza el ``VideoCapture`` sin asumir su propiedad.
    Quien llama sigue siendo responsable de cerrarlo; esta función solo restablece la
    posición de lectura al inicio antes de regresar.
    """
    p = Path(path)
    if not p.exists():
        raise IOError(f"Video path does not exist: {p}")

    cap_local = cap if cap is not None else cv2.VideoCapture(str(p))
    own_cap = cap is None

    if not cap_local.isOpened():
        if own_cap:
            cap_local.release()
        raise IOError(f"Could not open the video: {p}")

    try:
        # Dimensiones
        width_val = int(cap_local.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height_val = int(cap_local.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        width = width_val if width_val > 0 else None
        height = height_val if height_val > 0 else None

        # FPS y número de fotogramas
        fps_raw = float(cap_local.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count_raw = int(cap_local.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = frame_count_raw if frame_count_raw > 0 else None

        fps: float | None
        duration: float | None
        fps_source: str | None

        fps_valid = math.isfinite(fps_raw) and fps_raw > 1.0

        if fps_valid:
            fps = fps_raw
            fps_source = "metadata"
            duration = (frame_count_raw / fps_raw) if frame_count_raw > 0 else None
        else:
            # Intentamos estimar la duración con el último frame y derivar FPS = frames/duración
            duration_est = _estimate_duration_seconds(cap_local, frame_count_raw)
            if duration_est > 0.0 and frame_count_raw > 0:
                fps = frame_count_raw / duration_est
                duration = duration_est
                fps_source = "estimated"
                logger.warning(
                    "Invalid metadata FPS. Estimated from duration: %.2f fps.", fps
                )
            else:
                fps = None
                duration = None
                fps_source = "reader"
                logger.warning(
                    "Invalid metadata FPS and unreliable duration. Falling back to reader."
                )

        # Rotación (mejor esfuerzo)
        rotation = _read_rotation_ffprobe(p)
        if rotation is None:
            rotation = 0

        # Codec (mejor intento a partir del FOURCC)
        fourcc_int = int(cap_local.get(cv2.CAP_PROP_FOURCC) or 0)
        if fourcc_int:
            # Convertimos el entero FOURCC a un código de 4 caracteres
            codec_chars = [
                chr((fourcc_int & 0xFF)),
                chr((fourcc_int >> 8) & 0xFF),
                chr((fourcc_int >> 16) & 0xFF),
                chr((fourcc_int >> 24) & 0xFF),
            ]
            codec = "".join(codec_chars).strip() or None
        else:
            codec = None

        logger.info(
            "VideoInfo: path=%s size=%sx%s frames=%s fps=%s (source=%s) rot=%s codec=%s",
            p.name, width, height, frame_count, fps, fps_source, rotation, codec
        )

        return VideoInfo(
            path=p,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration,
            rotation=rotation,
            codec=codec,
            fps_source=fps_source,
        )

    finally:
        if own_cap:
            cap_local.release()
        else:
            try:
                cap_local.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:  # pragma: no cover - dependiente del backend (ruta de respaldo)
                pass


def _run_ffprobe(path: Path) -> dict[str, Any]:
    """Recupera metadatos detallados usando ``ffprobe`` si está disponible."""

    if shutil.which("ffprobe") is None:
        return {}
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            (
                "format=format_name,format_long_name,duration,size:format_tags=creation_time:"
                "stream=codec_name,codec_long_name,profile,pix_fmt,width,height,"
                "nb_frames,avg_frame_rate,r_frame_rate,tags,codec_type"
            ),
            str(path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            return {}
        return json.loads(res.stdout or "{}")
    except Exception as exc:  # pragma: no cover - dependiente de entorno
        logger.warning("ffprobe metadata check failed: %s", exc)
        return {}


def _safe_relative_path(path: Path) -> str:
    """Construye una ruta relativa sin exponer directorios del usuario."""

    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:
        return path.name


def get_video_metadata(path: Path) -> dict[str, Any]:
    """Extrae un diccionario amplio de metadatos del archivo de vídeo.

    Se apoya en OpenCV y, si está disponible, en ``ffprobe`` para rellenar
    información adicional. Los campos que no se puedan determinar quedarán
    en ``None`` o cadenas vacías.
    """

    p = Path(path)
    metadata: dict[str, Any] = {
        "input_file_name": p.name,
        "input_file_path": _safe_relative_path(p),
        "file_size_bytes": None,
        "container_format": None,
        "video_codec": None,
        "video_codec_long_name": None,
        "profile": None,
        "pixel_format": None,
        "width": None,
        "height": None,
        "rotation": None,
        "duration_s": None,
        "fps_r_frame_rate": None,
        "fps_avg_frame_rate": None,
        "total_frames_estimated": None,
        "audio_present": None,
        "audio_codec": None,
        "creation_time": None,
        "timestamp_extracted_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "debug_opencv": {},
    }

    try:
        metadata["file_size_bytes"] = int(p.stat().st_size)
    except Exception:
        metadata["file_size_bytes"] = None

    try:
        info = read_video_file_info(p)
    except Exception as exc:  # pragma: no cover - defensivo
        logger.warning("Could not read video info via OpenCV: %s", exc)
        info = None

    if info:
        metadata.update(
            {
                "width": info.width,
                "height": info.height,
                "rotation": info.rotation,
                "duration_s": info.duration_sec,
                "total_frames_estimated": info.frame_count,
            }
        )

        metadata["debug_opencv"] = {
            "fps_metadata": info.fps,
            "fps_source": info.fps_source,
            "codec_fourcc": info.codec,
        }

        if metadata.get("fps_avg_frame_rate") is None and info.fps:
            metadata["fps_avg_frame_rate"] = info.fps
        if metadata.get("fps_r_frame_rate") is None and info.fps:
            metadata["fps_r_frame_rate"] = info.fps

    ffprobe_data = _run_ffprobe(p)
    format_data = ffprobe_data.get("format", {}) if isinstance(ffprobe_data, dict) else {}
    streams = ffprobe_data.get("streams", []) if isinstance(ffprobe_data, dict) else []

    video_stream = None
    audio_stream = None
    for stream in streams or []:
        if not isinstance(stream, dict):
            continue
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        if stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if format_data:
        metadata["container_format"] = format_data.get("format_name") or metadata.get(
            "container_format"
        )
        metadata["duration_s"] = (
            float(format_data["duration"])
            if format_data.get("duration") not in (None, "N/A")
            else metadata.get("duration_s")
        )
        if format_data.get("size"):
            try:
                metadata["file_size_bytes"] = int(format_data["size"])
            except Exception:
                pass
        creation_time = None
        tags = format_data.get("tags") or {}
        if isinstance(tags, dict):
            creation_time = tags.get("creation_time")
        metadata["creation_time"] = creation_time or metadata.get("creation_time")

    if video_stream:
        metadata.update(
            {
                "video_codec": video_stream.get("codec_name")
                or metadata.get("video_codec"),
                "video_codec_long_name": video_stream.get("codec_long_name")
                or metadata.get("video_codec_long_name"),
                "profile": video_stream.get("profile") or metadata.get("profile"),
                "pixel_format": video_stream.get("pix_fmt")
                or metadata.get("pixel_format"),
                "width": video_stream.get("width") or metadata.get("width"),
                "height": video_stream.get("height") or metadata.get("height"),
                "fps_r_frame_rate": video_stream.get("r_frame_rate")
                or metadata.get("fps_r_frame_rate"),
                "fps_avg_frame_rate": video_stream.get("avg_frame_rate")
                or metadata.get("fps_avg_frame_rate"),
                "total_frames_estimated": video_stream.get("nb_frames")
                or metadata.get("total_frames_estimated"),
            }
        )

        rotation_tag = None
        tags = video_stream.get("tags") or {}
        if isinstance(tags, dict):
            rotation_tag = tags.get("rotate")
            creation_time = tags.get("creation_time")
            metadata["creation_time"] = creation_time or metadata.get("creation_time")
        if rotation_tag is not None:
            try:
                metadata["rotation"] = _normalize_rotation(int(float(rotation_tag)))
            except Exception:
                metadata["rotation"] = metadata.get("rotation")

        fps_avg = _safe_fraction_to_float(metadata.get("fps_avg_frame_rate"))
        if metadata.get("total_frames_estimated") in (None, "") and fps_avg and metadata.get("duration_s"):
            try:
                metadata["total_frames_estimated"] = int(
                    round(float(metadata["duration_s"]) * fps_avg)
                )
            except Exception:
                pass

    if audio_stream:
        metadata["audio_present"] = True
        metadata["audio_codec"] = audio_stream.get("codec_name") or metadata.get(
            "audio_codec"
        )
    elif metadata.get("audio_present") is None:
        metadata["audio_present"] = False

    return metadata
