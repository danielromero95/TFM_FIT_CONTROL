"""Herramientas dedicadas a la generación de videos de depuración con *overlays* de pose.

Aquí reunimos funciones cuyo único objetivo es reconstruir cuadros originales y renderizar
las poses filtradas sobre ellos. De esta forma el módulo principal del pipeline permanece
enfocado en la orquestación y delega la lógica específica de visualización.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from src.A_preprocessing.frame_extraction import extract_frames_stream
from src.D_visualization import render_landmarks_video
from src.D_visualization.landmark_video_io import make_web_safe_h264


@dataclass(slots=True)
class OverlayVideoResult:
    """Agrupa artefactos generados al crear el overlay con landmarks."""

    raw_path: Path
    stream_path: Path
    web_safe_ok: bool


def iter_original_frames_for_overlay(
    video_path: Path,
    *,
    rotate: int,
    sample_rate: int,
    target_fps: Optional[float],
    max_frames: int,
    max_long_side: Optional[int] = None,
) -> Iterable[np.ndarray]:
    """Recuperar cuadros originales utilizando el mismo muestreo de la pipeline."""

    target = float(target_fps) if target_fps and target_fps > 0 else None
    sampling_mode = "time" if target is not None else "index"

    kwargs: dict[str, object] = {
        "video_path": str(video_path),
        "sampling": sampling_mode,
        "rotate": int(rotate),
        "resize_to": None,
    }
    if sampling_mode == "time":
        kwargs["target_fps"] = target
    else:
        kwargs["every_n"] = max(1, int(sample_rate))

    iterator = extract_frames_stream(**kwargs)
    produced = 0
    for finfo in iterator:
        frame = finfo.array
        if max_long_side and max_long_side > 0:
            height, width = frame.shape[:2]
            long_side = max(height, width)
            if long_side > max_long_side:
                scale = max_long_side / float(long_side)
                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        yield frame
        produced += 1
        if max_frames and produced >= max_frames:
            break


def generate_overlay_video(
    video_path: Path,
    session_dir: Path,
    *,
    frame_sequence: np.ndarray,
    crop_boxes: Optional[np.ndarray],
    processed_size: Optional[tuple[int, int]],
    rotate: int,
    sample_rate: int,
    target_fps: Optional[float],
    fps_for_writer: float,
    output_rotate: int = 0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    overlay_max_long_side: Optional[int] = None,
) -> Optional[OverlayVideoResult]:
    """Renderizar un video de depuración con las poses reconstruidas.

    Devuelve ``OverlayVideoResult`` con la ruta original y la copia web-safe
    preparada para su reproducción en navegadores. Si el proceso falla se
    retorna ``None``.
    """

    if frame_sequence is None:
        return None

    total_frames = len(frame_sequence)
    if total_frames == 0:
        return None

    if not processed_size or processed_size[0] <= 0 or processed_size[1] <= 0:
        return None

    is_df = False
    try:
        import pandas as pd  # noqa: WPS433 - importación interna para no obligar dependencia global

        is_df = isinstance(frame_sequence, pd.DataFrame)
    except Exception:
        is_df = False

    if is_df:
        cols_x = [c for c in frame_sequence.columns if c.startswith("x")]
        num_landmarks = len(cols_x)
        normalized_sequence: list[list[dict[str, float]]] = []
        for _, row in frame_sequence.iterrows():
            frame_landmarks: list[dict[str, float]] = []
            for i in range(num_landmarks):
                x_val = float(row.get(f"x{i}", float("nan")))
                y_val = float(row.get(f"y{i}", float("nan")))
                frame_landmarks.append({"x": x_val, "y": y_val})
            normalized_sequence.append(frame_landmarks)
        frame_sequence = normalized_sequence  # type: ignore[assignment]

    processed_w = int(processed_size[0])
    processed_h = int(processed_size[1])

    overlay_path = session_dir / f"{session_dir.name}_overlay.mp4"
    fps_value = float(fps_for_writer if fps_for_writer and fps_for_writer > 0 else 1.0)

    frames_iter = iter_original_frames_for_overlay(
        video_path,
        rotate=rotate,
        sample_rate=sample_rate,
        target_fps=target_fps,
        max_frames=total_frames,
        max_long_side=overlay_max_long_side,
    )

    stats = render_landmarks_video(
        frames_iter,
        frame_sequence,
        crop_boxes,
        str(overlay_path),
        fps=float(fps_value),
        processed_size=(processed_w, processed_h),
        output_rotate=int(output_rotate) % 360,
        tighten_to_subject=False,
        subject_margin=0.15,
        progress_cb=progress_cb,
    )

    if stats.frames_written <= 0:
        overlay_path.unlink(missing_ok=True)
        return None

    web_safe = make_web_safe_h264(overlay_path)
    stream_path = overlay_path
    if web_safe.ok and web_safe.output_path is not None:
        stream_path = web_safe.output_path

    return OverlayVideoResult(
        raw_path=overlay_path,
        stream_path=stream_path,
        web_safe_ok=web_safe.ok,
    )
