"""Utilities for rendering uniformly sized videos in Streamlit."""

from __future__ import annotations

import inspect
import io
import math
import mimetypes
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit.components.v1 import html

from src.A_preprocessing.video_metadata import read_video_file_info

VideoData = Union[str, bytes, BinaryIO]


@dataclass(frozen=True)
class PortraitPreview:
    """Container with a pre-rendered portrait preview and metadata."""

    image_bytes: bytes
    used_smart_center: bool


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    rotation = rotation % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _weighted_center(weights: np.ndarray) -> tuple[float | None, float]:
    if weights.size == 0:
        return None, 0.0
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None, 0.0
    positions = np.arange(weights.size, dtype=np.float64)
    center = float(np.dot(positions, weights) / total)
    peak_ratio = float(weights.max() / total) if total else 0.0
    return center, peak_ratio


@st.cache_data(show_spinner=False)
def detect_video_orientation(path: str | Path) -> str:
    """Return ``horizontal`` or ``vertical`` based on the video dimensions."""

    info = read_video_file_info(path)
    width = info.width or 0
    height = info.height or 0
    rotation = info.rotation or 0
    if rotation in (90, 270):
        width, height = height, width
    if width <= 0 or height <= 0:
        return "unknown"
    return "vertical" if height >= width else "horizontal"


@st.cache_data(show_spinner=False)
def generate_portrait_preview(
    path: str | Path,
    *,
    width: int = 270,
    sample_frames: int = 8,
) -> PortraitPreview | None:
    """Return a portrait (9:16) preview cropped using a smart center heuristic."""

    p = Path(path)
    if not p.exists():
        return None

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None

    try:
        try:
            info = read_video_file_info(p, cap=cap)
        except Exception:
            info = None

        rotation = (info.rotation if info else 0) or 0

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 0 and sample_frames > 0:
            indices = np.linspace(0, max(frame_count - 1, 0), num=sample_frames, dtype=int)
        else:
            indices = np.arange(min(sample_frames, max(frame_count, 1)))

        col_energy: np.ndarray | None = None
        row_energy: np.ndarray | None = None
        preview_frame: np.ndarray | None = None

        for idx in indices:
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _apply_rotation(frame, rotation)
            if preview_frame is None:
                preview_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 32, 128)

            cols = edges.sum(axis=0).astype(np.float64)
            rows = edges.sum(axis=1).astype(np.float64)
            col_energy = cols if col_energy is None else col_energy + cols
            row_energy = rows if row_energy is None else row_energy + rows

        if preview_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
            preview_frame = _apply_rotation(frame, rotation)

        frame_h, frame_w = preview_frame.shape[:2]
        target_ratio = 9 / 16

        center_x, peak_x = _weighted_center(col_energy) if col_energy is not None else (None, 0.0)
        center_y, peak_y = _weighted_center(row_energy) if row_energy is not None else (None, 0.0)

        smart_x = center_x is not None and peak_x >= 0.01
        smart_y = center_y is not None and peak_y >= 0.01

        cx = float(center_x) if smart_x else frame_w / 2.0
        cy = float(center_y) if smart_y else frame_h / 2.0

        if frame_w / frame_h >= target_ratio:
            crop_w = int(round(frame_h * target_ratio))
            crop_w = min(max(crop_w, 1), frame_w)
            x0 = int(round(cx - crop_w / 2))
            x0 = max(0, min(x0, frame_w - crop_w))
            y0 = 0
            crop_h = frame_h
        else:
            crop_h = int(round(frame_w / target_ratio))
            crop_h = min(max(crop_h, 1), frame_h)
            y0 = int(round(cy - crop_h / 2))
            y0 = max(0, min(y0, frame_h - crop_h))
            x0 = 0
            crop_w = frame_w

        crop = preview_frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
        if crop.size == 0:
            crop = preview_frame

        target_height = max(1, int(round(width * 16 / 9)))
        resized = cv2.resize(crop, (int(width), target_height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return PortraitPreview(buffer.getvalue(), used_smart_center=smart_x or smart_y)
    finally:
        cap.release()


@st.cache_data(show_spinner=False)
def _data_uri(path: str, *, mtime: float, mime: str) -> str:
    """Return a base64 data URI for a local video file."""

    data = Path(path).read_bytes()
    encoded = b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _normalize_mime(value: str | None) -> str | None:
    if value is None:
        return None
    mime = value.strip()
    if not mime:
        return None
    if "/" in mime:
        return mime
    return f"video/{mime.lstrip('.')}"


def _detect_mime(path: Path, fallback: str | None) -> str:
    mime_hint = _normalize_mime(fallback)
    if mime_hint:
        return mime_hint
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    return "video/mp4"


def _as_bytes(data: BinaryIO) -> bytes:
    position = getattr(data, "tell", None)
    if position is not None:
        try:
            start = data.tell()
        except Exception:
            start = None
    else:
        start = None
    try:
        return data.read()
    finally:
        if start is not None:
            try:
                data.seek(start)
            except Exception:
                pass


def _video_source(
    data: VideoData,
    *,
    format: str | None,
) -> tuple[str, str]:
    """Return ``(source_uri, mime_type)`` for ``data``."""

    if isinstance(data, (bytes, bytearray)):
        mime = _normalize_mime(format) or "video/mp4"
        encoded = b64encode(bytes(data)).decode("ascii")
        return f"data:{mime};base64,{encoded}", mime

    if hasattr(data, "read"):
        buffer = _as_bytes(data)  # type: ignore[arg-type]
        mime = _normalize_mime(format) or "video/mp4"
        encoded = b64encode(buffer).decode("ascii")
        return f"data:{mime};base64,{encoded}", mime

    path = Path(str(data)).expanduser()
    mime = _detect_mime(path, format)
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        raise FileNotFoundError(f"Video file not found: {path}")
    return _data_uri(str(path), mtime=mtime, mime=mime), mime


def get_video_source(
    data: VideoData,
    *,
    format: str | None = None,
) -> tuple[str, str]:
    """Public wrapper around _video_source for reuse in other modules."""

    return _video_source(data, format=format)


def render_uniform_video(
    data: VideoData,
    *,
    format: str | None = None,
    start_time: int = 0,
    key: str | None = None,
    bottom_margin: float = 1.25,
    max_width: int | None = 720,
    viewport_height: int | None = None,
) -> None:
    """Render ``data`` inside a consistent viewport."""

    source, mime = _video_source(data, format=format)
    component_key = key or f"uniform_video_{uuid4().hex}"
    marker_id = f"uniform-video-marker-{uuid4().hex}"
    inner_id = f"uniform-video-inner-{uuid4().hex}"
    video_id = f"uniform-video-{uuid4().hex}"

    st.markdown(f'<div id="{marker_id}"></div>', unsafe_allow_html=True)

    padding = 24
    if viewport_height is not None:
        height = max(int(viewport_height), 120)
        initial_height = height + padding
    else:
        reference_width = max_width or 720
        initial_height = int(round(reference_width * 9 / 16)) + padding
        height = None
    height_expr = str(height) if height is not None else "Math.round(w * 9 / 16)"

    html_kwargs: dict[str, object] = {"height": initial_height, "scrolling": False}
    try:
        signature = inspect.signature(html)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "key" in signature.parameters:
        if component_key is not None:
            html_kwargs["key"] = component_key

    container_styles = [
        "position:relative",
        "width:100%",
        "background:#000",
        "border-radius:12px",
        "overflow:hidden",
    ]
    if height is not None:
        container_styles.append(f"height:{height}px")
    else:
        container_styles.append("padding-top:56.25%")

    wrapper_styles = ["width:100%", "margin:0 auto"]
    if max_width is not None:
        wrapper_styles.append(f"max-width:{max_width}px")

    html(
        f"""
        <div id="{inner_id}" style="{';'.join(wrapper_styles)};">
          <div style="{';'.join(container_styles)};">
            <video
              id="{video_id}"
              controls
              preload="metadata"
              playsinline
              webkit-playsinline
              style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;"
            >
              <source src="{source}" type="{mime}">
            </video>
          </div>
        </div>
        <script>
          (function() {{
            const pad = {padding};
            const inner = document.getElementById('{inner_id}');
            const video = document.getElementById('{video_id}');
            if (!inner || !video) return;
            const fit = () => {{
              const w = inner.clientWidth || {max_width or 720};
              const h = {height_expr} + pad;
              if (window.frameElement) {{
                window.frameElement.style.height = h + 'px';
              }}
            }};
            if (typeof ResizeObserver !== 'undefined') {{
              new ResizeObserver(() => fit()).observe(inner);
            }} else {{
              window.addEventListener('resize', () => fit());
            }}
            fit();
            window.addEventListener('load', () => fit(), {{ once: true }});
            const start = Math.max({start_time}, 0);
            video.addEventListener('loadedmetadata', () => {{
              if (!start) return;
              try {{
                video.currentTime = start;
              }} catch (err) {{
                console.warn('Unable to seek video start time', err);
              }}
            }}, {{ once: true }});
          }})();
        </script>
        """,
        **html_kwargs,
    )

    margin_value = max(bottom_margin, 0.0)
    max_width_css = f"max-width:{max_width}px" if max_width is not None else "max-width:100%"
    st.markdown(
        f"""
        <style>
        #{marker_id} + iframe {{
          width: 100% !important;
          {max_width_css} !important;
          margin: 0 auto {margin_value:.2f}rem auto !important;
          display: block !important;
          border-radius: 12px !important;
          background: #000 !important;
          overflow: hidden !important;
          box-shadow: 0 18px 36px rgba(15, 23, 42, 0.35);
        }}
        #{marker_id} + iframe > iframe {{
          border-radius: 12px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

