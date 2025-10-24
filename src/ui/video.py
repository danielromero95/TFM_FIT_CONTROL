"""Utilities for rendering uniformly sized videos in Streamlit."""

from __future__ import annotations

import inspect
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import BinaryIO, Dict, Tuple, Union
from uuid import uuid4

import cv2
import numpy as np
import streamlit as st
from streamlit.components.v1 import html

VideoData = Union[str, bytes, BinaryIO]

PORTRAIT_RATIO = 9.0 / 16.0
DEFAULT_PREVIEW_HEIGHT = 960


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
) -> None:
    """Render ``data`` inside a responsive 16:9 viewport."""

    source, mime = _video_source(data, format=format)
    component_key = key or f"uniform_video_{uuid4().hex}"
    marker_id = f"uniform-video-marker-{uuid4().hex}"
    inner_id = f"uniform-video-inner-{uuid4().hex}"
    video_id = f"uniform-video-{uuid4().hex}"

    st.markdown(f'<div id="{marker_id}"></div>', unsafe_allow_html=True)

    padding = 24
    initial_height = int(round(720 * 9 / 16)) + padding
    html_kwargs: dict[str, object] = {"height": initial_height, "scrolling": False}
    try:
        signature = inspect.signature(html)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "key" in signature.parameters:
        if component_key is not None:
            html_kwargs["key"] = component_key

    html(
        f"""
        <div id="{inner_id}" style="width:100%;max-width:720px;margin:0 auto;">
          <div style="position:relative;width:100%;padding-top:56.25%;background:#000;border-radius:12px;overflow:hidden;">
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
              const w = inner.clientWidth || 720;
              const h = Math.round(w * 9 / 16) + pad;
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
    st.markdown(
        f"""
        <style>
        #{marker_id} + iframe {{
          width: min(100%, 720px) !important;
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


@st.cache_data(show_spinner=False)
def _video_snapshot(path: str) -> Dict[str, object]:
    """Return cached metadata and a representative RGB frame for ``path``."""

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        return {"width": 0, "height": 0, "frame": None}

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(total_frames // 2, 0))

    success, frame = capture.read()
    if not success:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = capture.read()

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    if not success or frame is None:
        return {"width": width, "height": height, "frame": None}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return {"width": width, "height": height, "frame": rgb_frame}


def detect_video_orientation(path: str) -> Tuple[str, Tuple[int, int]]:
    """Return ``(orientation, (width, height))`` for ``path``."""

    snapshot = _video_snapshot(path)
    width = int(snapshot.get("width") or 0)
    height = int(snapshot.get("height") or 0)

    if width <= 0 or height <= 0:
        return "unknown", (0, 0)

    if width > height:
        orientation = "horizontal"
    elif height > width:
        orientation = "vertical"
    else:
        orientation = "square"

    return orientation, (width, height)


def _estimate_subject_center(frame: np.ndarray) -> int | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    column_scores = edges.sum(axis=0).astype(np.float64)
    max_score = float(column_scores.max(initial=0.0))
    if max_score <= 0:
        return None

    emphasis = np.where(column_scores >= max_score * 0.2, column_scores, 0.0)
    total = float(emphasis.sum())
    if total <= 0:
        return None

    indices = np.arange(emphasis.shape[0], dtype=np.float64)
    center = int(round(float(np.dot(indices, emphasis) / total)))
    return center


def _crop_to_portrait(frame: np.ndarray, center_x: int | None) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return frame

    frame_ratio = width / float(height)
    if frame_ratio >= PORTRAIT_RATIO:
        crop_width = max(1, min(width, int(round(height * PORTRAIT_RATIO))))
        if center_x is None:
            center_x = width // 2
        half = crop_width // 2
        x0 = max(0, min(width - crop_width, center_x - half))
        x1 = x0 + crop_width
        cropped = frame[:, x0:x1]
    else:
        crop_height = max(1, min(height, int(round(width / PORTRAIT_RATIO))))
        center_y = height // 2
        half = crop_height // 2
        y0 = max(0, min(height - crop_height, center_y - half))
        y1 = y0 + crop_height
        cropped = frame[y0:y1, :]

    return cropped


@st.cache_data(show_spinner=False)
def generate_portrait_preview(
    path: str,
    *,
    max_height: int = DEFAULT_PREVIEW_HEIGHT,
) -> np.ndarray | None:
    """Return an RGB numpy array representing a portrait 9:16 preview."""

    snapshot = _video_snapshot(path)
    frame = snapshot.get("frame")
    if frame is None or not isinstance(frame, np.ndarray):
        return None

    center_x = _estimate_subject_center(frame)
    cropped = _crop_to_portrait(frame, center_x)
    if cropped.size == 0:
        return None

    crop_height, crop_width = cropped.shape[:2]
    if crop_height <= 0 or crop_width <= 0:
        return None

    if center_x is None:
        # Try again with middle crop when heuristic failed.
        fallback = _crop_to_portrait(frame, frame.shape[1] // 2)
        if fallback.size != 0:
            cropped = fallback
            crop_height, crop_width = cropped.shape[:2]

    target_height = min(max_height, crop_height)
    target_width = max(1, int(round(target_height * PORTRAIT_RATIO)))
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

