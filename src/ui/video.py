"""Utilities for rendering uniformly sized videos in Streamlit."""

from __future__ import annotations

import mimetypes
import inspect
from base64 import b64encode
from pathlib import Path
from math import isclose
from typing import BinaryIO, Final, Union
from uuid import uuid4

import streamlit as st
from streamlit.components.v1 import html

VideoData = Union[str, bytes, BinaryIO]
VIDEO_VIEWPORT_HEIGHT_PX: Final[int] = 400
DEFAULT_VIDEO_BOTTOM_REM: Final[float] = 1.25


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
    bottom_margin: float = DEFAULT_VIDEO_BOTTOM_REM,
    fixed_height_px: int | None = VIDEO_VIEWPORT_HEIGHT_PX,
) -> None:
    """Render ``data`` inside a viewport with fixed height and full width."""

    source, mime = _video_source(data, format=format)
    component_key = key or f"uniform_video_{uuid4().hex}"
    marker_id = f"uniform-video-marker-{uuid4().hex}"
    inner_id = f"uniform-video-inner-{uuid4().hex}"
    video_id = f"uniform-video-{uuid4().hex}"

    st.markdown(f'<div id="{marker_id}"></div>', unsafe_allow_html=True)

    iframe_pad_px = 24
    fixed_h = int(fixed_height_px or 400)
    initial_height = fixed_h + iframe_pad_px
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
        <div id="{inner_id}" style="width:100%;margin:0 auto;">
          <div style="position:relative;width:100%;height:{fixed_h}px;background:#000;border-radius:var(--fc-radius, 12px);overflow:hidden;">
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
            const fallbackPad = {iframe_pad_px};
            const padVar = getComputedStyle(document.documentElement).getPropertyValue('--fc-iframe-pad').trim();
            const parsedPad = Number.parseFloat(padVar || '');
            const pad = Number.isNaN(parsedPad) ? fallbackPad : parsedPad;
            const inner = document.getElementById('{inner_id}');
            const video = document.getElementById('{video_id}');
            if (!inner) return;
            const fit = () => {{
              const h = {fixed_h} + pad;
              if (window.frameElement) {{
                window.frameElement.style.height = h + 'px';
              }}
            }};
            if (typeof ResizeObserver !== 'undefined') {{
              new ResizeObserver(() => fit()).observe(inner);
            }} else {{
              window.addEventListener('resize', fit);
            }}
            fit();
            window.addEventListener('load', () => fit(), {{ once: true }});
            if (video) {{
              const start = Math.max({start_time}, 0);
              video.addEventListener('loadedmetadata', () => {{
                if (!start) return;
                try {{
                  video.currentTime = start;
                }} catch (err) {{
                  console.warn('Unable to seek video start time', err);
                }}
              }}, {{ once: true }});
            }}
          }})();
        </script>
        """,
        **html_kwargs,
    )

    margin_value = max(bottom_margin, 0.0)
    override_line = ""
    if not isclose(margin_value, DEFAULT_VIDEO_BOTTOM_REM, rel_tol=1e-9, abs_tol=1e-9):
        override_line = f"          --uniform-video-bottom: {margin_value:.2f}rem;\n"
    st.markdown(
        f"""
        <style>
        #{marker_id} + iframe {{
          width: 100% !important;
{override_line}          margin: 0 auto var(--uniform-video-bottom, var(--fc-video-bottom, 1.25rem)) auto !important;
          display: block !important;
          border-radius: var(--fc-radius, 12px) !important;
          background: #000 !important;
          overflow: hidden !important;
          box-shadow: var(--fc-shadow-elev-2, 0 18px 36px rgba(15, 23, 42, .35));
        }}
        #{marker_id} + iframe > iframe {{
          border-radius: var(--fc-radius, 12px) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

