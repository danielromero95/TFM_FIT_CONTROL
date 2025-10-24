"""Utilities for rendering uniformly sized videos in Streamlit."""

from __future__ import annotations

import inspect
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import BinaryIO, Union
from uuid import uuid4

import streamlit as st
from streamlit.components.v1 import html

VideoData = Union[str, bytes, BinaryIO]


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

