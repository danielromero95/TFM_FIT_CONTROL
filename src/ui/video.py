"""Utilities for rendering uniformly sized videos in Streamlit."""

from __future__ import annotations

import mimetypes
import inspect
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

    html_template = r"""
        <div id="__INNER_ID__" style="width:100%;max-width:720px;margin:0 auto;">
          <div style="position:relative;width:100%;padding-top:56.25%;background:#000;border-radius:12px;overflow:hidden;">
            <video
              id="__VIDEO_ID__"
              controls
              preload="metadata"
              playsinline
              webkit-playsinline
              style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;"
            >
              <source src="__SOURCE__" type="__MIME__">
            </video>
          </div>
        </div>
        <script>
          (function() {
            const pad = __PADDING__;
            const inner = document.getElementById('__INNER_ID__');
            const video = document.getElementById('__VIDEO_ID__');
            if (!inner || !video) return;
            const fit = () => {
              const w = inner.clientWidth || 720;
              const h = Math.round(w * 9 / 16) + pad;
              if (window.frameElement) {
                window.frameElement.style.height = h + 'px';
              }
            };
            if (typeof ResizeObserver !== 'undefined') {
              new ResizeObserver(() => fit()).observe(inner);
            } else {
              window.addEventListener('resize', () => fit());
            }
            fit();
            window.addEventListener('load', () => fit(), { once: true });
            const start = Math.max(__START_TIME__, 0);
            video.addEventListener('loadedmetadata', () => {
              if (!start) return;
              try {
                video.currentTime = start;
              } catch (err) {
                console.warn('Unable to seek video start time', err);
              }
            }, { once: true });
          })();
        </script>
    """

    html_content = (
        html_template
        .replace("__INNER_ID__", inner_id)
        .replace("__VIDEO_ID__", video_id)
        .replace("__SOURCE__", source)
        .replace("__MIME__", mime)
        .replace("__PADDING__", str(padding))
        .replace("__START_TIME__", str(start_time))
    )

    html(
        html_content,
        **html_kwargs,
    )

    margin_value = max(bottom_margin, 0.0)
    css_template = r"""
        <style>
        #__MARKER_ID__ + iframe {
          width: min(100%, 720px) !important;
          margin: 0 auto __MARGIN_VALUE__rem auto !important;
          display: block !important;
          border-radius: 12px !important;
          background: #000 !important;
          overflow: hidden !important;
          box-shadow: 0 18px 36px rgba(15, 23, 42, 0.35);
        }
        #__MARKER_ID__ + iframe > iframe {
          border-radius: 12px !important;
        }
        </style>
    """

    css = (
        css_template
        .replace("__MARKER_ID__", marker_id)
        .replace("__MARGIN_VALUE__", format(margin_value, ".2f"))
    )

    st.markdown(
        css,
        unsafe_allow_html=True,
    )

