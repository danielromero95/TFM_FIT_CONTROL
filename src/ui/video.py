"""Utilidades para renderizar vídeos con altura uniforme en Streamlit."""

from __future__ import annotations

import json
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
    """Devuelve un data URI en base64 para un vídeo almacenado en disco."""

    data = Path(path).read_bytes()
    encoded = b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _normalize_mime(value: str | None) -> str | None:
    """Normaliza un identificador MIME incompleto hacia el formato ``tipo/subtipo``."""

    if value is None:
        return None
    mime = value.strip()
    if not mime:
        return None
    if "/" in mime:
        return mime
    return f"video/{mime.lstrip('.')}"


def _detect_mime(path: Path, fallback: str | None) -> str:
    """Intenta inferir el tipo MIME del archivo, usando ``fallback`` si hace falta."""

    mime_hint = _normalize_mime(fallback)
    if mime_hint:
        return mime_hint
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    return "video/mp4"


def _as_bytes(data: BinaryIO) -> bytes:
    """Lee un buffer binario preservando la posición original del puntero."""

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
    """Devuelve ``(source_uri, mime_type)`` a partir de los datos de vídeo recibidos."""

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
    """Envoltura pública de ``_video_source`` para reutilizar en otros módulos."""

    return _video_source(data, format=format)


def render_uniform_video(
    data: VideoData,
    *,
    format: str | None = None,
    start_time: int = 0,
    key: str | None = None,
    bottom_margin: float = DEFAULT_VIDEO_BOTTOM_REM,
    fixed_height_px: int | None = VIDEO_VIEWPORT_HEIGHT_PX,
    portrait_height_px: int = 560,
    sync_channel: str | None = None,
) -> None:
    """Renderiza el vídeo dentro de un contenedor ancho con altura controlada."""

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
              style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;object-position:center;background:#000;"
            >
              <source src="{source}" type="{mime}">
            </video>
          </div>
        </div>
        <script>
          (function() {{
            const SYNC_CHANNEL = {json.dumps(sync_channel)};
            const fallbackPad = {iframe_pad_px};
            const padVar = getComputedStyle(document.documentElement).getPropertyValue('--fc-iframe-pad').trim();
            const parsedPad = Number.parseFloat(padVar || '');
            const pad = Number.isNaN(parsedPad) ? fallbackPad : parsedPad;
            const inner = document.getElementById('{inner_id}');
            const video = document.getElementById('{video_id}');
            if (!inner) return;
            let viewportHeight = {fixed_h};
            const applyHeight = () => {{
              const h = viewportHeight + pad;
              if (window.frameElement) {{
                window.frameElement.style.height = h + 'px';
              }}
            }};
            const setViewportHeight = (value) => {{
              const next = Number.parseFloat(value);
              if (!Number.isFinite(next) || next <= 0) {{
                viewportHeight = {fixed_h};
              }} else {{
                viewportHeight = next;
              }}
              applyHeight();
            }};
            if (typeof ResizeObserver !== 'undefined') {{
              new ResizeObserver(() => applyHeight()).observe(inner);
            }} else {{
              window.addEventListener('resize', applyHeight);
            }}
            applyHeight();
            window.addEventListener('load', () => applyHeight(), {{ once: true }});
            if (video) {{
              const start = Math.max({start_time}, 0);
              video.addEventListener('loadedmetadata', () => {{
                const vw = video.videoWidth || 0;
                const vh = video.videoHeight || 0;
                if (vh > 0 && vw > 0) {{
                  const isPortrait = vh > vw;
                  const rect = inner.getBoundingClientRect();
                  const containerWidth = rect.width || video.clientWidth || 0;
                  let targetHeight = isPortrait ? {portrait_height_px} : {fixed_h};
                  if (containerWidth > 0) {{
                    const candidate = Math.round(containerWidth * (vh / vw));
                    if (candidate > targetHeight) {{
                      targetHeight = candidate;
                    }}
                  }}
                  setViewportHeight(targetHeight);
                }}
                if (start) {{
                  try {{
                    video.currentTime = start;
                  }} catch (err) {{
                    console.warn('Unable to seek video start time', err);
                  }}
                }}
              }}, {{ once: true }});

              // BroadcastChannel sync (segundos)
              const bc = (SYNC_CHANNEL && typeof BroadcastChannel !== 'undefined')
                ? new BroadcastChannel(SYNC_CHANNEL) : null;
              if (bc) {{
                const publish = () => bc.postMessage({{ type: 'time', t: video.currentTime || 0 }});
                ['timeupdate','seeked','play'].forEach((ev) => video.addEventListener(ev, publish));
                const loop = () => {{
                  if (!video.paused && !video.ended) {{
                    bc.postMessage({{ type: 'time', t: video.currentTime || 0 }});
                    requestAnimationFrame(loop);
                  }}
                }};
                video.addEventListener('play', loop);
                bc.onmessage = (ev) => {{
                  const msg = ev && ev.data ? ev.data : null;
                  if (!msg || typeof msg !== 'object') return;
                  if (msg.type === 'seek' && Number.isFinite(msg.t)) {{
                    try {{ video.currentTime = Math.max(0, msg.t); }} catch (e) {{}}
                  }}
                }};
              }}
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

