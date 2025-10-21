# src/ui/video.py

"""Custom HTML helpers for rendering videos with consistent sizing."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Optional, Union

import streamlit as st
import streamlit.components.v1 as components

VideoPath = Union[str, Path]


@st.cache_data(show_spinner=False)
def _data_uri(path: str, *, mtime: float) -> str:
    """Return a base64 data URI for the provided video path."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "video/mp4"
    with open(path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def render_uniform_video(
    video_path: VideoPath,
    *,
    max_width: int = 720,
    key: Optional[str] = None,
) -> None:
    """Render a video inside a centered, 16:9 container using custom HTML."""
    path_obj = Path(video_path)
    path = str(path_obj)
    try:
        stat = path_obj.stat()
        src = _data_uri(path, mtime=stat.st_mtime)
    except OSError as exc:
        st.error(f"Could not load video: {video_path}")
        print(f"Error encoding video '{video_path}': {exc}")
        return

    iframe_height = int(max_width * 9 / 16) + 24
    html = f"""
    <div style="display:flex;justify-content:center;">
      <div style="width:100%;max-width:{max_width}px">
        <div style="position:relative;width:100%;padding-top:56.25%;background:#000;border-radius:12px;overflow:hidden">
          <video
            src="{src}"
            controls
            playsinline
            preload="metadata"
            style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000"
          ></video>
        </div>
      </div>
    </div>
    """
    components.html(html, height=iframe_height, scrolling=False, key=key)
