# src/ui/video.py

"""
Utilities for rendering uniformly sized videos in Streamlit using
custom HTML and CSS for 100% layout control.
"""

from __future__ import annotations
import base64
from pathlib import Path
from typing import Union
import streamlit as st

VideoPath = Union[str, Path]

def _get_video_b64(video_path: VideoPath) -> str:
    """Reads video file from a path and returns a base64 encoded string."""
    try:
        video_bytes = Path(video_path).read_bytes()
        b64_string = base64.b64encode(video_bytes).decode()
        return f"data:video/mp4;base64,{b64_string}"
    except Exception as e:
        # Log the error for debugging, but don't crash the app
        print(f"Error encoding video '{video_path}': {e}")
        return ""


def render_uniform_video(video_path: VideoPath) -> None:
    """
    Renders a video in a fixed-aspect container using a custom HTML
    <video> tag, bypassing st.video() for full CSS control.
    """
    video_data_uri = _get_video_b64(video_path)
    if not video_data_uri:
        st.error(f"Could not load video: {video_path}")
        return

    # Render the custom HTML component. This structure is self-contained
    # and will be styled by the rules in `src/ui/styles.css`.
    st.markdown(
        f"""
        <div class="video-viewport-container">
            <video width="100%" height="100%" controls autoplay muted loop playsinline>
                <source src="{video_data_uri}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """,
        unsafe_allow_html=True,
    )
