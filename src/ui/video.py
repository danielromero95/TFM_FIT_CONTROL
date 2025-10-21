"""Utilities for rendering uniformly sized videos in Streamlit."""

from __future__ import annotations

from typing import BinaryIO, Union

import streamlit as st


VideoData = Union[str, bytes, BinaryIO]


def render_uniform_video(
    data: VideoData,
    *,
    format: str | None = None,
    start_time: int = 0,
) -> None:
    """Render a video in a fixed-aspect container to ensure consistent sizing.

    Parameters
    ----------
    data:
        A string, bytes, or buffer-like object accepted by ``st.video``.
    format:
        Optional format hint for the video, forwarded to ``st.video``.
    start_time:
        Start playback offset in seconds, forwarded to ``st.video``.
    """

    container = st.container()
    container.markdown(
        """
        <div class="uniform-video-wrapper">
          <div class="uniform-video-aspect">
        """,
        unsafe_allow_html=True,
    )
    container.video(data, format=format, start_time=start_time)
    container.markdown(
        """
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

