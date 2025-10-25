from __future__ import annotations

import hashlib

import streamlit as st

from src.ui.state import Step, get_state, go_to, reset_state
from ..utils import ensure_video_path, step_container


def _upload_step() -> None:
    with step_container("upload"):
        st.markdown("### 1. Upload your video")
        uploaded = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv", "mpg", "mpeg", "wmv"],
            key="video_uploader",
            label_visibility="collapsed",
        )

        state = get_state()
        previous_token = state.active_upload_token

        if uploaded is not None:
            data_bytes = uploaded.getvalue()
            new_token = (
                uploaded.name,
                len(data_bytes),
                hashlib.md5(data_bytes).hexdigest(),
            )
            if previous_token != new_token:
                reset_state(preserve_upload=True)
                state = get_state()
                state.upload_data = {"name": uploaded.name, "bytes": data_bytes}
                state.upload_token = new_token
                state.active_upload_token = new_token
                ensure_video_path()
                state = get_state()
                if state.video_path:
                    go_to(Step.DETECT)
            else:
                state.active_upload_token = new_token
        else:
            uploader_state = state.video_uploader
            if (
                previous_token is not None
                and state.video_path
                and uploader_state in (None, "")
            ):
                reset_state()
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
