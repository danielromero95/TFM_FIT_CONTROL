"""Helpers for building stable sync tokens for video/metrics components."""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

SESSION_TOKEN_KEY = "metrics_sync_run_token"
SESSION_ANCHOR_KEY = "metrics_sync_run_anchor"


def _safe_parent_name(candidate: object | None) -> str | None:
    if not candidate:
        return None
    try:
        parent = Path(candidate).expanduser().parent
    except Exception:
        return None
    return parent.name or None


def _path_fingerprint(candidate: object | None) -> str | None:
    if not candidate:
        return None
    try:
        path = Path(candidate).expanduser()
    except Exception:
        return None
    stem = path.stem or None
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    if stem:
        return f"{stem}-{digest}"
    return digest or None


def _session_name_from_paths(*candidates: object | None) -> str | None:
    for candidate in candidates:
        name = _safe_parent_name(candidate)
        if name:
            return name
    return None


def _normalized_path(candidate: object | None) -> str | None:
    if not candidate:
        return None
    try:
        return str(Path(candidate).expanduser())
    except Exception:
        try:
            return str(candidate)
        except Exception:
            return None


def _derive_run_token(state, stats) -> str:
    session_name = _session_name_from_paths(
        getattr(state, "metrics_path", None),
        getattr(stats, "config_path", None),
    )
    if session_name:
        return session_name

    for candidate in (
        getattr(state, "metrics_path", None),
        getattr(state, "video_path", None),
    ):
        fingerprint = _path_fingerprint(candidate)
        if fingerprint:
            return fingerprint

    config_sha = getattr(stats, "config_sha1", None)
    frames_val = getattr(stats, "frames", None)
    if config_sha:
        return f"{config_sha}-{frames_val}" if frames_val is not None else config_sha

    raw_parts = [
        str(part)
        for part in (
            getattr(state, "metrics_path", None),
            getattr(state, "video_path", None),
            config_sha,
            frames_val,
        )
        if part is not None
    ]
    if not raw_parts:
        raw_parts.append("anonymous")

    digest = hashlib.sha1("|".join(raw_parts).encode("utf-8")).hexdigest()[:12]
    return f"run-{digest}"


def _anchor_signature(state, stats):
    metrics_path = _normalized_path(getattr(state, "metrics_path", None))
    video_path = _normalized_path(getattr(state, "video_path", None))
    if metrics_path or video_path:
        return ("paths", metrics_path, video_path)

    config_sha = getattr(stats, "config_sha1", None)
    frames_val = getattr(stats, "frames", None)
    if config_sha:
        suffix = f"-{frames_val}" if frames_val is not None else ""
        return ("config", f"{config_sha}{suffix}")

    return None


def _anchors_share_video(anchor, cached_anchor) -> bool:
    if not anchor or not cached_anchor:
        return False
    if anchor[0] != "paths" or cached_anchor[0] != "paths":
        return False
    _, _, anchor_video = anchor
    _, _, cached_video = cached_anchor
    return bool(anchor_video) and anchor_video == cached_video


def metrics_run_token(state, stats=None) -> str:
    """Return a stable token for the current run and persist it in session state."""

    stats = stats or getattr(state, "report", None) and getattr(getattr(state, "report", None), "stats", None)
    anchor = _anchor_signature(state, stats)

    cached_anchor = st.session_state.get(SESSION_ANCHOR_KEY)
    cached_token = st.session_state.get(SESSION_TOKEN_KEY)
    if cached_token:
        if anchor and cached_anchor and (anchor == cached_anchor or _anchors_share_video(anchor, cached_anchor)):
            st.session_state[SESSION_ANCHOR_KEY] = anchor or cached_anchor
            return cached_token
        if anchor is None:
            return cached_token

    token = _derive_run_token(state, stats)
    st.session_state[SESSION_TOKEN_KEY] = token
    if anchor:
        st.session_state[SESSION_ANCHOR_KEY] = anchor
    return token


def metrics_chart_key(run_token: str) -> str:
    """Build the Streamlit chart key for the metrics widget."""

    return f"results_video_metrics_sync_{run_token}"


def sync_channel_for_run(run_token: str) -> str:
    """Return the BroadcastChannel name for video/metrics sync."""

    return f"vmx-sync-{run_token}"
