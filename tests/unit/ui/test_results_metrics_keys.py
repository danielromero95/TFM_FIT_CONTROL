import streamlit as st

from src.ui.metrics_sync.run_tokens import sync_channel_for_run
from src.ui.state.model import AppState
from src.ui.steps.results.results import _metrics_chart_key, _metrics_run_token


def test_metrics_run_token_prefers_metrics_session_dir(tmp_path):
    st.session_state.clear()
    session_dir = tmp_path / "session_abc"
    session_dir.mkdir()
    metrics_path = session_dir / "metrics.csv"
    metrics_path.write_text("frame_idx,value\n0,1\n", encoding="utf-8")

    state = AppState(metrics_path=str(metrics_path))

    token = _metrics_run_token(state, None)

    assert token == "session_abc"


def test_metrics_chart_key_changes_with_video(tmp_path):
    st.session_state.clear()
    first_state = AppState(video_path=str(tmp_path / "first-video.mp4"))
    second_state = AppState(video_path=str(tmp_path / "second-video.mp4"))

    first_token = _metrics_run_token(first_state, None)
    second_token = _metrics_run_token(second_state, None)

    assert first_token != second_token
    assert _metrics_chart_key(first_token) != _metrics_chart_key(second_token)


def test_metrics_run_token_is_stable_across_calls(tmp_path):
    st.session_state.clear()
    metrics_path = tmp_path / "session_xyz" / "metrics.csv"
    metrics_path.parent.mkdir()
    metrics_path.write_text("frame_idx,value\n0,1\n", encoding="utf-8")

    state = AppState(metrics_path=str(metrics_path))

    first = _metrics_run_token(state, None)
    second = _metrics_run_token(state, None)

    assert first == second


def test_metrics_run_token_remains_when_metrics_added(tmp_path):
    st.session_state.clear()
    video_path = tmp_path / "videos" / "demo.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"")

    state = AppState(video_path=str(video_path))

    initial = _metrics_run_token(state, None)

    metrics_path = tmp_path / "session" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("frame_idx,value\n0,1\n", encoding="utf-8")

    updated = _metrics_run_token(
        AppState(video_path=str(video_path), metrics_path=str(metrics_path)), None
    )

    assert initial == updated


def test_metrics_run_token_does_not_fallback_to_plain_run():
    st.session_state.clear()
    token = _metrics_run_token(AppState(), None)

    assert token != "run"
    assert token


def test_metrics_run_token_differs_for_same_name_files(tmp_path):
    st.session_state.clear()

    first_video = tmp_path / "a" / "video.mp4"
    second_video = tmp_path / "b" / "video.mp4"
    first_video.parent.mkdir(parents=True, exist_ok=True)
    second_video.parent.mkdir(parents=True, exist_ok=True)
    first_video.write_bytes(b"1")
    second_video.write_bytes(b"2")

    first_token = _metrics_run_token(AppState(video_path=str(first_video)), None)
    second_token = _metrics_run_token(AppState(video_path=str(second_video)), None)

    assert first_token != second_token


def test_sync_channel_matches_chart_token(tmp_path):
    st.session_state.clear()
    state = AppState(video_path=str(tmp_path / "video.mp4"))

    run_token = _metrics_run_token(state, None)

    assert sync_channel_for_run(run_token) == f"vmx-sync-{run_token}"
    assert _metrics_chart_key(run_token) == f"results_video_metrics_sync_{run_token}"
