from src.ui.state.model import AppState
from src.ui.steps.results.results import _metrics_chart_key, _metrics_run_token


def test_metrics_run_token_prefers_metrics_session_dir(tmp_path):
    session_dir = tmp_path / "session_abc"
    session_dir.mkdir()
    metrics_path = session_dir / "metrics.csv"
    metrics_path.write_text("frame_idx,value\n0,1\n", encoding="utf-8")

    state = AppState(metrics_path=str(metrics_path))

    token = _metrics_run_token(state, None)

    assert token == "session_abc"


def test_metrics_chart_key_changes_with_video(tmp_path):
    first_state = AppState(video_path=str(tmp_path / "first-video.mp4"))
    second_state = AppState(video_path=str(tmp_path / "second-video.mp4"))

    first_token = _metrics_run_token(first_state, None)
    second_token = _metrics_run_token(second_state, None)

    assert first_token != second_token
    assert _metrics_chart_key(first_token) != _metrics_chart_key(second_token)
