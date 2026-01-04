import os
import pytest

pytestmark = pytest.mark.manual


@pytest.mark.skipif(
    os.getenv("RUN_MANUAL_CV2_CHECK") != "1",
    reason="Manual OpenCV environment check (opt-in via RUN_MANUAL_CV2_CHECK=1).",
)
def test_cv2_video_smoke():
    cv2 = pytest.importorskip("cv2")

    assert getattr(cv2, "__version__", None), "OpenCV is installed but has no __version__"

    if hasattr(cv2, "getBuildInformation"):
        info = cv2.getBuildInformation()
        assert isinstance(info, str)
