"""
Default tunable parameters and settings for the analysis pipeline and GUI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_environment() -> None:
    """Configure logging verbosity and Windows DLL lookup paths."""

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["GLOG_minloglevel"] = "2"

    try:
        from absl import logging as absl_logging  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        absl_logging.set_verbosity(absl_logging.ERROR)

    if sys.platform.startswith("win"):
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            dll_dir = Path(conda_prefix) / "Library" / "bin"
            if dll_dir.exists():
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


# --- PIPELINE PARAMETERS ---
MODEL_COMPLEXITY = 1
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_HEIGHT = 256

# --- COUNTING PARAMETERS (legacy) ---
SQUAT_HIGH_THRESH = 160.0
SQUAT_LOW_THRESH = 100.0
PEAK_PROMINENCE = 10  # Prominence used by the peak detector
PEAK_DISTANCE = 15    # Minimum distance in frames between repetitions

# --- DEFAULT GUI / APP VALUES ---
DEFAULT_SAMPLE_RATE = 3
DEFAULT_ROTATION = "0"
DEFAULT_USE_CROP = True
DEFAULT_GENERATE_VIDEO = True
DEFAULT_DEBUG_MODE = True
DEFAULT_DARK_MODE = False
