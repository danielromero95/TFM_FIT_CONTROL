"""Runtime helpers for configuring the host environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_environment() -> None:
    """Apply runtime tweaks required by TensorFlow/MediaPipe on import.

    The Streamlit application historically tweaked a couple of environment
    settings at import time. Centralise those tweaks here so they can be reused
    from different entry points while keeping ``src.app`` lean.
    """

    _configure_windows_dll_resolution()
    _tame_noisy_logs()


def _configure_windows_dll_resolution() -> None:
    if not sys.platform.startswith("win"):
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    dll_dir = Path(conda_prefix) / "Library" / "bin"
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


def _tame_noisy_logs() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")

    try:
        from absl import logging as absl_logging
    except Exception:
        return

    absl_logging.set_verbosity(absl_logging.ERROR)
