"""
Configuration package for the Gym Performance Analyzer.

This __init__.py file conveniently exports the main configuration models
and loading utilities from their respective modules, allowing for easy
imports like:

    from src.config import Config, load_default
"""

from .models import (
    Config,
    PoseConfig,
    VideoConfig,
    CountingConfig,
    FaultConfig,
    DebugConfig,
    OutputConfig
)
from .utils import (
    load_default,
    from_yaml
)
from .constants import (
    APP_NAME,
    PROJECT_ROOT,
    VIDEO_EXTENSIONS
)
from .visualization import (
    POSE_CONNECTIONS,
    LANDMARK_COLOR,
    CONNECTION_COLOR
)

# This makes `from src.config import *` work cleanly, exporting only the key items.
__all__ = [
    # Models
    "Config",
    "PoseConfig",
    "VideoConfig",
    "CountingConfig",
    "FaultConfig",
    "DebugConfig",
    "OutputConfig",
    
    # Utilities
    "load_default",
    "from_yaml",
    
    # Constants
    "APP_NAME",
    "PROJECT_ROOT",
    "VIDEO_EXTENSIONS",
    
    # Visualization
    "POSE_CONNECTIONS",
    "LANDMARK_COLOR",
    "CONNECTION_COLOR",
]
