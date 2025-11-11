from __future__ import annotations

# Re-export public counting API for convenience
from .repetition_counter import count_repetitions_with_config, CountingDebugInfo

__all__ = ["count_repetitions_with_config", "CountingDebugInfo"]
