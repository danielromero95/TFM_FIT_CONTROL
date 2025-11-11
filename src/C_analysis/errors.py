"""Domain-specific exceptions for the analysis services."""


class PipelineError(Exception):
    """Base exception for fatal pipeline failures."""


class VideoOpenError(PipelineError):
    """Raised when a video file cannot be opened for reading."""


class NoFramesExtracted(PipelineError):
    """Raised when frame extraction yields no frames."""
