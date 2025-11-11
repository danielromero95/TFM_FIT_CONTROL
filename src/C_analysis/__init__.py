"""Paquete que agrupa la lógica modularizada del servicio de análisis."""

from .pipeline import _prepare_output_paths, run_full_pipeline_in_memory, run_pipeline
from .repetition_counter import CountingDebugInfo, count_repetitions_with_config
from .sampling import (
    SamplingPlan,
    compute_sample_rate,
    make_sampling_plan,
    normalize_detection,
    open_video_cap,
    read_info_and_initial_sampling,
)

__all__ = [
    "run_pipeline",
    "run_full_pipeline_in_memory",
    "_prepare_output_paths",
    "SamplingPlan",
    "make_sampling_plan",
    "compute_sample_rate",
    "normalize_detection",
    "open_video_cap",
    "read_info_and_initial_sampling",
    "count_repetitions_with_config",
    "CountingDebugInfo",
]
