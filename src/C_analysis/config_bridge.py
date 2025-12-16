"""Utilidades para mapear configuraciones legadas hacia el objeto `Config`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src import config


def apply_settings(cfg: config.Config, settings: Dict[str, Any]) -> None:
    """Trasladar un diccionario legado hacia las propiedades tipadas del `Config`."""

    if "output_dir" in settings:
        base_dir = Path(settings["output_dir"]).expanduser()
        cfg.output.base_dir = base_dir
        cfg.output.counts_dir = base_dir / "counts"
        cfg.output.poses_dir = base_dir / "poses"
    if "sample_rate" in settings:
        cfg.video.manual_sample_rate = int(settings["sample_rate"])
    if "rotate" in settings:
        cfg.pose.rotate = settings["rotate"]
    if "target_width" in settings:
        cfg.pose.target_width = int(settings["target_width"])
    if "target_height" in settings:
        cfg.pose.target_height = int(settings["target_height"])
    if "pose_target_fps_default" in settings:
        cfg.pose.target_fps_default = float(settings["pose_target_fps_default"])
    if "pose_target_fps_fallback" in settings:
        cfg.pose.target_fps_fallback = float(settings["pose_target_fps_fallback"])
    if "pose_target_size_fallback" in settings:
        cfg.pose.target_size_fallback = int(settings["pose_target_size_fallback"])
    if "pose_quality_fallback_threshold" in settings:
        cfg.pose.quality_fallback_threshold = float(settings["pose_quality_fallback_threshold"])
    if "use_crop" in settings:
        cfg.pose.use_crop = bool(settings["use_crop"])
    if "generate_debug_video" in settings:
        cfg.debug.generate_debug_video = bool(settings["generate_debug_video"])
    if "debug_mode" in settings:
        cfg.debug.debug_mode = bool(settings["debug_mode"])
    if "low_thresh" in settings:
        cfg.faults.low_thresh = float(settings["low_thresh"])
    if "high_thresh" in settings:
        cfg.faults.high_thresh = float(settings["high_thresh"])
    # ``fps_override`` se ignora intencionalmente: la pipeline depende del FPS del metadato.
