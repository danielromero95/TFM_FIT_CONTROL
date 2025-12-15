"""Utilities to build downloadable debug bundles for offline troubleshooting."""

from __future__ import annotations

import io
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd

from src import config
from src.pipeline_data import Report
from src.utils.json_safety import json_safe


def _default_run_info() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "timestamp_utc": now.isoformat(),
    }


def _metrics_summary(df_metrics: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df_metrics is None:
        return {"row_count": 0, "columns": []}

    summary: Dict[str, Any] = {
        "row_count": int(len(df_metrics)),
        "columns": list(df_metrics.columns),
    }

    if df_metrics.empty:
        return summary

    describe = df_metrics.describe(include="all").to_dict()
    nan_ratio: Dict[str, Any] = {}
    for col in df_metrics.columns:
        col_numeric = pd.to_numeric(df_metrics[col], errors="coerce")
        if len(col_numeric):
            invalid_fraction = 1.0 - float(np.isfinite(col_numeric).mean())
            nan_ratio[col] = invalid_fraction
        else:
            nan_ratio[col] = None

    summary["describe"] = json_safe(describe)
    summary["nan_ratio"] = json_safe(nan_ratio)
    return summary


def _embed_configs_if_missing(
    *,
    debug_report: Dict[str, Any],
    config_used_path: Optional[Path],
    effective_config_path: Optional[Path],
    cfg: config.Config,
) -> Dict[str, str]:
    embedded: Dict[str, Any] = {}
    files: Dict[str, str] = {}

    if config_used_path and config_used_path.exists():
        files[config_used_path.name] = config_used_path.read_text(encoding="utf-8")
    else:
        embedded["config_used"] = cfg.to_serializable_dict()

    if effective_config_path and effective_config_path.exists():
        files[effective_config_path.name] = effective_config_path.read_text(encoding="utf-8")
    else:
        embedded["config_effective"] = cfg.to_serializable_dict()

    if embedded:
        debug_report.setdefault("configs", {}).update(json_safe(embedded))

    return files


def build_debug_report_bundle(
    report: Report,
    *,
    video_path: str,
    cfg: config.Config,
    df_metrics: Optional[pd.DataFrame] = None,
    extra: Optional[Dict[str, Any]] = None,
    metrics_csv_max_rows: int = 50_000,
) -> Tuple[bytes, str]:
    """Construct a ZIP bundle with exhaustive debug information.

    Returns the in-memory bytes and the suggested filename.
    """

    df_metrics = df_metrics if df_metrics is not None else report.metrics
    base_summary = deepcopy(report.debug_summary) if report.debug_summary else {}
    debug_report: Dict[str, Any] = json_safe(base_summary) if base_summary else {}

    debug_report.setdefault("run_info", _default_run_info())
    debug_report.setdefault("config_sha1", getattr(report.stats, "config_sha1", None))
    debug_report.setdefault("warnings", list(getattr(report.stats, "warnings", []) or []))

    video_path_obj = Path(video_path)
    input_video = debug_report.get("input_video", {}) or {}
    input_video.setdefault("name", video_path_obj.name)
    input_video.setdefault("extension", video_path_obj.suffix)
    try:
        size_bytes = int(video_path_obj.stat().st_size)
    except OSError:
        size_bytes = None
    input_video.setdefault("size_bytes", size_bytes)
    debug_report["input_video"] = input_video

    if extra:
        debug_report.update(json_safe(extra))

    debug_report["metrics_summary"] = _metrics_summary(df_metrics)

    embedded_files = _embed_configs_if_missing(
        debug_report=debug_report,
        config_used_path=report.stats.config_path,
        effective_config_path=report.effective_config_path,
        cfg=cfg,
    )

    buffer = io.BytesIO()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    zip_name = f"debug_report_{video_path_obj.stem}_{timestamp}.zip"

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        debug_json = json.dumps(json_safe(debug_report), ensure_ascii=False, indent=2)
        zip_file.writestr("debug_report.json", debug_json)

        for name, content in embedded_files.items():
            zip_file.writestr(name, content)

        if df_metrics is not None and len(df_metrics) <= metrics_csv_max_rows:
            csv_content = df_metrics.to_csv(index=False)
            if csv_content:
                zip_file.writestr(f"{video_path_obj.stem}_metrics.csv", csv_content)

    return buffer.getvalue(), zip_name

