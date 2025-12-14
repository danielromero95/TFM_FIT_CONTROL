from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

from .constants import (
    EPS,
    VIEW_MIN_RELIABLE_FRAMES,
    VIEW_TH_HI,
    VIEW_TH_LO,
    W_VISDELTA,
    W_WIDTHNORM,
    W_ZDELTA,
)
from .stats import safe_nanmedian
from .types import ViewResult

logger = logging.getLogger(__name__)


def classify_view(
    shoulder_yaw: np.ndarray | None,
    shoulder_z_delta: np.ndarray | None,
    shoulder_width: np.ndarray | None,
    ankle_width: np.ndarray | None,  # noqa: ARG001 - mantenemos la firma para compatibilidad
    *,
    reliability: Mapping[str, Any] | None = None,
) -> ViewResult:
    """Devuelve una etiqueta estable (frontal/lateral) con heurísticas robustas."""

    # Normalizamos entradas y enmascaramos por fiabilidad de fotograma
    _ = _to_array(shoulder_yaw)
    z_delta = _to_array(shoulder_z_delta)
    width = _to_array(shoulder_width)
    rel_mask = _reliable_mask(reliability, width.size, fallback=width)

    width_masked = _mask_invalid(width, rel_mask)
    z_masked = _mask_invalid(z_delta, rel_mask)
    vis_delta = _visibility_delta(reliability, rel_mask)

    width_med, width_mad = _median_and_mad(width_masked)
    z_med, z_mad = _median_and_mad(z_masked)
    vis_med, vis_mad = _median_and_mad(vis_delta)

    lateral_score = _lateral_score(width_med, z_med, vis_med, width_mad, z_mad, vis_mad)

    reliable_frames = int(rel_mask.sum())
    total_frames_sampled = int(_get_reliability_value(reliability, "total_frames_sampled", default=width.size))
    reliability_ratio = reliable_frames / max(float(total_frames_sampled), 1.0)

    label, base_confidence = _decide_view(lateral_score, reliable_frames)
    dispersion = float(np.nanmean([width_mad, z_mad, vis_mad])) if not np.isnan(width_mad) else z_mad
    stability_factor = 1.0 / (1.0 + max(dispersion, 0.0)) if np.isfinite(dispersion) else 1.0
    confidence = float(np.clip(base_confidence * reliability_ratio * stability_factor, 0.0, 1.0))

    side = None
    if label == "side":
        side = _infer_side(reliability, rel_mask)
        if base_confidence < 0.5:
            confidence *= 0.8

    stats = {
        "width_med": width_med,
        "width_mad": width_mad,
        "z_med": z_med,
        "z_mad": z_mad,
        "vis_med": vis_med,
        "vis_mad": vis_mad,
        "lateral_score": lateral_score,
        "reliable_frames": reliable_frames,
        "total_frames_sampled": total_frames_sampled,
        "reliability_ratio": reliability_ratio,
    }

    votes = {"front": 0, "side": 0}
    if np.isfinite(lateral_score) and reliable_frames > 0:
        side_votes = int(round(lateral_score * reliable_frames))
        front_votes = int(round((1.0 - lateral_score) * reliable_frames))
        votes["side"] = max(side_votes, 0)
        votes["front"] = max(front_votes, 0)

    logger.info(
        "view=%s lateral_score=%.3f conf=%.2f rel=%d/%d width_med=%.3f(%.3f) z_med=%.3f(%.3f) vis_med=%.3f side=%s",
        label,
        lateral_score,
        confidence,
        reliable_frames,
        total_frames_sampled,
        width_med,
        width_mad,
        z_med,
        z_mad,
        vis_med,
        side,
    )

    return ViewResult(
        label=label,
        scores={"lateral": lateral_score},
        votes=votes,
        confidence=confidence,
        lateral_score=lateral_score,
        stats=stats,
        side=side,
    )


def _lateral_score(
    width_med: float,
    z_med: float,
    vis_med: float,
    width_mad: float,
    z_mad: float,
    vis_mad: float,
) -> float:
    if not np.isfinite(width_med) and not np.isfinite(z_med):
        return float("nan")

    width_component = 0.0
    if np.isfinite(width_med):
        width_component = np.clip((0.65 - width_med) / 0.65, 0.0, 1.0)
        if np.isfinite(width_mad) and width_mad > 0:
            width_component *= 1.0 / (1.0 + width_mad)

    z_component = 0.0
    if np.isfinite(z_med):
        z_component = np.clip(z_med / 0.12, 0.0, 1.0)
        if np.isfinite(z_mad) and z_mad > 0:
            z_component *= 1.0 / (1.0 + z_mad)

    vis_component = 0.0
    if np.isfinite(vis_med):
        vis_component = np.clip(vis_med, 0.0, 1.0)
        if np.isfinite(vis_mad) and vis_mad > 0:
            vis_component *= 1.0 / (1.0 + vis_mad)

    weighted = (
        W_ZDELTA * z_component
        + W_WIDTHNORM * width_component
        + W_VISDELTA * vis_component
    )
    denom = max(W_ZDELTA + W_WIDTHNORM + W_VISDELTA, EPS)
    return float(np.clip(weighted / denom, 0.0, 1.0))


def _decide_view(lateral_score: float, reliable_frames: int) -> tuple[str, float]:
    if reliable_frames < VIEW_MIN_RELIABLE_FRAMES or not np.isfinite(lateral_score):
        return "unknown", 0.0

    if lateral_score >= VIEW_TH_HI:
        margin = (lateral_score - VIEW_TH_HI) / max(1.0 - VIEW_TH_HI, EPS)
        confidence = 0.5 + 0.5 * margin
        return "side", float(np.clip(confidence, 0.0, 1.0))
    if lateral_score <= VIEW_TH_LO:
        margin = (VIEW_TH_LO - lateral_score) / max(VIEW_TH_LO, EPS)
        confidence = 0.5 + 0.5 * margin
        return "front", float(np.clip(confidence, 0.0, 1.0))

    margin = (min(lateral_score - VIEW_TH_LO, VIEW_TH_HI - lateral_score)) / max(VIEW_TH_HI - VIEW_TH_LO, EPS)
    confidence = 0.35 * np.clip(1.0 - margin, 0.0, 1.0)
    label = "side" if lateral_score >= 0.5 else "front"
    return label, float(confidence)


def _visibility_delta(reliability: Mapping[str, Any] | None, mask: np.ndarray) -> np.ndarray:
    left = _to_array(_get_reliability_value(reliability, "shoulder_vis_left", default=[]))
    right = _to_array(_get_reliability_value(reliability, "shoulder_vis_right", default=[]))
    size = max(left.size, right.size, mask.size)
    left = _resize(left, size)
    right = _resize(right, size)
    delta = np.abs(left - right)
    return _mask_invalid(delta, mask)


def _infer_side(reliability: Mapping[str, Any] | None, mask: np.ndarray) -> str | None:
    shoulder_sign = _mask_invalid(
        _to_array(_get_reliability_value(reliability, "shoulder_z_sign", default=[])), mask
    )
    hip_sign = _mask_invalid(_to_array(_get_reliability_value(reliability, "hip_z_sign", default=[])), mask)

    for series in (shoulder_sign, hip_sign):
        if series.size == 0:
            continue
        sign_med = safe_nanmedian(series)
        if np.isfinite(sign_med) and abs(sign_med) > EPS:
            return "right" if sign_med > 0 else "left"
    return None


def _median_and_mad(arr: np.ndarray) -> tuple[float, float]:
    med = safe_nanmedian(arr)
    if not np.isfinite(med):
        return med, float("nan")
    deviations = np.abs(arr - med)
    mad = safe_nanmedian(deviations)
    return med, float(mad)


def _to_array(series: np.ndarray | None) -> np.ndarray:
    if series is None:
        return np.array([], dtype=float)
    return np.asarray(series, dtype=float)


def _reliable_mask(reliability: Mapping[str, Any] | None, length: int, *, fallback: np.ndarray | None = None) -> np.ndarray:
    if reliability is None:
        if fallback is not None and fallback.size:
            return np.isfinite(_resize(fallback, length))
        return np.zeros(length, dtype=bool)
    mask = np.asarray(reliability.get("frame_reliability", []), dtype=bool)
    if mask.size < length:
        pad = np.zeros(length - mask.size, dtype=bool)
        return np.concatenate([mask, pad])
    if mask.size > length:
        return mask[:length]
    return mask


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.size < size:
        pad = np.full(size - arr.size, np.nan, dtype=float)
        return np.concatenate([arr, pad])
    if arr.size > size:
        return arr[:size]
    return arr


def _resize_bool(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.size < size:
        pad = np.zeros(size - mask.size, dtype=bool)
        return np.concatenate([mask.astype(bool), pad])
    if mask.size > size:
        return mask.astype(bool)[:size]
    return mask.astype(bool)


def _mask_invalid(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = _resize(arr, max(arr.size, mask.size))
    mask = _resize_bool(mask, arr.size)
    out = np.array(arr, dtype=float, copy=True)
    invalid = ~mask
    out[invalid] = np.nan
    return out


def _get_reliability_value(reliability: Mapping[str, Any] | None, key: str, *, default: Any = np.nan) -> Any:
    if reliability is None:
        return default
    return reliability.get(key, default)


__all__ = ["classify_view"]
