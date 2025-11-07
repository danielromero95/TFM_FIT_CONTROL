"""Filtering and interpolation of pose landmarks."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from ..constants import LANDMARK_COUNT
from ..types import Landmark

logger = logging.getLogger(__name__)


def filter_and_interpolate_landmarks(
    df_raw: pd.DataFrame,
    min_confidence: float = 0.5,
    *,
    vis_hysteresis_low: float = 0.35,
    vis_hysteresis_high: float = 0.55,
    vis_hang: int = 3,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Filter landmarks below ``min_confidence`` and interpolate gaps."""

    logger.info("Filtering and interpolating %d landmark frames.", len(df_raw))
    n_frames = len(df_raw)
    crop_coords = (
        df_raw[["crop_x1", "crop_y1", "crop_x2", "crop_y2"]].to_numpy()
        if {"crop_x1", "crop_y1", "crop_x2", "crop_y2"}.issubset(df_raw.columns)
        else None
    )
    if n_frames == 0:
        return np.array([], dtype=object), crop_coords

    n_points = LANDMARK_COUNT
    cols_x = [f"x{idx}" for idx in range(n_points)]
    cols_y = [f"y{idx}" for idx in range(n_points)]
    cols_z = [f"z{idx}" for idx in range(n_points)]
    cols_v = [f"v{idx}" for idx in range(n_points)]

    xs = df_raw[cols_x].to_numpy(dtype=float, copy=True)
    ys = df_raw[cols_y].to_numpy(dtype=float, copy=True)
    zs = df_raw[cols_z].to_numpy(dtype=float, copy=True)
    vs = df_raw[cols_v].to_numpy(dtype=float, copy=False)

    finite_xy = np.isfinite(xs) & np.isfinite(ys)

    def _apply_visibility_hysteresis(column: np.ndarray) -> np.ndarray:
        n = column.shape[0]
        visible = np.zeros(n, dtype=bool)
        state = False
        seen_high = False
        hang_remaining = 0

        for idx, raw_val in enumerate(column):
            val = float(raw_val) if np.isfinite(raw_val) else 0.0
            next_state = state

            if val >= vis_hysteresis_high:
                seen_high = True
                next_state = True
                hang_remaining = max(0, vis_hang)
            elif seen_high:
                if val < vis_hysteresis_low:
                    if state and hang_remaining > 0:
                        hang_remaining -= 1
                    else:
                        next_state = False
                        hang_remaining = 0
                elif not state and val >= min_confidence:
                    next_state = True
                    hang_remaining = max(0, vis_hang)
            else:
                next_state = val >= min_confidence

            state = next_state
            visible[idx] = state

        return visible

    visibility_mask = np.zeros_like(xs, dtype=bool)
    for column_index in range(xs.shape[1]):
        col_mask = _apply_visibility_hysteresis(vs[:, column_index])
        visibility_mask[:, column_index] = col_mask & finite_xy[:, column_index]

    xs[~visibility_mask] = np.nan
    ys[~visibility_mask] = np.nan
    zs[~visibility_mask] = np.nan

    def _interp_columns(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        if out.size == 0:
            return out
        idx = np.arange(out.shape[0])
        for column_index in range(out.shape[1]):
            column = out[:, column_index]
            valid = np.isfinite(column)
            if valid.sum() >= 2:
                out[:, column_index] = np.interp(idx, idx[valid], column[valid])
        return out

    xs_interp = _interp_columns(xs)
    ys_interp = _interp_columns(ys)
    zs_interp = _interp_columns(zs)

    vis_output = np.where(
        visibility_mask,
        np.where(np.isfinite(vs), vs, min_confidence),
        0.0,
    )

    def _build_landmark(x: float, y: float, z: float, v: float) -> Landmark:
        return Landmark(x=float(x), y=float(y), z=float(z), visibility=float(v))

    vectorized_builder = np.vectorize(_build_landmark, otypes=[object])
    landmarks_array = vectorized_builder(xs_interp, ys_interp, zs_interp, vis_output)
    frames_list = landmarks_array.tolist()
    filtered_sequence = np.empty(n_frames, dtype=object)
    filtered_sequence[:] = frames_list
    return filtered_sequence, crop_coords


__all__ = ["filter_and_interpolate_landmarks"]
