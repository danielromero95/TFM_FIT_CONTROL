"""Utilities to guarantee strict JSON-serializable payloads."""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Recursively convert ``value`` into a JSON-serializable structure.

    - NumPy scalars/arrays are converted to Python types/lists.
    - ``Path`` and ``Enum`` become strings.
    - Non-finite floats (NaN/Inf) are converted to ``None``.
    - Mapping keys are stringified to avoid invalid JSON objects.
    """

    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    if isinstance(value, np.generic):
        return json_safe(value.item())

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time()).isoformat()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    try:
        if isinstance(value, float) and not math.isfinite(value):
            return None
    except Exception:
        pass

    return value

