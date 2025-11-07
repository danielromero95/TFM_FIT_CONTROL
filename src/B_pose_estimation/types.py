"""Common typing helpers and lightweight data containers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TypeVar

import numpy as np

LandmarkLike = Mapping[str, float]


@dataclass(frozen=True)
class Landmark(Mapping[str, float]):
    """Mapping-compatible pose landmark representation."""

    x: float
    y: float
    z: float
    visibility: float

    def __getitem__(self, key: str) -> float:  # type: ignore[override]
        if key == "x":
            return float(self.x)
        if key == "y":
            return float(self.y)
        if key == "z":
            return float(self.z)
        if key == "visibility":
            return float(self.visibility)
        raise KeyError(key)

    def __iter__(self):  # type: ignore[override]
        yield from ("x", "y", "z", "visibility")

    def __len__(self) -> int:  # type: ignore[override]
        return 4

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z), "visibility": float(self.visibility)}

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "Landmark":
        return cls(
            x=float(data.get("x", np.nan)),
            y=float(data.get("y", np.nan)),
            z=float(data.get("z", np.nan)),
            visibility=float(data.get("visibility", np.nan)),
        )


PoseFrame = Optional[List[Landmark]]
PoseSequence = Sequence[PoseFrame]


@dataclass
class PoseResult:
    """Container returned by pose estimators."""

    landmarks: PoseFrame
    annotated_image: Optional[np.ndarray]
    crop_box: Optional[Sequence[int]]

    def as_tuple(self) -> Tuple[PoseFrame, Optional[np.ndarray], Optional[Sequence[int]]]:
        return self.landmarks, self.annotated_image, self.crop_box


T = TypeVar("T")


def ensure_sequence(sequence: Iterable[T]) -> List[T]:
    """Materialise ``sequence`` preserving order."""

    return list(sequence)


__all__ = ["Landmark", "PoseFrame", "PoseResult", "PoseSequence", "ensure_sequence"]
