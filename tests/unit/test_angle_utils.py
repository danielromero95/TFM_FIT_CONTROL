from __future__ import annotations

import numpy as np

from src.utils.angles import apply_warmup_mask, maybe_convert_radians_to_degrees, suppress_spikes


def test_radians_are_converted_but_degrees_are_not():
    radians = np.array([1.4, 1.6, 1.5])
    converted, was_rad = maybe_convert_radians_to_degrees(radians)
    assert was_rad
    assert np.allclose(converted, np.degrees(radians))

    degrees = np.array([82.0, 84.0, 79.0])
    untouched, was_rad = maybe_convert_radians_to_degrees(degrees)
    assert not was_rad
    assert np.allclose(untouched, degrees)


def test_spikes_and_warmup_are_removed():
    values = np.array([60.0, 62.0, 150.0, 63.0, 61.0])

    cleaned, spikes = suppress_spikes(values, threshold_deg=40.0)
    assert spikes == 1
    assert np.isnan(cleaned[2])

    warmed, applied = apply_warmup_mask(cleaned, warmup_frames=2)
    assert applied == 2
    assert np.isnan(warmed[:2]).all()
