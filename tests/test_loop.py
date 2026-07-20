import numpy as np
import pytest

from audio2instrument.loop import find_loop_points


def test_find_loop_points_on_periodic_signal() -> None:
    sample_rate = 4000
    t = np.arange(sample_rate * 2) / sample_rate
    samples = np.sin(2 * np.pi * 100 * t) * np.exp(-0.1 * t)
    points = find_loop_points(samples, sample_rate, search_start=0.4, search_end=1.5, min_duration=0.3, max_duration=0.7)
    assert round((points.end - points.start) / (sample_rate / 100)) == pytest.approx((points.end - points.start) / (sample_rate / 100), abs=0.1)
    assert points.start < points.end
    assert points.crossfade > 0
