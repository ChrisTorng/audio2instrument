from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LoopPoints:
    start: int
    end: int
    crossfade: int
    score: float


def _boundary_cost(samples: np.ndarray, start: int, end: int, window: int) -> float:
    before = samples[start : start + window]
    after = samples[end : end + window]
    if len(before) != window or len(after) != window:
        return float("inf")
    before_rms = np.sqrt(np.mean(before * before) + 1e-15)
    after_rms = np.sqrt(np.mean(after * after) + 1e-15)
    normalized_before = before / before_rms
    normalized_after = after / after_rms
    waveform = np.mean((normalized_before - normalized_after) ** 2)
    slope = np.mean((np.diff(normalized_before) - np.diff(normalized_after)) ** 2)
    level = abs(np.log((before_rms + 1e-12) / (after_rms + 1e-12)))
    return float(waveform + 0.25 * slope + 0.1 * level)


def find_loop_points(
    samples: np.ndarray,
    sample_rate: int,
    *,
    search_start: float,
    search_end: float,
    min_duration: float = 0.25,
    max_duration: float = 0.8,
    crossfade: float = 0.03,
) -> LoopPoints:
    """Find phase-compatible loop boundaries inside a stable sample body."""
    values = np.asarray(samples, dtype=np.float64)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 <= search_start < search_end:
        raise ValueError("search range must be positive and ordered")
    if not 0 < min_duration <= max_duration:
        raise ValueError("loop duration range is invalid")

    first = round(search_start * sample_rate)
    last = min(len(values), round(search_end * sample_rate))
    min_length = round(min_duration * sample_rate)
    max_length = round(max_duration * sample_rate)
    crossfade_samples = max(1, round(crossfade * sample_rate))
    window = max(32, crossfade_samples)
    if last - first < min_length + window:
        raise ValueError("search range is too short for the requested loop")

    coarse_step = max(8, sample_rate // 1000)
    best: tuple[float, int, int] | None = None
    for start in range(first, last - min_length - window, coarse_step):
        maximum_end = min(last - window, start + max_length)
        for end in range(start + min_length, maximum_end + 1, coarse_step):
            cost = _boundary_cost(values, start, end, window)
            if best is None or cost < best[0]:
                best = (cost, start, end)
    if best is None:
        raise ValueError("no valid loop candidate found")

    _, coarse_start, coarse_end = best
    radius = coarse_step
    refined = best
    for start in range(max(first, coarse_start - radius), min(last, coarse_start + radius + 1)):
        minimum_end = max(start + min_length, coarse_end - radius)
        maximum_end = min(last - window, start + max_length, coarse_end + radius)
        for end in range(minimum_end, maximum_end + 1):
            cost = _boundary_cost(values, start, end, window)
            if cost < refined[0]:
                refined = (cost, start, end)

    score, start, end = refined
    effective_crossfade = min(crossfade_samples, max(1, (end - start) // 4))
    return LoopPoints(start=start, end=end, crossfade=effective_crossfade, score=score)
