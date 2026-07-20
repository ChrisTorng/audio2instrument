from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


@dataclass(frozen=True, slots=True)
class VolumeAutomation:
    times: np.ndarray
    gain_db: np.ndarray
    gate: np.ndarray
    global_gain: float

    def validate(self) -> None:
        if self.global_gain <= 0:
            raise ValueError("global_gain must be positive")
        if not (len(self.times) == len(self.gain_db) == len(self.gate)):
            raise ValueError("volume-automation arrays must have equal lengths")
        if len(self.times) == 0:
            raise ValueError("volume automation must contain at least one point")
        if np.any(np.diff(self.times) < 0):
            raise ValueError("volume-automation times must be ordered")


def _mono(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        return np.mean(values, axis=1)
    raise ValueError("audio samples must be mono or two-dimensional")


def _frame_rms(
    samples: np.ndarray,
    sample_rate: int,
    *,
    window: float,
    hop: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if window <= 0 or hop <= 0:
        raise ValueError("window and hop must be positive")
    values = _mono(samples)
    window_samples = max(1, round(window * sample_rate))
    hop_samples = max(1, round(hop * sample_rate))
    starts = np.arange(0, max(1, len(values) - window_samples + 1), hop_samples, dtype=np.int64)
    ends = np.minimum(starts + window_samples, len(values))
    power = values * values
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    rms = np.sqrt(
        (cumulative[ends] - cumulative[starts]) / np.maximum(1, ends - starts) + 1e-15
    )
    times = (starts + (ends - starts) / 2.0) / sample_rate
    return times, rms


def extract_volume_automation(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
    *,
    window: float = 0.2,
    hop: float = 0.05,
    minimum_gain_db: float = -18.0,
    maximum_gain_db: float = 12.0,
) -> VolumeAutomation:
    """Extract a smoothed source-volume curve for a reconstructed track."""
    if minimum_gain_db > maximum_gain_db:
        raise ValueError("minimum_gain_db must not exceed maximum_gain_db")
    count = min(len(reference), len(estimate))
    if count == 0:
        raise ValueError("reference and estimate must not be empty")
    times, reference_rms = _frame_rms(reference[:count], sample_rate, window=window, hop=hop)
    _, estimate_rms = _frame_rms(estimate[:count], sample_rate, window=window, hop=hop)

    floor = max(float(np.percentile(reference_rms, 15)) * 0.25, 1e-5)
    active = reference_rms > floor
    reference_level = (
        float(np.median(reference_rms[active]))
        if np.any(active)
        else float(np.sqrt(np.mean(_mono(reference[:count]) ** 2) + 1e-15))
    )
    estimate_active = estimate_rms > max(float(np.percentile(estimate_rms, 15)) * 0.25, 1e-6)
    estimate_level = (
        float(np.median(estimate_rms[estimate_active]))
        if np.any(estimate_active)
        else float(np.sqrt(np.mean(_mono(estimate[:count]) ** 2) + 1e-15))
    )
    global_gain = reference_level / max(estimate_level, 1e-12)
    estimate_after_global = estimate_rms * global_gain
    gain_db = 20.0 * np.log10((reference_rms + 1e-6) / (estimate_after_global + 1e-6))
    valid = active & (estimate_after_global > 1e-6)
    indexes = np.arange(len(gain_db))
    if np.any(valid):
        gain_db[~valid] = np.interp(indexes[~valid], indexes[valid], gain_db[valid])
    gain_db = np.clip(gain_db, minimum_gain_db, maximum_gain_db)
    gain_db = median_filter(gain_db, size=9, mode="nearest")
    gain_db = gaussian_filter1d(gain_db, sigma=4, mode="nearest")
    gate = np.clip((20.0 * np.log10(reference_rms + 1e-8) + 70.0) / 18.0, 0.0, 1.0)
    gate = gaussian_filter1d(gate, sigma=3, mode="nearest")
    result = VolumeAutomation(times, gain_db, gate, global_gain)
    result.validate()
    return result


def apply_volume_automation(
    samples: np.ndarray,
    sample_rate: int,
    automation: VolumeAutomation,
) -> np.ndarray:
    automation.validate()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = np.asarray(samples, dtype=np.float64)
    positions = np.arange(len(values), dtype=np.float64) / sample_rate
    local_gain = np.interp(positions, automation.times, 10.0 ** (automation.gain_db / 20.0))
    gate = np.interp(positions, automation.times, automation.gate)
    gain = automation.global_gain * local_gain * gate
    if values.ndim == 1:
        return values * gain
    if values.ndim == 2:
        return values * gain[:, None]
    raise ValueError("audio samples must be mono or two-dimensional")
