from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True, slots=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def read_mono(path: str | Path) -> AudioData:
    samples, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    mono = np.mean(samples, axis=1)
    return AudioData(samples=mono, sample_rate=sample_rate)


def write_audio(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, np.asarray(samples, dtype=np.float32), sample_rate, subtype="PCM_24")


def slice_audio(audio: AudioData, start: float, end: float) -> np.ndarray:
    if start < 0:
        raise ValueError("start must be non-negative")
    if end <= start:
        raise ValueError("end must be greater than start")
    first = min(len(audio.samples), round(start * audio.sample_rate))
    last = min(len(audio.samples), round(end * audio.sample_rate))
    return audio.samples[first:last].copy()


def rms(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def apply_edge_fades(samples: np.ndarray, sample_rate: int, fade_in: float = 0.003, fade_out: float = 0.02) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64).copy()
    in_count = min(len(result), max(0, round(fade_in * sample_rate)))
    out_count = min(len(result), max(0, round(fade_out * sample_rate)))
    if in_count:
        result[:in_count] *= np.linspace(0.0, 1.0, in_count, endpoint=True)
    if out_count:
        result[-out_count:] *= np.linspace(1.0, 0.0, out_count, endpoint=True)
    return result


def detect_onset_near(audio: AudioData, expected_time: float, search_before: float = 0.12, search_after: float = 0.15, window: float = 0.02, hop: float = 0.0025) -> float:
    """Locate the strongest local log-energy rise near an expected MIDI onset."""
    if expected_time < 0:
        raise ValueError("expected_time must be non-negative")
    start = max(0.0, expected_time - search_before)
    end = min(audio.duration, expected_time + search_after)
    half_window = max(1, round(window * audio.sample_rate / 2))
    hop_samples = max(1, round(hop * audio.sample_rate))
    first_center = round(start * audio.sample_rate)
    last_center = round(end * audio.sample_rate)
    centers = np.arange(first_center, last_center + 1, hop_samples, dtype=np.int64)
    if len(centers) < 3:
        return expected_time
    energies = np.empty(len(centers), dtype=np.float64)
    for index, center in enumerate(centers):
        left = max(0, center - half_window)
        right = min(len(audio.samples), center + half_window)
        frame = audio.samples[left:right]
        energies[index] = np.mean(frame * frame) if len(frame) else 0.0
    log_energy = np.log(energies + 1e-12)
    rises = np.diff(log_energy)
    best = int(np.argmax(rises)) + 1
    return float(centers[best] / audio.sample_rate)


def estimate_global_onset_offset(audio: AudioData, midi_onsets: list[float], *, max_events: int = 16) -> float:
    if not midi_onsets:
        raise ValueError("at least one MIDI onset is required")
    selected = midi_onsets[:max_events]
    offsets = [detect_onset_near(audio, onset) - onset for onset in selected]
    return float(np.median(offsets))
