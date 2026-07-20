from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.optimize import nnls

from audio2instrument.midi import NoteEvent


def overlap_count(notes: list[NoteEvent], target: NoteEvent) -> int:
    return sum(
        1
        for note in notes
        if note != target and note.start < target.end and note.end > target.start
    )


def select_isolated_source(
    notes: list[NoteEvent],
    pitch: int,
    *,
    excluded_ranges: tuple[tuple[float, float], ...] = (),
) -> NoteEvent:
    candidates = [
        note
        for note in notes
        if note.note == pitch
        and not any(start <= note.start < end for start, end in excluded_ranges)
    ]
    if not candidates:
        raise ValueError(f"no source note found for MIDI {pitch}")
    return min(
        candidates,
        key=lambda note: (overlap_count(notes, note), -min(note.duration, 0.8), note.start),
    )


def score_aligned_onset(
    reference_midi_onset: float,
    reference_audio_onset: float,
    target_midi_onset: float,
) -> float:
    """Apply a reliable score/audio offset when a repeated onset is acoustically masked."""
    return target_midi_onset + (reference_audio_onset - reference_midi_onset)


def harmonic_mask_sample(
    samples: np.ndarray,
    sample_rate: int,
    midi_note: int,
    *,
    attack_keep: float = 0.055,
) -> np.ndarray:
    """Suppress other pitched notes while retaining the broadband piano attack."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 <= midi_note <= 127:
        raise ValueError("midi_note must be between 0 and 127")
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("samples must be a non-empty mono or multichannel array")

    n_fft = 4096
    hop = 512
    f0 = 440.0 * 2.0 ** ((midi_note - 69) / 12.0)
    outputs: list[np.ndarray] = []
    for channel in range(values.shape[1]):
        frequencies, times, spectrum = signal.stft(
            values[:, channel],
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            boundary="zeros",
            padded=True,
        )
        frequency_mask = np.zeros_like(frequencies)
        harmonic_count = min(32, int((sample_rate / 2) // f0))
        for harmonic in range(1, harmonic_count + 1):
            center = f0 * harmonic
            width = max(10.0, center * (2.0 ** (0.45 / 12.0) - 1.0))
            frequency_mask = np.maximum(
                frequency_mask,
                np.exp(-0.5 * ((frequencies - center) / width) ** 2),
            )
        frequency_mask = 0.08 + 0.92 * frequency_mask
        time_mix = np.clip((times - attack_keep) / 0.08, 0.0, 1.0)
        mask = (1.0 - time_mix)[None, :] + time_mix[None, :] * frequency_mask[:, None]
        _, reconstructed = signal.istft(
            spectrum * mask,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
        )
        outputs.append(reconstructed[: len(values)])
    return np.stack(outputs, axis=1)


def render_one_shot_bank(
    events: list[NoteEvent],
    samples: dict[int, np.ndarray],
    sample_rate: int,
    *,
    origin: float,
    duration: float,
    gains: dict[int, float] | None = None,
) -> np.ndarray:
    if sample_rate <= 0 or duration <= 0:
        raise ValueError("sample_rate and duration must be positive")
    output = np.zeros((round(duration * sample_rate), 2), dtype=np.float64)
    note_gains = gains or {}
    for event in events:
        if event.note not in samples:
            raise ValueError(f"missing sample for MIDI {event.note}")
        sample = np.asarray(samples[event.note], dtype=np.float64)
        if sample.ndim == 1:
            sample = np.repeat(sample[:, None], 2, axis=1)
        first = round((event.start - origin) * sample_rate)
        if first < 0:
            sample = sample[-first:]
            first = 0
        last = min(len(output), first + len(sample))
        if last > first:
            gain = note_gains.get(event.note, 1.0) * event.velocity / 64.0
            output[first:last] += sample[: last - first] * gain
    return output


def _average_magnitude(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    _, _, spectrum = signal.stft(mono, fs=sample_rate, nperseg=2048, noverlap=1536)
    return np.mean(np.abs(spectrum), axis=1)


def fit_note_spectral_gains(
    calibration_reference: np.ndarray,
    calibration_events: list[NoteEvent],
    samples: dict[int, np.ndarray],
    sample_rate: int,
    *,
    origin: float,
    duration: float,
) -> dict[int, float]:
    """Fit non-negative per-note weights on one chord; intended only as a risk diagnostic."""
    pitches = sorted({event.note for event in calibration_events})
    if not pitches:
        raise ValueError("calibration_events must not be empty")
    target = _average_magnitude(calibration_reference, sample_rate)
    columns = []
    for pitch in pitches:
        rendered = render_one_shot_bank(
            [event for event in calibration_events if event.note == pitch],
            samples,
            sample_rate,
            origin=origin,
            duration=duration,
        )
        columns.append(_average_magnitude(rendered, sample_rate))
    weights, _ = nnls(np.stack(columns, axis=1), target)
    positive = weights[weights > 0]
    if positive.size:
        weights /= np.median(positive)
    return {
        pitch: float(np.clip(weight, 0.1, 4.0))
        for pitch, weight in zip(pitches, weights, strict=True)
    }


def log_spectral_distance(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_rate: int,
) -> float:
    reference_mono = reference.mean(axis=1) if reference.ndim == 2 else reference
    estimate_mono = estimate.mean(axis=1) if estimate.ndim == 2 else estimate
    count = min(len(reference_mono), len(estimate_mono))
    _, _, ref_stft = signal.stft(
        reference_mono[:count], fs=sample_rate, nperseg=2048, noverlap=1536
    )
    _, _, est_stft = signal.stft(
        estimate_mono[:count], fs=sample_rate, nperseg=2048, noverlap=1536
    )
    ref_db = 20.0 * np.log10(np.abs(ref_stft) + 1e-7)
    est_db = 20.0 * np.log10(np.abs(est_stft) + 1e-7)
    return float(np.mean(np.sqrt(np.mean((ref_db - est_db) ** 2, axis=0))))


def render_exact_key_sfz(sample_names: dict[int, str], *, release: float = 0.45) -> str:
    lines = [
        "<control>",
        "default_path=Samples/",
        "",
        "<global>",
        f"ampeg_release={release:.4f}",
        "",
    ]
    for note, filename in sorted(sample_names.items()):
        lines.extend(
            [
                "<region>",
                f"sample={Path(filename).name}",
                f"pitch_keycenter={note}",
                f"lokey={note}",
                f"hikey={note}",
                "loop_mode=no_loop",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
