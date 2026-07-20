from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from audio2instrument.audio import AudioData, read_mono
from audio2instrument.midi import NoteEvent, read_notes, write_notes


@dataclass(frozen=True, slots=True)
class EventEvidence:
    level: float
    harmonic_ratio: float
    accepted: bool


def target_harmonic_ratio(samples: np.ndarray, sample_rate: int, midi_note: int) -> float:
    """Return the fraction of short-time spectral energy near a note's harmonic series."""
    values = np.asarray(samples, dtype=np.float64)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0 <= midi_note <= 127:
        raise ValueError("midi_note must be between 0 and 127")
    if len(values) < 256:
        return 0.0
    windowed = values * np.hanning(len(values))
    power = np.abs(np.fft.rfft(windowed)) ** 2
    frequencies = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    total = float(np.sum(power) + 1e-15)
    fundamental = 440.0 * 2.0 ** ((midi_note - 69) / 12.0)
    mask = np.zeros(len(power), dtype=bool)
    harmonic = 1
    while harmonic * fundamental < sample_rate / 2.0:
        center = harmonic * fundamental
        bandwidth = max(8.0, center * 0.025)
        mask |= np.abs(frequencies - center) <= bandwidth
        harmonic += 1
    return float(np.sum(power[mask]) / total)


def evaluate_event(
    audio: AudioData,
    note: NoteEvent,
    *,
    analysis_duration: float = 0.18,
    minimum_level: float = 0.001,
    minimum_harmonic_ratio: float = 0.03,
) -> EventEvidence:
    if analysis_duration <= 0:
        raise ValueError("analysis_duration must be positive")
    first = max(0, round(note.start * audio.sample_rate))
    count = max(256, round(min(analysis_duration, note.duration) * audio.sample_rate))
    segment = audio.samples[first : first + count]
    if len(segment) < 256:
        return EventEvidence(0.0, 0.0, False)
    level = float(np.sqrt(np.mean(segment * segment) + 1e-15))
    harmonic_ratio = target_harmonic_ratio(segment, audio.sample_rate, note.note)
    accepted = level >= minimum_level and harmonic_ratio >= minimum_harmonic_ratio
    return EventEvidence(level, harmonic_ratio, accepted)


def filter_events_by_audio(
    audio: AudioData,
    notes: list[NoteEvent],
    *,
    analysis_duration: float = 0.18,
    minimum_level: float = 0.001,
    minimum_harmonic_ratio: float = 0.03,
) -> tuple[list[NoteEvent], list[EventEvidence]]:
    evidence = [
        evaluate_event(
            audio,
            note,
            analysis_duration=analysis_duration,
            minimum_level=minimum_level,
            minimum_harmonic_ratio=minimum_harmonic_ratio,
        )
        for note in notes
    ]
    accepted = [note for note, item in zip(notes, evidence, strict=True) if item.accepted]
    return accepted, evidence


def correct_midi_from_audio(
    audio_path: str | Path,
    midi_path: str | Path,
    output_path: str | Path,
    *,
    minimum_level: float = 0.001,
    minimum_harmonic_ratio: float = 0.03,
) -> dict[str, object]:
    audio = read_mono(audio_path)
    notes = read_notes(midi_path)
    accepted, evidence = filter_events_by_audio(
        audio,
        notes,
        minimum_level=minimum_level,
        minimum_harmonic_ratio=minimum_harmonic_ratio,
    )
    write_notes(output_path, accepted)
    return {
        "input_events": len(notes),
        "accepted_events": len(accepted),
        "rejected_events": len(notes) - len(accepted),
        "minimum_level": minimum_level,
        "minimum_harmonic_ratio": minimum_harmonic_ratio,
        "events": [
            {
                "note": asdict(note),
                "evidence": asdict(item),
            }
            for note, item in zip(notes, evidence, strict=True)
        ],
    }
