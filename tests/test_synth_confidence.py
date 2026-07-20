from pathlib import Path

import numpy as np

from audio2instrument.audio import AudioData
from audio2instrument.midi import NoteEvent, read_notes
from audio2instrument.synth_confidence import (
    correct_midi_from_audio,
    filter_events_by_audio,
    split_register,
    target_harmonic_ratio,
)


def test_target_harmonic_ratio_detects_matching_sine() -> None:
    sample_rate = 8_000
    time = np.arange(1_600) / sample_rate
    samples = np.sin(2.0 * np.pi * 440.0 * time)
    assert target_harmonic_ratio(samples, sample_rate, 69) > 0.9
    assert target_harmonic_ratio(samples, sample_rate, 48) < 0.2


def test_filter_events_rejects_silent_false_positive() -> None:
    sample_rate = 8_000
    samples = np.zeros(sample_rate, dtype=np.float64)
    time = np.arange(1_600) / sample_rate
    samples[800:2_400] = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)
    audio = AudioData(samples, sample_rate)
    sounding = NoteEvent(0.1, 0.3, 69, 64, 0)
    silent = NoteEvent(0.6, 0.8, 69, 64, 0)

    accepted, evidence = filter_events_by_audio(audio, [sounding, silent])

    assert accepted == [sounding]
    assert evidence[0].accepted
    assert not evidence[1].accepted


def test_correct_midi_writes_only_supported_events(tmp_path: Path, monkeypatch) -> None:
    from audio2instrument import synth_confidence

    sample_rate = 8_000
    samples = np.zeros(sample_rate, dtype=np.float64)
    time = np.arange(1_600) / sample_rate
    samples[800:2_400] = 0.1 * np.sin(2.0 * np.pi * 440.0 * time)
    notes = [
        NoteEvent(0.1, 0.3, 69, 64, 0),
        NoteEvent(0.6, 0.8, 69, 64, 0),
    ]
    monkeypatch.setattr(synth_confidence, "read_mono", lambda _: AudioData(samples, sample_rate))
    monkeypatch.setattr(synth_confidence, "read_notes", lambda _: notes)
    output = tmp_path / "corrected.mid"

    report = correct_midi_from_audio("audio.wav", "input.mid", output)

    written = read_notes(output)
    assert report["accepted_events"] == 1
    assert len(written) == 1
    assert written[0].note == 69


def test_register_split() -> None:
    assert split_register(40) == "low"
    assert split_register(69) == "lead"
