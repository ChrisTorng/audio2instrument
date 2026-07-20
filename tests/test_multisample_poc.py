from pathlib import Path

import numpy as np
import soundfile as sf

from audio2instrument.midi import NoteEvent, write_notes
from audio2instrument.multisample_poc import BassMultisamplePocConfig, run_bass_multisample_poc


def _synthesize_note(sample_rate: int, duration: float, frequency: float, amplitude: float) -> np.ndarray:
    t = np.arange(round(duration * sample_rate)) / sample_rate
    attack = np.minimum(1.0, t / 0.01)
    decay = np.exp(-0.08 * t)
    return amplitude * attack * decay * np.sin(2 * np.pi * frequency * t)


def test_multisample_poc_creates_instrument_and_two_validations(tmp_path: Path) -> None:
    sample_rate = 4000
    notes: list[NoteEvent] = []
    audio = np.zeros(sample_rate * 15, dtype=np.float64)
    sequence = [40, 42, 44, 40, 42, 44]
    for index, midi_note in enumerate(sequence):
        start = index * 2.05
        end = start + 2.0
        note = NoteEvent(start=start, end=end, note=midi_note, velocity=64, channel=0)
        notes.append(note)
        frequency = 440.0 * 2 ** ((midi_note - 69) / 12)
        voice = _synthesize_note(sample_rate, 2.0, frequency, 0.4 + 0.03 * index)
        first = round(start * sample_rate)
        audio[first : first + len(voice)] += voice

    audio_path = tmp_path / "bass.wav"
    midi_path = tmp_path / "bass.mid"
    output_dir = tmp_path / "out"
    sf.write(audio_path, audio, sample_rate, subtype="FLOAT")
    write_notes(midi_path, notes)

    report = run_bass_multisample_poc(
        BassMultisamplePocConfig(
            audio_path=audio_path,
            midi_path=midi_path,
            output_dir=output_dir,
            sample_duration=1.2,
            candidate_minimum_duration=1.3,
            maximum_roots=3,
            validation_minimum_duration=1.5,
            same_segment_start=0.0,
            held_out_start=6.0,
        )
    )

    roots = report["instrument"]["roots"]
    assert [root["midi_note"] for root in roots] == [40, 42, 44]
    assert (output_dir / "Bass-Multisample.sfz").exists()
    assert (output_dir / "validation-same" / "03-ab-original-then-reconstructed.wav").exists()
    assert (output_dir / "validation-held-out" / "03-ab-original-then-reconstructed.wav").exists()
    assert len(report["validations"]) == 2
