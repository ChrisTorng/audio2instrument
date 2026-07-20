import numpy as np

from audio2instrument.midi import NoteEvent
from audio2instrument.piano_risk import (
    harmonic_mask_sample,
    render_exact_key_sfz,
    score_aligned_onset,
    select_isolated_source,
)


def test_score_aligned_onset_uses_reference_offset() -> None:
    assert score_aligned_onset(10.0, 9.98, 12.0) == 11.98


def test_select_isolated_source_prefers_no_overlap() -> None:
    overlapping = NoteEvent(0.0, 1.0, 60, 64, 0)
    companion = NoteEvent(0.2, 0.8, 67, 64, 0)
    isolated = NoteEvent(2.0, 2.4, 60, 64, 0)
    assert select_isolated_source([overlapping, companion, isolated], 60) == isolated


def test_harmonic_mask_suppresses_off_pitch_body() -> None:
    sample_rate = 16_000
    time = np.arange(sample_rate) / sample_rate
    target = np.sin(2 * np.pi * 440.0 * time)
    interferer = np.sin(2 * np.pi * 659.25 * time)
    mixture = target + np.where(time > 0.12, interferer, 0.0)
    masked = harmonic_mask_sample(mixture, sample_rate, 69)[:, 0]
    frequencies = np.fft.rfftfreq(len(masked), 1 / sample_rate)
    spectrum = np.abs(np.fft.rfft(masked))
    target_bin = np.argmin(np.abs(frequencies - 440.0))
    interferer_bin = np.argmin(np.abs(frequencies - 659.25))
    assert spectrum[target_bin] > spectrum[interferer_bin]


def test_exact_key_sfz_does_not_claim_unobserved_notes() -> None:
    text = render_exact_key_sfz({60: "Piano_60.wav", 64: "Piano_64.wav"})
    assert "lokey=60" in text and "hikey=60" in text
    assert "lokey=64" in text and "hikey=64" in text
