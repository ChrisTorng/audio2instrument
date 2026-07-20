from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio2instrument.audio import AudioData, detect_onset_near, read_mono, slice_audio


def test_read_mono_averages_channels(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    samples = np.column_stack([np.ones(100), np.zeros(100)])
    sf.write(path, samples, 1000, subtype="FLOAT")
    audio = read_mono(path)
    assert audio.sample_rate == 1000
    assert audio.duration == pytest.approx(0.1)
    assert np.mean(audio.samples) == pytest.approx(0.5)


def test_detect_onset_near_finds_energy_rise() -> None:
    sample_rate = 4000
    samples = np.zeros(sample_rate * 2)
    samples[round(1.04 * sample_rate) :] = 0.5
    audio = AudioData(samples=samples, sample_rate=sample_rate)
    detected = detect_onset_near(audio, 1.0, hop=0.001)
    assert detected == pytest.approx(1.04, abs=0.01)


def test_slice_audio_rejects_invalid_range() -> None:
    audio = AudioData(samples=np.zeros(100), sample_rate=100)
    with pytest.raises(ValueError):
        slice_audio(audio, 1.0, 0.5)
