import numpy as np

from audio2instrument.volume_automation import (
    apply_volume_automation,
    extract_volume_automation,
)


def test_volume_automation_tracks_source_dynamics() -> None:
    sample_rate = 2_000
    time = np.arange(sample_rate * 2) / sample_rate
    carrier = np.sin(2.0 * np.pi * 110.0 * time)
    reference = carrier * np.where(time < 1.0, 0.1, 0.4)
    estimate = carrier * 0.2

    automation = extract_volume_automation(reference, estimate, sample_rate)
    rendered = apply_volume_automation(estimate, sample_rate, automation)

    quiet_rms = np.sqrt(np.mean(rendered[400:1_600] ** 2))
    loud_rms = np.sqrt(np.mean(rendered[2_400:3_600] ** 2))
    assert loud_rms > quiet_rms * 2.5
    assert rendered.shape == estimate.shape


def test_volume_automation_supports_stereo() -> None:
    sample_rate = 1_000
    reference = np.full(sample_rate, 0.25)
    estimate = np.full(sample_rate, 0.1)
    stereo = np.column_stack((estimate, estimate))

    automation = extract_volume_automation(reference, estimate, sample_rate)
    rendered = apply_volume_automation(stereo, sample_rate, automation)

    assert rendered.shape == stereo.shape
    assert np.allclose(rendered[:, 0], rendered[:, 1])
