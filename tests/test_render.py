import numpy as np
import pytest

from audio2instrument.render import RenderNote, SampleInstrument, render_sequence, render_voice


def _instrument(*, loop: bool = False) -> SampleInstrument:
    sample_rate = 1000
    t = np.arange(500) / sample_rate
    samples = np.sin(2 * np.pi * 100 * t)
    return SampleInstrument(
        samples=samples,
        sample_rate=sample_rate,
        root_note=60,
        release=0.1,
        loop_start=100 if loop else None,
        loop_end=400 if loop else None,
        loop_crossfade=20 if loop else 0,
    )


def test_render_voice_extends_loop_beyond_sample() -> None:
    voice = render_voice(_instrument(loop=True), RenderNote(0.0, 1.2, 60))
    assert len(voice) == 1300
    assert np.max(np.abs(voice[900:1100])) > 0.1
    assert voice[-1] == pytest.approx(0.0, abs=1e-9)


def test_render_sequence_places_notes() -> None:
    rendered = render_sequence(
        _instrument(),
        [RenderNote(0.0, 0.2, 60), RenderNote(0.5, 0.2, 60)],
    )
    assert len(rendered) == 800
    assert np.max(np.abs(rendered[:300])) > 0.1
    assert np.max(np.abs(rendered[350:450])) == pytest.approx(0.0)
    assert np.max(np.abs(rendered[500:700])) > 0.1


def test_render_multisample_sequence_selects_nearest_region() -> None:
    from audio2instrument.render import SampleRegion, render_multisample_sequence

    low = _instrument()
    high = SampleInstrument(
        samples=np.full(500, 0.25),
        sample_rate=1000,
        root_note=67,
        release=0.1,
    )
    regions = [
        SampleRegion(instrument=low, low_key=0, high_key=63),
        SampleRegion(instrument=high, low_key=64, high_key=127),
    ]
    rendered = render_multisample_sequence(regions, [RenderNote(0.0, 0.2, 67)])
    assert rendered[10] == pytest.approx(0.25)
