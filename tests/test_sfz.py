import pytest

from audio2instrument.sfz import SfzRegion, render_sfz


def test_render_sfz_includes_loop_and_release() -> None:
    text = render_sfz([SfzRegion(sample="Bass_D2.wav", root_key=38, low_key=35, high_key=41, release=0.2, loop_start=100, loop_end=300, loop_crossfade=0.03)])
    assert "sample=Bass_D2.wav" in text
    assert "pitch_keycenter=38" in text
    assert "loop_mode=loop_sustain" in text
    assert "loop_end=299" in text
    assert "loop_crossfade=0.030000" in text
    assert "ampeg_release=0.200000" in text


def test_sfz_rejects_partial_loop() -> None:
    with pytest.raises(ValueError):
        render_sfz([SfzRegion(sample="x.wav", root_key=60, loop_start=10)])
