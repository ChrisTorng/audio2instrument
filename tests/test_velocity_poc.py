from __future__ import annotations

from pathlib import Path

from audio2instrument.velocity_poc import BassVelocityPocConfig


def test_velocity_poc_defaults_exclude_all_validation_ranges() -> None:
    config = BassVelocityPocConfig(Path("bass.wav"), Path("bass.mid"), Path("out"))
    assert len(config.validation_segments) == 3
    assert all(end > start for _, start, end in config.validation_segments)
    assert config.root_notes == (35, 43, 44, 45, 47)
