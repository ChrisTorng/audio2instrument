from __future__ import annotations

import numpy as np

from audio2instrument.expressive import RenderEvent, SampleVariant
from audio2instrument.loop import LoopPoints
from audio2instrument.midi import NoteEvent
from audio2instrument.velocity import (
    AttackCandidate,
    TimbreFeatures,
    VelocityLayer,
    VelocityRootLayer,
    choose_velocity_layer,
    extract_timbre_features,
    render_velocity_sequence,
    render_velocity_sfz,
    score_attack_candidates,
    velocity_layout,
)


def _variant(value: float, name: str) -> SampleVariant:
    return SampleVariant(
        samples=np.full(2000, value, dtype=np.float64),
        sample_rate=1000,
        root_note=45,
        source_note=NoteEvent(0.0, 2.0, 45, 64, 0),
        detected_onset=0.0,
        source_level=value,
        loop=LoopPoints(500, 1200, 50, 0.0),
        name=name,
    )


def test_extract_timbre_features_detects_brightness() -> None:
    rate = 8000
    t = np.arange(rate // 5) / rate
    dark = np.sin(2 * np.pi * 100 * t)
    bright = dark + 0.6 * np.sin(2 * np.pi * 1600 * t)
    assert (
        extract_timbre_features(bright, rate).spectral_centroid_hz
        > extract_timbre_features(dark, rate).spectral_centroid_hz
    )


def test_scoring_orders_level_and_brightness() -> None:
    base = NoteEvent(0.0, 1.0, 45, 64, 0)
    items = [
        AttackCandidate(base, 0.0, np.zeros(1), TimbreFeatures(0.05, 0.0, 120.0, 0.001)),
        AttackCandidate(base, 0.0, np.zeros(1), TimbreFeatures(0.10, 0.0, 180.0, 0.02)),
        AttackCandidate(base, 0.0, np.zeros(1), TimbreFeatures(0.15, 0.0, 240.0, 0.05)),
    ]
    scores = score_attack_candidates(items)
    assert scores[0].intensity_score < scores[1].intensity_score < scores[2].intensity_score


def test_velocity_layout_covers_full_range() -> None:
    layout = velocity_layout(3)
    assert layout[0][0] == 1
    assert layout[-1][1] == 127
    assert layout[0][1] + 1 == layout[1][0]


def test_choose_velocity_layer() -> None:
    root = VelocityRootLayer(
        45,
        (
            VelocityLayer(1, 64, 32, (_variant(0.2, "soft.wav"),), (0.0,)),
            VelocityLayer(65, 127, 100, (_variant(0.8, "hard.wav"),), (1.0,)),
        ),
    )
    assert choose_velocity_layer(root, 40).center_velocity == 32
    assert choose_velocity_layer(root, 100).center_velocity == 100


def test_render_velocity_sequence_uses_layer_timbre() -> None:
    root = VelocityRootLayer(
        45,
        (
            VelocityLayer(1, 64, 32, (_variant(0.2, "soft.wav"),), (0.0,)),
            VelocityLayer(65, 127, 100, (_variant(0.8, "hard.wav"),), (1.0,)),
        ),
    )
    out = render_velocity_sequence(
        [root],
        [RenderEvent(0.0, 0.2, 45, 32), RenderEvent(0.4, 0.2, 45, 100)],
        round_robin=False,
    )
    assert np.mean(np.abs(out[420:550])) > np.mean(np.abs(out[20:150])) * 4


def test_sfz_contains_velocity_and_round_robin_ranges() -> None:
    root = VelocityRootLayer(
        45,
        (
            VelocityLayer(
                1,
                64,
                32,
                (_variant(0.2, "soft1.wav"), _variant(0.3, "soft2.wav")),
                (0.0, 0.1),
            ),
            VelocityLayer(65, 127, 100, (_variant(0.8, "hard.wav"),), (1.0,)),
        ),
    )
    sfz = render_velocity_sfz([root], low_key=35, high_key=50)
    assert "lovel=1" in sfz
    assert "hivel=64" in sfz
    assert "seq_length=2" in sfz
    assert "amp_veltrack=100" in sfz
