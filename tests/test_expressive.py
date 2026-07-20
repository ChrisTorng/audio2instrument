from __future__ import annotations

from dataclasses import replace

import numpy as np

from audio2instrument.expressive import (
    RenderEvent,
    RootLayer,
    SampleVariant,
    adaptive_root_weights,
    equal_power_root_weights,
    hybridize_round_robin_layers,
    infer_velocities,
    render_expressive_sequence,
    render_sfz_round_robin_crossfade,
    reject_ambiguous_same_pitch_retrigger,
    select_round_robin_candidates,
)
from audio2instrument.loop import LoopPoints
from audio2instrument.midi import NoteEvent


def _variant(root: int, value: float, name: str) -> SampleVariant:
    return SampleVariant(
        samples=np.full(2000, value, dtype=np.float64),
        sample_rate=1000,
        root_note=root,
        source_note=NoteEvent(0.0, 2.0, root, 64, 0),
        detected_onset=0.0,
        source_level=abs(value),
        loop=LoopPoints(500, 1200, 50, 0.0),
        name=name,
    )


def test_equal_power_weights_sum_to_constant_power() -> None:
    weights = equal_power_root_weights([40, 44], 42)
    assert set(weights) == {40, 44}
    assert np.isclose(sum(weight * weight for weight in weights.values()), 1.0)


def test_exact_root_does_not_crossfade() -> None:
    assert equal_power_root_weights([40, 44], 44) == {44: 1.0}


def test_rejects_tied_same_pitch_retrigger() -> None:
    notes = [NoteEvent(0.0, 1.0, 45, 64, 0), NoteEvent(1.0, 3.0, 45, 64, 0)]
    assert reject_ambiguous_same_pitch_retrigger(notes, 1)
    assert select_round_robin_candidates(notes, 45, minimum_duration=1.5) == []


def test_round_robin_changes_repeated_note_output() -> None:
    layer = RootLayer(45, (_variant(45, 0.25, "a.wav"), _variant(45, 0.75, "b.wav")))
    events = [RenderEvent(0.0, 0.2, 45), RenderEvent(0.4, 0.2, 45)]
    output = render_expressive_sequence([layer], events, round_robin=True)
    first = np.mean(output[20:150])
    second = np.mean(output[420:550])
    assert second > first * 2


def test_velocity_inference_preserves_relative_level() -> None:
    velocities = infer_velocities([0.5, 1.0, 2.0])
    assert velocities[0] < velocities[1] < velocities[2]


def test_sfz_contains_round_robin_and_key_crossfade() -> None:
    layers = [
        RootLayer(40, (_variant(40, 0.5, "40a.wav"), _variant(40, 0.6, "40b.wav"))),
        RootLayer(44, (_variant(44, 0.5, "44a.wav"),)),
    ]
    sfz = render_sfz_round_robin_crossfade(layers, low_key=36, high_key=48)
    assert "seq_length=2" in sfz
    assert "seq_position=2" in sfz
    assert "xfout_lokey=40" in sfz
    assert "xfin_hikey=44" in sfz
    assert "xf_keycurve=power" in sfz


def test_hybrid_round_robin_returns_to_canonical_body() -> None:
    t = np.arange(2000) / 1000.0
    canonical = replace(
        _variant(45, 0.25, "a.wav"),
        samples=0.25 * np.sin(2 * np.pi * 80 * t),
    )
    alternate = replace(
        _variant(45, 0.25, "b.wav"),
        samples=0.25 * np.sin(2 * np.pi * 160 * t),
    )
    layer = hybridize_round_robin_layers(
        [RootLayer(45, (canonical, alternate))],
        attack_duration=0.1,
        transition_duration=0.05,
    )[0]
    assert not np.allclose(layer.variants[1].samples[:80], canonical.samples[:80])
    assert np.allclose(layer.variants[1].samples[200:], canonical.samples[200:])
    assert layer.variants[1].loop == canonical.loop


def test_adaptive_weights_disable_incompatible_crossfade() -> None:
    assert adaptive_root_weights([35, 43], 39, set()) == {35: 1.0}
    weights = adaptive_root_weights([35, 43], 39, {(35, 43)})
    assert set(weights) == {35, 43}
