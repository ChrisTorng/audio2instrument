import pytest

from audio2instrument.strict_isolation import (
    SampleKind,
    audit_sample,
    instrument_sample_path,
    nearest_root_zones,
    strict_sample_window,
)


def test_window_stops_before_next_acoustic_onset() -> None:
    window = strict_sample_window(
        10.0,
        maximum_duration=0.30,
        minimum_duration=0.08,
        next_midi_onset=10.28,
        next_audio_onset=10.21,
    )
    assert window.end == pytest.approx(10.198)
    assert window.limiting_event == "next_audio_onset"


def test_window_rejects_an_unusably_short_gap() -> None:
    with pytest.raises(ValueError, match="minimum"):
        strict_sample_window(
            1.0,
            maximum_duration=0.30,
            minimum_duration=0.08,
            next_audio_onset=1.05,
        )


def test_note_requires_exactly_one_attack() -> None:
    result = audit_sample(
        SampleKind.NOTE,
        duration=0.15,
        detected_onsets=2,
        minimum_duration=0.08,
    )
    assert not result.accepted
    assert "exactly one" in result.reason


def test_note_rejects_excess_non_target_energy() -> None:
    result = audit_sample(
        SampleKind.NOTE,
        duration=0.15,
        detected_onsets=1,
        minimum_duration=0.08,
        non_target_energy_ratio=0.42,
        maximum_non_target_energy_ratio=0.36,
    )
    assert not result.accepted


def test_chord_may_contain_micro_onsets_when_labeled_as_chord() -> None:
    result = audit_sample(
        SampleKind.CHORD,
        duration=0.18,
        detected_onsets=2,
        minimum_duration=0.08,
    )
    assert result.accepted


def test_nearest_root_zones_cover_the_keyboard_without_overlap() -> None:
    assert nearest_root_zones([60, 64, 67]) == {
        60: (0, 62),
        64: (63, 65),
        67: (66, 127),
    }


def test_instrument_scoped_path() -> None:
    path = instrument_sample_path("Piano", "HybridSingleNotes", "Piano_note_60.wav")
    assert path.as_posix() == "Samples/Piano/HybridSingleNotes/Piano_note_60.wav"
