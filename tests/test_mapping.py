import numpy as np

from audio2instrument.mapping import (
    assign_key_ranges,
    levels_to_velocities,
    onset_level,
    select_root_candidates,
)
from audio2instrument.midi import NoteEvent


def test_assign_key_ranges_uses_nearest_midpoints() -> None:
    assert assign_key_ranges([35, 43, 44, 45, 47], low_key=26, high_key=61) == {
        35: (26, 39),
        43: (40, 43),
        44: (44, 44),
        45: (45, 46),
        47: (47, 61),
    }


def test_select_root_candidates_keeps_longest_note_per_pitch() -> None:
    notes = [
        NoteEvent(0.0, 1.6, 40, 64, 0),
        NoteEvent(2.0, 4.5, 40, 64, 0),
        NoteEvent(5.0, 7.0, 45, 64, 0),
    ]
    selected = select_root_candidates(notes, minimum_duration=1.5, maximum_roots=8)
    assert [(item.note.note, item.note.duration) for item in selected] == [(40, 2.5), (45, 2.0)]


def test_levels_to_velocities_preserves_relative_amplitude() -> None:
    assert levels_to_velocities([0.5, 1.0, 1.5], reference_level=1.0) == [32, 64, 96]


def test_onset_level_ignores_initial_silence() -> None:
    samples = np.concatenate([np.zeros(30), np.ones(200), np.zeros(100)])
    assert onset_level(samples, 1000, offset=0.03, duration=0.2) == 1.0


def test_select_root_candidates_penalizes_tied_same_pitch_retrigger() -> None:
    notes = [
        NoteEvent(0.0, 3.7, 47, 64, 0),
        NoteEvent(3.7, 8.2, 47, 64, 0),
    ]
    selected = select_root_candidates(notes, minimum_duration=1.5, maximum_roots=1)
    assert selected[0].note.start == 0.0
