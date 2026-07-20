import pytest

from audio2instrument.midi import NoteEvent
from audio2instrument.sound_groups import (
    SoundGroup,
    assign_events_to_sound_groups,
    validate_sound_groups,
)


def test_sound_groups_route_by_section_not_pitch() -> None:
    groups = (
        SoundGroup("IntroPulse", 0.0, 1.0, 24),
        SoundGroup("LeadA", 1.0, 2.0, 25),
    )
    notes = [
        NoteEvent(0.2, 0.4, 80, 64, 0),
        NoteEvent(0.8, 0.9, 35, 64, 0),
        NoteEvent(1.2, 1.4, 35, 64, 0),
        NoteEvent(2.2, 2.4, 80, 64, 0),
    ]

    assigned, unassigned = assign_events_to_sound_groups(notes, groups)

    assert [note.note for note in assigned["IntroPulse"]] == [80, 35]
    assert [note.note for note in assigned["LeadA"]] == [35]
    assert unassigned == [notes[-1]]


def test_sound_groups_reject_overlap() -> None:
    with pytest.raises(ValueError):
        validate_sound_groups(
            (
                SoundGroup("A", 0.0, 1.1, 24),
                SoundGroup("B", 1.0, 2.0, 25),
            )
        )
