import pytest

from audio2instrument.midi import NoteEvent
from audio2instrument.poc import select_long_monophonic_run


def test_select_long_monophonic_run_skips_short_notes() -> None:
    notes = [
        NoteEvent(0.0, 0.5, 40, 64, 0),
        NoteEvent(1.0, 4.0, 45, 64, 0),
        NoteEvent(4.0, 7.2, 44, 64, 0),
        NoteEvent(7.2, 10.5, 43, 64, 0),
    ]
    selected = select_long_monophonic_run(notes, search_start=0.0, count=3, minimum_duration=2.5)
    assert [note.note for note in selected] == [45, 44, 43]


def test_select_long_monophonic_run_fails_without_candidate() -> None:
    with pytest.raises(ValueError):
        select_long_monophonic_run([NoteEvent(0.0, 0.5, 40, 64, 0)], search_start=0.0, count=3, minimum_duration=2.5)
