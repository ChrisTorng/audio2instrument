from audio2instrument.guitar_risk_poc import cluster_notes, note_is_isolated
from audio2instrument.midi import NoteEvent


def test_cluster_notes_groups_chord_onsets() -> None:
    notes = [
        NoteEvent(1.0, 1.4, 60, 80, 0),
        NoteEvent(1.008, 1.4, 64, 80, 0),
        NoteEvent(1.3, 1.5, 67, 80, 0),
    ]
    groups = cluster_notes(notes)
    assert [[item.note for item in group] for group in groups] == [[60, 64], [67]]


def test_isolated_note_rejects_overlap() -> None:
    first = NoteEvent(0.0, 1.0, 60, 80, 0)
    second = NoteEvent(0.5, 1.5, 64, 80, 0)
    assert not note_is_isolated(first, [first, second])
    assert note_is_isolated(first, [first])
