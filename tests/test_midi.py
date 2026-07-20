from pathlib import Path

import mido
import pytest

from audio2instrument.midi import read_notes, select_notes


def test_read_notes_honors_tempo_and_velocity_zero_note_off(tmp_path: Path) -> None:
    path = tmp_path / "test.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=90, time=0))
    track.append(mido.Message("note_on", note=60, velocity=0, time=480))
    track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
    track.append(mido.Message("note_on", note=62, velocity=70, time=0))
    track.append(mido.Message("note_off", note=62, velocity=0, time=480))
    midi.save(path)
    notes = read_notes(path)
    assert len(notes) == 2
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].duration == pytest.approx(0.5)
    assert notes[0].velocity == 90
    assert notes[1].start == pytest.approx(0.5)
    assert notes[1].duration == pytest.approx(1.0)


def test_select_notes_includes_overlaps() -> None:
    from audio2instrument.midi import NoteEvent
    notes = [NoteEvent(0.0, 1.0, 60, 64, 0), NoteEvent(1.0, 2.0, 62, 64, 0), NoteEvent(2.0, 3.0, 64, 64, 0)]
    selected = select_notes(notes, 0.5, 2.5)
    assert [note.note for note in selected] == [60, 62, 64]


def test_write_notes_round_trip(tmp_path: Path) -> None:
    from audio2instrument.midi import NoteEvent, write_notes
    path = tmp_path / "round-trip.mid"
    expected = [NoteEvent(0.0, 0.75, 45, 70, 0), NoteEvent(0.75, 1.5, 44, 80, 0)]
    write_notes(path, expected)
    actual = read_notes(path)
    assert len(actual) == 2
    assert actual[0].start == pytest.approx(0.0, abs=0.002)
    assert actual[0].end == pytest.approx(0.75, abs=0.002)
    assert actual[1].start == pytest.approx(0.75, abs=0.002)
    assert actual[1].end == pytest.approx(1.5, abs=0.002)
