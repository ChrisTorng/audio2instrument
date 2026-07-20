from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, deque

import mido


@dataclass(frozen=True, slots=True)
class NoteEvent:
    start: float
    end: float
    note: int
    velocity: int
    channel: int

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_notes(path: str | Path) -> list[NoteEvent]:
    midi = mido.MidiFile(path)
    tempo = 500_000
    time_seconds = 0.0
    active: dict[tuple[int, int], deque[tuple[float, int]]] = defaultdict(deque)
    notes: list[NoteEvent] = []

    for message in mido.merge_tracks(midi.tracks):
        time_seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            continue
        if message.type == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)].append((time_seconds, message.velocity))
            continue
        is_note_off = message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        )
        if not is_note_off:
            continue
        key = (message.channel, message.note)
        if not active[key]:
            continue
        start, velocity = active[key].popleft()
        if time_seconds > start:
            notes.append(
                NoteEvent(
                    start=start,
                    end=time_seconds,
                    note=message.note,
                    velocity=velocity,
                    channel=message.channel,
                )
            )

    return sorted(notes, key=lambda event: (event.start, event.note, event.channel))


def select_notes(notes: list[NoteEvent], start: float, end: float) -> list[NoteEvent]:
    if end <= start:
        raise ValueError("end must be greater than start")
    return [note for note in notes if note.start < end and note.end > start]


def write_notes(
    path: str | Path,
    notes: list[NoteEvent],
    *,
    tempo: int = 500_000,
    ticks_per_beat: int = 960,
) -> None:
    """Write note events whose times are expressed in seconds to a simple type-0 MIDI file."""
    if tempo <= 0 or ticks_per_beat <= 0:
        raise ValueError("tempo and ticks_per_beat must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events: list[tuple[float, int, mido.Message]] = []
    for note in notes:
        if note.start < 0 or note.end <= note.start:
            raise ValueError("notes must have non-negative starts and positive durations")
        events.append((note.start, 1, mido.Message("note_on", channel=note.channel, note=note.note, velocity=note.velocity, time=0)))
        events.append((note.end, 0, mido.Message("note_off", channel=note.channel, note=note.note, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], item[1]))
    previous_seconds = 0.0
    for event_seconds, _, message in events:
        delta_seconds = event_seconds - previous_seconds
        message.time = round(mido.second2tick(delta_seconds, ticks_per_beat, tempo))
        track.append(message)
        previous_seconds = event_seconds
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(output)
