from __future__ import annotations

from dataclasses import dataclass

from audio2instrument.midi import NoteEvent


@dataclass(frozen=True, slots=True)
class SoundGroup:
    """One explicitly identified instrument or timbre state on a source timeline."""

    name: str
    start: float
    end: float
    trigger_key: int

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("sound group name must not be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("sound group time range must be positive and ordered")
        if not 0 <= self.trigger_key <= 127:
            raise ValueError("trigger_key must be between 0 and 127")


def validate_sound_groups(groups: list[SoundGroup] | tuple[SoundGroup, ...]) -> None:
    ordered = sorted(groups, key=lambda item: item.start)
    for index, group in enumerate(ordered):
        group.validate()
        if index and group.start < ordered[index - 1].end:
            raise ValueError("sound groups must not overlap")
    names = [group.name for group in ordered]
    if len(names) != len(set(names)):
        raise ValueError("sound group names must be unique")


def assign_events_to_sound_groups(
    notes: list[NoteEvent],
    groups: list[SoundGroup] | tuple[SoundGroup, ...],
) -> tuple[dict[str, list[NoteEvent]], list[NoteEvent]]:
    """Assign events by source section, never by MIDI register.

    The caller identifies distinct sounds from timbre, automation, patch changes, or musical
    sections. MIDI pitch is deliberately not part of the routing decision.
    """
    validate_sound_groups(groups)
    assigned = {group.name: [] for group in groups}
    unassigned: list[NoteEvent] = []
    for note in notes:
        group = next((item for item in groups if item.start <= note.start < item.end), None)
        if group is None:
            unassigned.append(note)
        else:
            assigned[group.name].append(note)
    return assigned, unassigned
