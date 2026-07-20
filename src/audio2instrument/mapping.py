from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio2instrument.audio import rms
from audio2instrument.midi import NoteEvent


@dataclass(frozen=True, slots=True)
class SampleCandidate:
    note: NoteEvent
    score: float


def assign_key_ranges(
    root_notes: list[int] | tuple[int, ...],
    *,
    low_key: int = 0,
    high_key: int = 127,
) -> dict[int, tuple[int, int]]:
    """Assign each root note a nearest-neighbour MIDI key range."""
    roots = sorted(set(root_notes))
    if not roots:
        raise ValueError("at least one root note is required")
    if not 0 <= low_key <= high_key <= 127:
        raise ValueError("key range must be between 0 and 127")
    if roots[0] < low_key or roots[-1] > high_key:
        raise ValueError("all root notes must fall inside the requested key range")

    ranges: dict[int, tuple[int, int]] = {}
    next_low = low_key
    for index, root in enumerate(roots):
        if index == len(roots) - 1:
            region_high = high_key
        else:
            region_high = (root + roots[index + 1]) // 2
        ranges[root] = (next_low, region_high)
        next_low = region_high + 1
    return ranges


def select_root_candidates(
    notes: list[NoteEvent],
    *,
    minimum_duration: float,
    maximum_roots: int = 8,
    minimum_pitch_spacing: int = 1,
) -> list[SampleCandidate]:
    """Pick the longest cleanly separated note for each useful pitch.

    The first version intentionally relies on MIDI structure only. Duration is rewarded while
    overlaps and very small gaps are penalized. The deterministic spacing pass limits redundant
    roots when the source contains many adjacent pitches.
    """
    if minimum_duration <= 0:
        raise ValueError("minimum_duration must be positive")
    if maximum_roots <= 0:
        raise ValueError("maximum_roots must be positive")
    if minimum_pitch_spacing < 1:
        raise ValueError("minimum_pitch_spacing must be at least one semitone")

    ordered = sorted(notes, key=lambda note: (note.start, note.note))
    best_by_pitch: dict[int, SampleCandidate] = {}
    for index, note in enumerate(ordered):
        if note.duration < minimum_duration:
            continue
        previous_end = ordered[index - 1].end if index else note.start - 1.0
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else note.end + 1.0
        previous = ordered[index - 1] if index else None
        overlap_before = max(0.0, previous_end - note.start)
        overlap_after = max(0.0, note.end - next_start)
        gap_before = max(0.0, note.start - previous_end)
        gap_after = max(0.0, next_start - note.end)
        score = note.duration + 0.15 * min(gap_before, 0.25) + 0.15 * min(gap_after, 0.25)
        score -= 4.0 * (overlap_before + overlap_after)
        if (
            previous is not None
            and previous.note == note.note
            and abs(previous.end - note.start) <= 0.08
        ):
            # A tied same-pitch retrigger often has no observable acoustic onset.
            score -= 2.0
        candidate = SampleCandidate(note=note, score=score)
        current = best_by_pitch.get(note.note)
        if current is None or candidate.score > current.score:
            best_by_pitch[note.note] = candidate

    ranked = sorted(best_by_pitch.values(), key=lambda item: (-item.score, item.note.note))
    selected: list[SampleCandidate] = []
    for candidate in ranked:
        if any(abs(candidate.note.note - other.note.note) < minimum_pitch_spacing for other in selected):
            continue
        selected.append(candidate)
        if len(selected) == maximum_roots:
            break
    return sorted(selected, key=lambda item: item.note.note)


def onset_level(samples: np.ndarray, sample_rate: int, *, offset: float = 0.03, duration: float = 0.2) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    first = max(0, round(offset * sample_rate))
    last = min(len(samples), round((offset + duration) * sample_rate))
    return rms(samples[first:last])


def levels_to_velocities(
    levels: list[float],
    *,
    reference_level: float | None = None,
    reference_velocity: int = 64,
) -> list[int]:
    """Encode relative linear onset amplitudes as deterministic MIDI velocities."""
    if not levels:
        raise ValueError("at least one level is required")
    if not 1 <= reference_velocity <= 127:
        raise ValueError("reference_velocity must be between 1 and 127")
    positive = [level for level in levels if level > 0]
    if not positive:
        return [reference_velocity for _ in levels]
    reference = reference_level if reference_level is not None else float(np.median(positive))
    if reference <= 0:
        raise ValueError("reference_level must be positive")
    return [int(np.clip(round(reference_velocity * level / reference), 1, 127)) for level in levels]
