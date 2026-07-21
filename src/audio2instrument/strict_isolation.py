from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


class SampleKind(StrEnum):
    NOTE = "note"
    DRUM = "drum"
    CHORD = "chord"
    PHRASE = "phrase"


@dataclass(frozen=True, slots=True)
class IsolationWindow:
    onset: float
    end: float
    duration: float
    limiting_event: str


@dataclass(frozen=True, slots=True)
class SampleAudit:
    kind: SampleKind
    duration: float
    detected_onsets: int
    accepted: bool
    reason: str
    non_target_energy_ratio: float | None = None
    synthetic_tail: bool = False


def strict_sample_window(
    onset: float,
    *,
    maximum_duration: float,
    minimum_duration: float,
    next_midi_onset: float | None = None,
    next_audio_onset: float | None = None,
    midi_safety_margin: float = 0.018,
    audio_safety_margin: float = 0.012,
) -> IsolationWindow:
    """Return a window ending before any later scored or detected attack."""
    if onset < 0:
        raise ValueError("onset must be non-negative")
    if maximum_duration <= 0:
        raise ValueError("maximum_duration must be positive")
    if minimum_duration <= 0 or minimum_duration > maximum_duration:
        raise ValueError("minimum_duration must be positive and no longer than maximum_duration")
    if midi_safety_margin < 0 or audio_safety_margin < 0:
        raise ValueError("safety margins must be non-negative")

    limits: list[tuple[float, str]] = [(onset + maximum_duration, "maximum_duration")]
    if next_midi_onset is not None:
        limits.append((next_midi_onset - midi_safety_margin, "next_midi_onset"))
    if next_audio_onset is not None:
        limits.append((next_audio_onset - audio_safety_margin, "next_audio_onset"))
    end, limiting_event = min(limits, key=lambda item: item[0])
    duration = end - onset
    if duration < minimum_duration:
        raise ValueError(
            f"only {duration:.6f} seconds remain before {limiting_event}; "
            f"minimum is {minimum_duration:.6f}"
        )
    return IsolationWindow(onset, end, duration, limiting_event)


def audit_sample(
    kind: SampleKind,
    *,
    duration: float,
    detected_onsets: int,
    minimum_duration: float,
    non_target_energy_ratio: float | None = None,
    maximum_non_target_energy_ratio: float | None = None,
    synthetic_tail: bool = False,
) -> SampleAudit:
    """Apply the hard acceptance policy for note, drum, chord, and phrase samples."""
    if duration < minimum_duration:
        return SampleAudit(
            kind,
            duration,
            detected_onsets,
            False,
            "sample is shorter than the minimum duration",
            non_target_energy_ratio,
            synthetic_tail,
        )
    if detected_onsets < 1:
        return SampleAudit(
            kind,
            duration,
            detected_onsets,
            False,
            "no attack was detected",
            non_target_energy_ratio,
            synthetic_tail,
        )
    if kind in {SampleKind.NOTE, SampleKind.DRUM} and detected_onsets != 1:
        return SampleAudit(
            kind,
            duration,
            detected_onsets,
            False,
            "note and drum samples must contain exactly one detected attack",
            non_target_energy_ratio,
            synthetic_tail,
        )
    if (
        kind is SampleKind.NOTE
        and maximum_non_target_energy_ratio is not None
        and non_target_energy_ratio is not None
        and non_target_energy_ratio > maximum_non_target_energy_ratio
    ):
        return SampleAudit(
            kind,
            duration,
            detected_onsets,
            False,
            "non-target spectral energy exceeds the allowed ratio",
            non_target_energy_ratio,
            synthetic_tail,
        )
    return SampleAudit(
        kind,
        duration,
        detected_onsets,
        True,
        "accepted",
        non_target_energy_ratio,
        synthetic_tail,
    )


def nearest_root_zones(roots: list[int]) -> dict[int, tuple[int, int]]:
    """Create non-overlapping nearest-root key zones for an ordered multisample map."""
    if not roots:
        raise ValueError("at least one root is required")
    ordered = sorted(set(roots))
    if ordered[0] < 0 or ordered[-1] > 127:
        raise ValueError("MIDI roots must be between 0 and 127")
    zones: dict[int, tuple[int, int]] = {}
    for index, root in enumerate(ordered):
        low = 0 if index == 0 else (ordered[index - 1] + root) // 2 + 1
        high = 127 if index == len(ordered) - 1 else (root + ordered[index + 1]) // 2
        zones[root] = (low, high)
    return zones


def instrument_sample_path(instrument: str, category: str, filename: str) -> PurePosixPath:
    """Keep samples isolated below Samples/<instrument>/<category>."""
    for value, label in ((instrument, "instrument"), (category, "category"), (filename, "filename")):
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"invalid {label}: {value!r}")
    return PurePosixPath("Samples") / instrument / category / filename
