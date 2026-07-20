from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from audio2instrument.audio import (
    apply_edge_fades,
    detect_onset_near,
    read_mono,
    rms,
    slice_audio,
    write_audio,
)
from audio2instrument.loop import find_loop_points
from audio2instrument.mapping import (
    assign_key_ranges,
    levels_to_velocities,
    onset_level,
    select_root_candidates,
)
from audio2instrument.metrics import compare_audio
from audio2instrument.midi import NoteEvent, read_notes, write_notes
from audio2instrument.poc import select_long_monophonic_run
from audio2instrument.render import (
    RenderNote,
    SampleInstrument,
    SampleRegion,
    render_multisample_sequence,
)
from audio2instrument.sfz import SfzRegion, render_sfz


@dataclass(frozen=True, slots=True)
class BassMultisamplePocConfig:
    audio_path: Path
    midi_path: Path
    output_dir: Path
    sample_duration: float = 1.35
    release: float = 0.08
    candidate_minimum_duration: float = 1.5
    maximum_roots: int = 5
    validation_note_count: int = 3
    validation_minimum_duration: float = 1.5
    same_segment_start: float = 110.0
    held_out_start: float = 170.0


@dataclass(frozen=True, slots=True)
class BuiltRoot:
    source_note: NoteEvent
    onset: float
    source_level: float
    normalized_samples: np.ndarray
    loop_start: int
    loop_end: int
    loop_crossfade: int
    loop_score: float
    sample_name: str


def _following_note(notes: list[NoteEvent], note: NoteEvent) -> NoteEvent | None:
    return next((item for item in notes if item.start > note.start and item.start >= note.end - 1e-6), None)


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length].copy()
    return np.pad(samples, (0, length - len(samples)))


def _detect_run_boundaries(
    audio_samples: np.ndarray,
    sample_rate: int,
    audio_duration: float,
    notes: list[NoteEvent],
    selected: list[NoteEvent],
) -> list[float]:
    from audio2instrument.audio import AudioData

    audio = AudioData(audio_samples, sample_rate)
    following = _following_note(notes, selected[-1])
    expected = [note.start for note in selected]
    expected.append(following.start if following is not None else selected[-1].end)
    return [detect_onset_near(audio, min(value, audio_duration)) for value in expected]


def _build_roots(
    config: BassMultisamplePocConfig,
    audio_samples: np.ndarray,
    sample_rate: int,
    notes: list[NoteEvent],
) -> tuple[list[BuiltRoot], float]:
    from audio2instrument.audio import AudioData

    audio = AudioData(audio_samples, sample_rate)
    candidates = select_root_candidates(
        notes,
        minimum_duration=max(config.candidate_minimum_duration, config.sample_duration + 0.05),
        maximum_roots=config.maximum_roots,
    )
    if len(candidates) < 2:
        raise ValueError("multisample proof of concept requires at least two root candidates")

    raw: list[tuple[NoteEvent, float, np.ndarray, float, object]] = []
    for candidate in candidates:
        note = candidate.note
        onset = detect_onset_near(audio, note.start)
        following = _following_note(notes, note)
        available_end = following.start if following is not None else note.end
        if following is not None:
            available_end = detect_onset_near(audio, following.start)
        sample_end = min(onset + config.sample_duration, available_end - 0.03)
        if sample_end - onset < 0.9:
            continue
        sample = slice_audio(audio, onset, sample_end)
        sample = apply_edge_fades(sample, sample_rate, fade_in=0.002, fade_out=0.0)
        level = onset_level(sample, sample_rate)
        search_end = min(len(sample) / sample_rate - 0.04, config.sample_duration - 0.05)
        loop = find_loop_points(
            sample,
            sample_rate,
            search_start=min(0.4, search_end - 0.3),
            search_end=search_end,
            min_duration=0.25,
            max_duration=min(0.7, search_end - 0.05),
            crossfade=0.03,
        )
        raw.append((note, onset, sample, level, loop))

    if len(raw) < 2:
        raise ValueError("fewer than two root samples survived extraction")
    positive_levels = [item[3] for item in raw if item[3] > 0]
    target_level = float(np.median(positive_levels)) if positive_levels else 1.0
    roots: list[BuiltRoot] = []
    for note, onset, sample, level, loop in raw:
        normalized = sample * (target_level / level if level > 0 else 1.0)
        roots.append(
            BuiltRoot(
                source_note=note,
                onset=onset,
                source_level=level,
                normalized_samples=normalized,
                loop_start=loop.start,
                loop_end=loop.end,
                loop_crossfade=loop.crossfade,
                loop_score=loop.score,
                sample_name=f"Bass_{note.note}.wav",
            )
        )
    return sorted(roots, key=lambda root: root.source_note.note), target_level


def _render_validation(
    *,
    label: str,
    output_dir: Path,
    audio_samples: np.ndarray,
    sample_rate: int,
    notes: list[NoteEvent],
    regions: list[SampleRegion],
    target_level: float,
    search_start: float,
    note_count: int,
    minimum_duration: float,
) -> dict[str, object]:
    selected = select_long_monophonic_run(
        notes,
        search_start=search_start,
        count=note_count,
        minimum_duration=minimum_duration,
    )
    boundaries = _detect_run_boundaries(
        audio_samples,
        sample_rate,
        len(audio_samples) / sample_rate,
        notes,
        selected,
    )
    excerpt_start = boundaries[0]
    excerpt_end = boundaries[-1]
    from audio2instrument.audio import AudioData

    source = slice_audio(AudioData(audio_samples, sample_rate), excerpt_start, excerpt_end)
    levels: list[float] = []
    for onset in boundaries[:-1]:
        first = round(onset * sample_rate)
        last = min(len(audio_samples), first + round(0.23 * sample_rate))
        levels.append(onset_level(audio_samples[first:last], sample_rate))
    velocities = levels_to_velocities(levels, reference_level=target_level)

    render_notes: list[RenderNote] = []
    midi_notes: list[NoteEvent] = []
    for index, note in enumerate(selected):
        start = boundaries[index] - excerpt_start
        end = boundaries[index + 1] - excerpt_start
        velocity = velocities[index]
        render_notes.append(RenderNote(start=start, duration=end - start, note=note.note, velocity=velocity))
        midi_notes.append(
            NoteEvent(start=start, end=end, note=note.note, velocity=velocity, channel=note.channel)
        )

    reconstruction = render_multisample_sequence(regions, render_notes)
    reconstruction = _fit_length(reconstruction, len(source))
    pre_gain_rms = rms(reconstruction)
    global_gain = rms(source) / pre_gain_rms if pre_gain_rms else 1.0
    reconstruction *= global_gain

    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "01-original.wav"
    reconstruction_path = output_dir / "02-reconstructed.wav"
    ab_path = output_dir / "03-ab-original-then-reconstructed.wav"
    midi_path = output_dir / "performance.mid"
    write_audio(source_path, source, sample_rate)
    write_audio(reconstruction_path, reconstruction, sample_rate)
    write_audio(
        ab_path,
        np.concatenate([source, np.zeros(round(0.5 * sample_rate)), reconstruction]),
        sample_rate,
    )
    write_notes(midi_path, midi_notes)

    return {
        "label": label,
        "excerpt": {
            "start_seconds": excerpt_start,
            "end_seconds": excerpt_end,
            "duration_seconds": excerpt_end - excerpt_start,
        },
        "selected_notes": [asdict(note) for note in selected],
        "detected_boundaries_seconds": boundaries,
        "inferred_velocities": velocities,
        "source_onset_levels": levels,
        "global_gain": global_gain,
        "metrics": compare_audio(source, reconstruction, sample_rate),
        "outputs": {
            "source": str(source_path),
            "reconstruction": str(reconstruction_path),
            "ab": str(ab_path),
            "midi": str(midi_path),
        },
    }


def run_bass_multisample_poc(config: BassMultisamplePocConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = config.output_dir / "Samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    audio = read_mono(config.audio_path)
    notes = read_notes(config.midi_path)
    roots, target_level = _build_roots(config, audio.samples, audio.sample_rate, notes)
    key_ranges = assign_key_ranges(
        [root.source_note.note for root in roots],
        low_key=min(note.note for note in notes),
        high_key=max(note.note for note in notes),
    )

    regions: list[SampleRegion] = []
    sfz_regions: list[SfzRegion] = []
    for root in roots:
        note_number = root.source_note.note
        sample_path = samples_dir / root.sample_name
        write_audio(sample_path, root.normalized_samples, audio.sample_rate)
        low_key, high_key = key_ranges[note_number]
        instrument = SampleInstrument(
            samples=root.normalized_samples,
            sample_rate=audio.sample_rate,
            root_note=note_number,
            release=config.release,
            loop_start=root.loop_start,
            loop_end=root.loop_end,
            loop_crossfade=root.loop_crossfade,
        )
        regions.append(SampleRegion(instrument=instrument, low_key=low_key, high_key=high_key))
        sfz_regions.append(
            SfzRegion(
                sample=root.sample_name,
                root_key=note_number,
                low_key=low_key,
                high_key=high_key,
                release=config.release,
                loop_start=root.loop_start,
                loop_end=root.loop_end,
                loop_crossfade=root.loop_crossfade / audio.sample_rate,
            )
        )

    sfz_path = config.output_dir / "Bass-Multisample.sfz"
    sfz_path.write_text(render_sfz(sfz_regions), encoding="utf-8")
    validations = [
        _render_validation(
            label="same-segment upper bound",
            output_dir=config.output_dir / "validation-same",
            audio_samples=audio.samples,
            sample_rate=audio.sample_rate,
            notes=notes,
            regions=regions,
            target_level=target_level,
            search_start=config.same_segment_start,
            note_count=config.validation_note_count,
            minimum_duration=config.validation_minimum_duration,
        ),
        _render_validation(
            label="held-out performance",
            output_dir=config.output_dir / "validation-held-out",
            audio_samples=audio.samples,
            sample_rate=audio.sample_rate,
            notes=notes,
            regions=regions,
            target_level=target_level,
            search_start=config.held_out_start,
            note_count=config.validation_note_count,
            minimum_duration=config.validation_minimum_duration,
        ),
    ]

    report: dict[str, object] = {
        "input": {
            "audio": str(config.audio_path),
            "midi": str(config.midi_path),
            "sample_rate": audio.sample_rate,
        },
        "instrument": {
            "target_onset_level": target_level,
            "observed_key_range": [min(note.note for note in notes), max(note.note for note in notes)],
            "sfz": str(sfz_path),
            "roots": [
                {
                    "midi_note": root.source_note.note,
                    "source_note": asdict(root.source_note),
                    "detected_onset": root.onset,
                    "source_level": root.source_level,
                    "normalization_gain": target_level / root.source_level if root.source_level else 1.0,
                    "key_range": list(key_ranges[root.source_note.note]),
                    "sample": str(samples_dir / root.sample_name),
                    "loop_start": root.loop_start,
                    "loop_end": root.loop_end,
                    "loop_crossfade": root.loop_crossfade,
                    "loop_score": root.loop_score,
                }
                for root in roots
            ],
        },
        "validations": validations,
    }
    report_path = config.output_dir / "comparison-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    return report
