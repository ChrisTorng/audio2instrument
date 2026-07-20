from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np

from audio2instrument.audio import (
    AudioData,
    detect_onset_near,
    read_mono,
    rms,
    slice_audio,
    write_audio,
)
from audio2instrument.metrics import compare_audio
from audio2instrument.midi import NoteEvent, read_notes, write_notes
from audio2instrument.sample_layout import instrument_sample_directory, sfz_default_path


@dataclass(frozen=True, slots=True)
class ElectricGuitarRiskConfig:
    audio_path: Path
    midi_path: Path
    output_dir: Path
    target_pattern: tuple[int, ...] = (57, 61, 64)
    heldout_start: float = 27.30
    heldout_end: float = 28.50
    source_chord_start: float = 85.20
    source_chord_end: float = 86.20
    sample_duration: float = 0.62


def cluster_notes(notes: list[NoteEvent], tolerance: float = 0.015) -> list[list[NoteEvent]]:
    ordered = sorted(notes, key=lambda item: (item.start, item.note))
    result: list[list[NoteEvent]] = []
    index = 0
    while index < len(ordered):
        start = ordered[index].start
        group = [ordered[index]]
        index += 1
        while index < len(ordered) and ordered[index].start - start <= tolerance:
            group.append(ordered[index])
            index += 1
        result.append(group)
    return result


def note_is_isolated(note: NoteEvent, notes: list[NoteEvent], margin: float = 0.01) -> bool:
    return not any(
        other is not note
        and other.start < note.end - margin
        and other.end > note.start + margin
        for other in notes
    )


def _edge_fade(samples: np.ndarray, rate: int) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64).copy()
    fade_in = min(len(result), round(0.002 * rate))
    fade_out = min(len(result), round(0.020 * rate))
    if fade_in:
        result[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    if fade_out:
        result[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    return result


def _render(events: list[tuple[float, np.ndarray]], rate: int, duration: float) -> np.ndarray:
    output = np.zeros(max(1, round(duration * rate)), dtype=np.float64)
    for start, sample in events:
        first = round(start * rate)
        if first >= len(output):
            continue
        last = min(len(output), first + len(sample))
        output[first:last] += sample[: last - first]
    return output


def _match_rms(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return estimate * (rms(reference) / max(rms(estimate), 1e-12))


def _select_clean_note_sample(
    audio: AudioData,
    notes: list[NoteEvent],
    pitch: int,
    excluded: tuple[float, float],
    duration: float,
) -> tuple[NoteEvent, float, np.ndarray, float]:
    candidates = [
        note
        for note in notes
        if note.note == pitch
        and not (excluded[0] <= note.start < excluded[1])
        and note_is_isolated(note, notes)
    ]
    if not candidates:
        raise ValueError(f"no isolated sample for MIDI note {pitch}")
    scored = []
    for note in candidates:
        onset = detect_onset_near(
            audio,
            note.start,
            search_before=0.08,
            search_after=0.10,
        )
        sample = slice_audio(audio, onset, onset + duration)
        post = rms(sample[: round(0.12 * audio.sample_rate)])
        pre = rms(
            slice_audio(
                audio,
                max(0.0, onset - 0.08),
                max(0.0, onset - 0.01),
            )
        )
        scored.append((post / max(pre, 1e-5), note.duration, note, onset, sample))
    quality, _, note, onset, sample = max(scored, key=lambda item: (item[0], item[1]))
    return note, onset, _edge_fade(sample, audio.sample_rate), quality


def run_electric_guitar_risk_poc(config: ElectricGuitarRiskConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    note_dir = config.output_dir / instrument_sample_directory("ElectricGuitar", "Notes")
    chord_dir = config.output_dir / instrument_sample_directory("ElectricGuitar", "Chords")
    note_dir.mkdir(parents=True, exist_ok=True)
    chord_dir.mkdir(parents=True, exist_ok=True)

    audio = read_mono(config.audio_path)
    notes = read_notes(config.midi_path)
    pattern = tuple(sorted(config.target_pattern))
    matching = [
        group
        for group in cluster_notes(notes)
        if tuple(sorted(item.note for item in group)) == pattern
    ]
    heldout = [
        group
        for group in matching
        if config.heldout_start <= group[0].start < config.heldout_end
    ][:5]
    sources = [
        group
        for group in matching
        if config.source_chord_start <= group[0].start < config.source_chord_end
    ]
    if len(heldout) < 2 or not sources:
        raise ValueError("configured electric-guitar chord occurrences were not found")

    heldout_midi_start = heldout[0][0].start
    heldout_audio_start = detect_onset_near(
        audio,
        heldout_midi_start,
        search_before=0.10,
        search_after=0.12,
    )
    score_offset = heldout_audio_start - heldout_midi_start
    starts = [group[0].start + score_offset - heldout_audio_start for group in heldout]
    duration = starts[-1] + 0.72
    reference = slice_audio(audio, heldout_audio_start, heldout_audio_start + duration)

    note_samples: dict[int, np.ndarray] = {}
    note_sources = []
    for pitch in pattern:
        source_note, onset, sample, quality = _select_clean_note_sample(
            audio,
            notes,
            pitch,
            (config.heldout_start, config.heldout_end),
            config.sample_duration,
        )
        filename = f"EG_note_{pitch}.wav"
        write_audio(note_dir / filename, sample, audio.sample_rate)
        note_samples[pitch] = sample
        note_sources.append(
            {
                "pitch": pitch,
                "source_note": asdict(source_note),
                "detected_onset": onset,
                "quality_ratio": quality,
                "sample": f"Samples/ElectricGuitar/Notes/{filename}",
            }
        )

    note_render = _render(
        [(start, note_samples[pitch]) for start in starts for pitch in pattern],
        audio.sample_rate,
        duration,
    )
    note_render = _match_rms(reference, note_render)

    source_group = sources[0]
    chord_onset = detect_onset_near(
        audio,
        source_group[0].start,
        search_before=0.10,
        search_after=0.12,
    )
    chord_sample = _edge_fade(
        slice_audio(audio, chord_onset, chord_onset + config.sample_duration),
        audio.sample_rate,
    )
    chord_filename = "EG_chord_57_61_64.wav"
    write_audio(chord_dir / chord_filename, chord_sample, audio.sample_rate)
    chord_render = _render(
        [(start, chord_sample) for start in starts],
        audio.sample_rate,
        duration,
    )
    chord_render = _match_rms(reference, chord_render)

    write_audio(config.output_dir / "01-original-guitar.wav", reference, audio.sample_rate)
    write_audio(config.output_dir / "02-note-sampler.wav", note_render, audio.sample_rate)
    write_audio(config.output_dir / "03-chord-one-shot.wav", chord_render, audio.sample_rate)
    silence = np.zeros(round(0.45 * audio.sample_rate), dtype=np.float64)
    write_audio(
        config.output_dir / "04-original-note-chord.wav",
        np.concatenate([reference, silence, note_render, silence, chord_render]),
        audio.sample_rate,
    )

    note_sfz = [
        "<control>",
        f"default_path={sfz_default_path('ElectricGuitar', 'Notes')}",
        "",
        "<global>",
        "ampeg_release=0.08",
        "",
    ]
    for pitch in pattern:
        note_sfz.extend(
            [
                "<region>",
                f"sample=EG_note_{pitch}.wav",
                f"key={pitch}",
                f"pitch_keycenter={pitch}",
                "",
            ]
        )
    (config.output_dir / "ElectricGuitar-ExactNotes.sfz").write_text(
        "\n".join(note_sfz).rstrip() + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "ElectricGuitar-ChordOneShot.sfz").write_text(
        "\n".join(
            [
                "<control>",
                f"default_path={sfz_default_path('ElectricGuitar', 'Chords')}",
                "",
                "<region>",
                f"sample={chord_filename}",
                "key=24",
                "pitch_keycenter=24",
                "loop_mode=one_shot",
                "",
            ]
        ),
        encoding="utf-8",
    )

    note_midi: list[NoteEvent] = []
    chord_midi: list[NoteEvent] = []
    for start, group in zip(starts, heldout, strict=True):
        for item in group:
            note_midi.append(
                NoteEvent(start, start + item.duration, item.note, item.velocity, 0)
            )
        chord_midi.append(
            NoteEvent(start, start + max(item.duration for item in group), 24, 100, 0)
        )
    write_notes(config.output_dir / "ElectricGuitar-ExactNotes-validation.mid", note_midi)
    write_notes(config.output_dir / "ElectricGuitar-ChordOneShot-validation.mid", chord_midi)

    report: dict[str, object] = {
        "instrument": "ElectricGuitar",
        "sample_directory": "Samples/ElectricGuitar",
        "heldout": {
            "midi_start": heldout_midi_start,
            "detected_audio_start": heldout_audio_start,
            "score_offset_seconds": score_offset,
            "pattern": list(pattern),
            "repetitions": len(heldout),
        },
        "note_sources": note_sources,
        "chord_source": {
            "midi_start": source_group[0].start,
            "detected_onset": chord_onset,
            "sample": f"Samples/ElectricGuitar/Chords/{chord_filename}",
        },
        "metrics": {
            "note_sampler": compare_audio(reference, note_render, audio.sample_rate),
            "chord_one_shot": compare_audio(reference, chord_render, audio.sample_rate),
        },
    }
    (config.output_dir / "comparison-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
