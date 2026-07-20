from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from audio2instrument.audio import apply_edge_fades, detect_onset_near, read_mono, rms, slice_audio, write_audio
from audio2instrument.loop import find_loop_points
from audio2instrument.metrics import compare_audio
from audio2instrument.midi import NoteEvent, read_notes, write_notes
from audio2instrument.render import RenderNote, SampleInstrument, render_sequence
from audio2instrument.sfz import SfzRegion, render_sfz


@dataclass(frozen=True, slots=True)
class BassPocConfig:
    audio_path: Path
    midi_path: Path
    output_dir: Path
    search_start: float = 100.0
    note_count: int = 3
    minimum_duration: float = 2.5
    sample_duration: float = 1.35
    release: float = 0.08


def select_long_monophonic_run(notes: list[NoteEvent], *, search_start: float, count: int, minimum_duration: float, maximum_gap: float = 0.08) -> list[NoteEvent]:
    candidates = [note for note in notes if note.start >= search_start]
    for index in range(len(candidates) - count + 1):
        run = candidates[index : index + count]
        if any(note.duration < minimum_duration for note in run):
            continue
        if any(run[i + 1].start - run[i].end > maximum_gap for i in range(len(run) - 1)):
            continue
        return run
    raise ValueError("no suitable long-note run found")


def _next_note(notes: list[NoteEvent], last: NoteEvent) -> NoteEvent | None:
    return next((note for note in notes if note.start >= last.end - 1e-6 and note.start > last.start), None)


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length].copy()
    return np.pad(samples, (0, length - len(samples)))


def _onset_gain(audio_samples: np.ndarray, sample_rate: int, onset: float, duration: float = 0.2) -> float:
    first = round((onset + 0.03) * sample_rate)
    last = min(len(audio_samples), round((onset + 0.03 + duration) * sample_rate))
    return rms(audio_samples[first:last])


def run_bass_poc(config: BassPocConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = config.output_dir / "Samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    audio = read_mono(config.audio_path)
    all_notes = read_notes(config.midi_path)
    selected = select_long_monophonic_run(all_notes, search_start=config.search_start, count=config.note_count, minimum_duration=config.minimum_duration)
    following = _next_note(all_notes, selected[-1])
    expected_boundaries = [note.start for note in selected]
    expected_boundaries.append(following.start if following is not None else selected[-1].end)
    detected_boundaries = [detect_onset_near(audio, value) for value in expected_boundaries]

    excerpt_start = detected_boundaries[0]
    excerpt_end = detected_boundaries[-1]
    source_excerpt = slice_audio(audio, excerpt_start, excerpt_end)

    root_note = selected[0].note
    root_end = min(excerpt_start + config.sample_duration, detected_boundaries[1] - 0.03)
    root_sample = slice_audio(audio, excerpt_start, root_end)
    root_sample = apply_edge_fades(root_sample, audio.sample_rate, fade_in=0.002, fade_out=0.0)
    loop = find_loop_points(root_sample, audio.sample_rate, search_start=0.4, search_end=min(len(root_sample) / audio.sample_rate - 0.04, 1.3), min_duration=0.25, max_duration=0.7, crossfade=0.03)

    sample_name = f"Bass_{root_note}.wav"
    sample_path = samples_dir / sample_name
    write_audio(sample_path, root_sample, audio.sample_rate)

    onset_levels = [_onset_gain(audio.samples, audio.sample_rate, onset) for onset in detected_boundaries[:-1]]
    base_level = onset_levels[0] or 1.0
    rendered_notes: list[RenderNote] = []
    adjusted_midi_notes: list[NoteEvent] = []
    for index, note in enumerate(selected):
        start = detected_boundaries[index] - excerpt_start
        end = detected_boundaries[index + 1] - excerpt_start
        gain = onset_levels[index] / base_level
        rendered_notes.append(RenderNote(start=start, duration=end - start, note=note.note, velocity=64, gain=gain))
        adjusted_midi_notes.append(NoteEvent(start=start, end=end, note=note.note, velocity=64, channel=note.channel))

    instrument = SampleInstrument(samples=root_sample, sample_rate=audio.sample_rate, root_note=root_note, release=config.release, loop_start=loop.start, loop_end=loop.end, loop_crossfade=loop.crossfade)
    reconstruction = render_sequence(instrument, rendered_notes)
    reconstruction = _fit_length(reconstruction, len(source_excerpt))
    reconstruction_rms = rms(reconstruction)
    global_gain = rms(source_excerpt) / reconstruction_rms if reconstruction_rms else 1.0
    reconstruction *= global_gain

    source_path = config.output_dir / "01-original-bass-excerpt.wav"
    reconstruction_path = config.output_dir / "02-reconstructed-bass-excerpt.wav"
    ab_path = config.output_dir / "03-ab-original-then-reconstructed.wav"
    midi_path = config.output_dir / "bass-poc.mid"
    sfz_path = config.output_dir / "Bass.sfz"
    report_path = config.output_dir / "comparison-report.json"

    write_audio(source_path, source_excerpt, audio.sample_rate)
    write_audio(reconstruction_path, reconstruction, audio.sample_rate)
    silence = np.zeros(round(0.5 * audio.sample_rate))
    write_audio(ab_path, np.concatenate([source_excerpt, silence, reconstruction]), audio.sample_rate)
    write_notes(midi_path, adjusted_midi_notes)
    sfz_path.write_text(render_sfz([SfzRegion(sample=sample_name, root_key=root_note, low_key=min(note.note for note in selected), high_key=max(note.note for note in selected), release=config.release, loop_start=loop.start, loop_end=loop.end, loop_crossfade=loop.crossfade / audio.sample_rate)]), encoding="utf-8")

    metrics = compare_audio(source_excerpt, reconstruction, audio.sample_rate)
    report: dict[str, object] = {
        "input": {"audio": str(config.audio_path), "midi": str(config.midi_path), "sample_rate": audio.sample_rate},
        "excerpt": {"start_seconds": excerpt_start, "end_seconds": excerpt_end, "duration_seconds": excerpt_end - excerpt_start},
        "selected_notes": [asdict(note) for note in selected],
        "detected_boundaries_seconds": detected_boundaries,
        "onset_offsets_seconds": [detected - expected for detected, expected in zip(detected_boundaries, expected_boundaries, strict=True)],
        "sample": {"root_note": root_note, "duration_seconds": len(root_sample) / audio.sample_rate, "loop_start": loop.start, "loop_end": loop.end, "loop_crossfade": loop.crossfade, "loop_score": loop.score},
        "per_note_gains": [note.gain for note in rendered_notes],
        "global_gain": global_gain,
        "metrics": metrics,
        "outputs": {"source": str(source_path), "reconstruction": str(reconstruction_path), "ab": str(ab_path), "midi": str(midi_path), "sfz": str(sfz_path), "sample": str(sample_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
