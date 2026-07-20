from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from audio2instrument.audio import AudioData, detect_onset_near, read_mono, rms, slice_audio, write_audio
from audio2instrument.expressive import (
    RenderEvent,
    build_root_layers,
    compatible_root_pairs,
    hybridize_round_robin_layers,
    infer_velocities,
    onset_level,
    onset_spectral_diversity,
    render_adaptive_sequence,
    render_expressive_sequence,
    render_sfz_adaptive,
    root_pair_similarity,
    write_samples,
)
from audio2instrument.metrics import compare_audio
from audio2instrument.midi import NoteEvent, read_notes


@dataclass(frozen=True, slots=True)
class BassExpressivePocConfig:
    audio_path: Path
    midi_path: Path
    output_dir: Path
    root_notes: tuple[int, ...] = (35, 43, 44, 45, 47)
    sample_duration: float = 1.35
    maximum_round_robins: int = 3
    release: float = 0.08
    crossfade_similarity_threshold: float = 0.94
    repetition_segment: tuple[float, float] = (158.19, 159.95)
    range_segment: tuple[float, float] = (164.60, 167.07)
    held_out_segment: tuple[float, float] = (170.60, 176.22)


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length].copy()
    return np.pad(samples, (0, length - len(samples)))


def _select_range(notes: list[NoteEvent], start: float, end: float) -> list[NoteEvent]:
    return [note for note in notes if start <= note.start < end]


def _boundaries_for(audio: AudioData, selected: list[NoteEvent], release: float) -> list[float]:
    fast = float(np.median([note.duration for note in selected])) < 0.5
    before = 0.035 if fast else 0.12
    after = 0.055 if fast else 0.15
    boundaries = [
        detect_onset_near(audio, note.start, search_before=before, search_after=after)
        for note in selected
    ]
    minimum_gap = 0.035
    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1] + minimum_gap:
            midi_gap = selected[index].start - selected[index - 1].start
            boundaries[index] = boundaries[index - 1] + max(minimum_gap, midi_gap)
    offset = float(
        np.median(
            [
                boundary - note.start
                for boundary, note in zip(boundaries, selected, strict=True)
            ]
        )
    )
    final = max(boundaries[-1] + 0.04, selected[-1].end + offset + release)
    return boundaries + [min(audio.duration, final)]


def _validate(
    label: str,
    audio: AudioData,
    notes: list[NoteEvent],
    layers,
    compatible_pairs: set[tuple[int, int]],
    segment: tuple[float, float],
    output_dir: Path,
    release: float,
) -> dict:
    target = output_dir / label
    target.mkdir(parents=True, exist_ok=True)
    selected = _select_range(notes, *segment)
    if not selected:
        raise ValueError(f"no notes selected for {label}")
    boundaries = _boundaries_for(audio, selected, release)
    excerpt_start, excerpt_end = boundaries[0], boundaries[-1]
    reference = slice_audio(audio, excerpt_start, excerpt_end)
    levels: list[float] = []
    for boundary in boundaries[:-1]:
        first = round(boundary * audio.sample_rate)
        last = min(len(audio.samples), first + round(0.18 * audio.sample_rate))
        levels.append(
            onset_level(
                audio.samples[first:last],
                audio.sample_rate,
                offset=0.02,
                duration=0.14,
            )
        )
    velocities = infer_velocities(levels)
    events = [
        RenderEvent(
            start=boundaries[index] - excerpt_start,
            duration=max(0.04, boundaries[index + 1] - boundaries[index]),
            note=note.note,
            velocity=velocities[index],
        )
        for index, note in enumerate(selected)
    ]
    baseline = render_expressive_sequence(
        layers,
        events,
        release=release,
        crossfade_roots=False,
        round_robin=False,
    )
    expressive = render_adaptive_sequence(
        layers,
        events,
        compatible_pairs=compatible_pairs,
        release=release,
        round_robin=True,
    )
    baseline = _fit_length(baseline, len(reference))
    expressive = _fit_length(expressive, len(reference))
    baseline *= rms(reference) / max(rms(baseline), 1e-12)
    expressive *= rms(reference) / max(rms(expressive), 1e-12)

    write_audio(target / "01-original.wav", reference, audio.sample_rate)
    write_audio(target / "02-hard-zone-single-variant.wav", baseline, audio.sample_rate)
    write_audio(target / "03-expressive-v3.wav", expressive, audio.sample_rate)
    silence = np.zeros(round(0.45 * audio.sample_rate), dtype=np.float64)
    write_audio(
        target / "04-original-baseline-v3.wav",
        np.concatenate([reference, silence, baseline, silence, expressive]),
        audio.sample_rate,
    )

    relative_onsets = [boundary - excerpt_start for boundary in boundaries[:-1]]
    repetition_diversity: dict[str, dict[str, float | int]] = {}
    for pitch in sorted(set(note.note for note in selected)):
        pitch_onsets = [
            relative_onsets[index]
            for index, note in enumerate(selected)
            if note.note == pitch
        ]
        if len(pitch_onsets) >= 3:
            repetition_diversity[str(pitch)] = {
                "count": len(pitch_onsets),
                "reference_diversity": onset_spectral_diversity(
                    reference, audio.sample_rate, pitch_onsets
                ),
                "baseline_diversity": onset_spectral_diversity(
                    baseline, audio.sample_rate, pitch_onsets
                ),
                "v3_diversity": onset_spectral_diversity(
                    expressive, audio.sample_rate, pitch_onsets
                ),
            }
    return {
        "label": label,
        "excerpt": {
            "start": excerpt_start,
            "end": excerpt_end,
            "duration": excerpt_end - excerpt_start,
        },
        "notes": [asdict(note) for note in selected],
        "detected_boundaries": boundaries,
        "velocities": velocities,
        "baseline": compare_audio(reference, baseline, audio.sample_rate),
        "v3": compare_audio(reference, expressive, audio.sample_rate),
        "repetition_diversity": repetition_diversity,
    }


def run_bass_expressive_poc(config: BassExpressivePocConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    audio = read_mono(config.audio_path)
    notes = read_notes(config.midi_path)
    layers, target_level = build_root_layers(
        audio,
        notes,
        config.root_notes,
        sample_duration=config.sample_duration,
        maximum_variants=config.maximum_round_robins,
        minimum_duration=1.25,
    )
    pair_similarities = {
        f"{left.root_note}-{right.root_note}": root_pair_similarity(left, right)
        for left, right in zip(layers, layers[1:])
        if right.root_note - left.root_note >= 2
    }
    compatible = compatible_root_pairs(
        layers,
        threshold=config.crossfade_similarity_threshold,
    )
    compatible_pairs = set(compatible)
    layers = hybridize_round_robin_layers(
        layers,
        attack_duration=0.09,
        transition_duration=0.05,
        maximum_attack_spectral_distance=0.25,
    )
    write_samples(config.output_dir / "Samples", layers)
    sfz = render_sfz_adaptive(
        layers,
        low_key=min(note.note for note in notes),
        high_key=max(note.note for note in notes),
        compatible_pairs=compatible_pairs,
        release=config.release,
    )
    sfz_path = config.output_dir / "Bass-Expressive-v3.sfz"
    sfz_path.write_text(sfz, encoding="utf-8")

    validations = [
        _validate(
            "validation-repetition",
            audio,
            notes,
            layers,
            compatible_pairs,
            config.repetition_segment,
            config.output_dir,
            config.release,
        ),
        _validate(
            "validation-range",
            audio,
            notes,
            layers,
            compatible_pairs,
            config.range_segment,
            config.output_dir,
            config.release,
        ),
        _validate(
            "validation-held-out-long",
            audio,
            notes,
            layers,
            compatible_pairs,
            config.held_out_segment,
            config.output_dir,
            config.release,
        ),
    ]
    report = {
        "input": {"audio": str(config.audio_path), "midi": str(config.midi_path)},
        "instrument": {
            "target_onset_level": target_level,
            "root_pair_similarities": pair_similarities,
            "compatible_crossfade_pairs": {
                f"{left}-{right}": score for (left, right), score in compatible.items()
            },
            "crossfade_similarity_threshold": config.crossfade_similarity_threshold,
            "round_robin_mode": (
                "alternate 90 ms attack, 50 ms transition to canonical sustain, "
                "maximum normalized attack-spectrum distance 0.25"
            ),
            "sfz": str(sfz_path),
            "roots": [
                {
                    "root_note": layer.root_note,
                    "variants": [
                        {
                            "name": variant.name,
                            "source_note": asdict(variant.source_note),
                            "detected_onset": variant.detected_onset,
                            "source_level": variant.source_level,
                            "loop": asdict(variant.loop),
                        }
                        for variant in layer.variants
                    ],
                }
                for layer in layers
            ],
        },
        "validations": validations,
    }
    (config.output_dir / "comparison-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
