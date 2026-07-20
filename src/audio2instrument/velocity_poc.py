from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from audio2instrument.audio import AudioData, detect_onset_near, read_mono, rms, slice_audio, write_audio
from audio2instrument.expressive import (
    RenderEvent,
    build_root_layers,
    hybridize_round_robin_layers,
    render_adaptive_sequence,
)
from audio2instrument.metrics import compare_audio
from audio2instrument.midi import NoteEvent, read_notes
from audio2instrument.velocity import (
    AttackCandidate,
    build_velocity_root_layers,
    collect_attack_candidates,
    extract_timbre_features,
    infer_velocity_from_candidates,
    render_velocity_sequence,
    render_velocity_sfz,
    write_velocity_samples,
)


@dataclass(frozen=True, slots=True)
class BassVelocityPocConfig:
    audio_path: Path
    midi_path: Path
    output_dir: Path
    root_notes: tuple[int, ...] = (35, 43, 44, 45, 47)
    sample_duration: float = 1.35
    release: float = 0.08
    maximum_attack_spectral_distance: float = 1.0
    validation_segments: tuple[tuple[str, float, float], ...] = (
        ("validation-mixed", 49.50, 53.35),
        ("validation-repetition", 125.00, 129.60),
        ("validation-dynamic-line", 83.50, 88.55),
    )


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
    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1] + 0.035:
            midi_gap = selected[index].start - selected[index - 1].start
            boundaries[index] = boundaries[index - 1] + max(0.035, midi_gap)
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


def _feature_at(samples: np.ndarray, sample_rate: int, onset: float) -> object:
    first = max(0, round(onset * sample_rate))
    return extract_timbre_features(samples[first : first + round(0.20 * sample_rate)], sample_rate)


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _timbre_tracking(
    reference: np.ndarray,
    baseline: np.ndarray,
    velocity: np.ndarray,
    sample_rate: int,
    onset_times: list[float],
) -> dict[str, object]:
    ref = [_feature_at(reference, sample_rate, value) for value in onset_times]
    base = [_feature_at(baseline, sample_rate, value) for value in onset_times]
    vel = [_feature_at(velocity, sample_rate, value) for value in onset_times]
    keys = (
        "level",
        "spectral_centroid_hz",
        "high_frequency_ratio",
        "attack_slope_db_per_second",
    )
    output: dict[str, object] = {}
    for key in keys:
        reference_values = [float(getattr(item, key)) for item in ref]
        baseline_values = [float(getattr(item, key)) for item in base]
        velocity_values = [float(getattr(item, key)) for item in vel]
        output[key] = {
            "reference": reference_values,
            "v3": baseline_values,
            "v4": velocity_values,
            "v3_correlation": _correlation(reference_values, baseline_values),
            "v4_correlation": _correlation(reference_values, velocity_values),
        }
    return output


def _validation(
    label: str,
    start: float,
    end: float,
    audio: AudioData,
    notes: list[NoteEvent],
    v3_layers,
    v4_layers,
    calibration: dict[int, list[AttackCandidate]],
    roots: tuple[int, ...],
    output_dir: Path,
    release: float,
) -> dict[str, object]:
    selected = _select_range(notes, start, end)
    if not selected:
        raise ValueError(f"no notes selected for {label}")
    boundaries = _boundaries_for(audio, selected, release)
    excerpt_start, excerpt_end = boundaries[0], boundaries[-1]
    reference = slice_audio(audio, excerpt_start, excerpt_end)
    velocities: list[int] = []
    original_features = []
    for boundary, note in zip(boundaries[:-1], selected, strict=True):
        absolute_first = round(boundary * audio.sample_rate)
        features = extract_timbre_features(
            audio.samples[absolute_first : absolute_first + round(0.20 * audio.sample_rate)],
            audio.sample_rate,
        )
        root = min(roots, key=lambda value: (abs(value - note.note), value))
        velocities.append(infer_velocity_from_candidates(features, calibration[root]))
        original_features.append(features)
    events = [
        RenderEvent(
            start=boundaries[index] - excerpt_start,
            duration=max(0.04, boundaries[index + 1] - boundaries[index]),
            note=note.note,
            velocity=velocities[index],
        )
        for index, note in enumerate(selected)
    ]
    baseline = render_adaptive_sequence(
        v3_layers,
        events,
        compatible_pairs=set(),
        release=release,
        round_robin=True,
    )
    enhanced = render_velocity_sequence(v4_layers, events, release=release, round_robin=True)
    baseline = _fit_length(baseline, len(reference))
    enhanced = _fit_length(enhanced, len(reference))
    baseline *= rms(reference) / max(rms(baseline), 1e-12)
    enhanced *= rms(reference) / max(rms(enhanced), 1e-12)

    target = output_dir / label
    target.mkdir(parents=True, exist_ok=True)
    write_audio(target / "01-original.wav", reference, audio.sample_rate)
    write_audio(target / "02-expressive-v3.wav", baseline, audio.sample_rate)
    write_audio(target / "03-velocity-v4.wav", enhanced, audio.sample_rate)
    silence = np.zeros(round(0.45 * audio.sample_rate), dtype=np.float64)
    write_audio(
        target / "04-original-v3-v4.wav",
        np.concatenate([reference, silence, baseline, silence, enhanced]),
        audio.sample_rate,
    )
    relative_onsets = [boundary - excerpt_start for boundary in boundaries[:-1]]
    return {
        "label": label,
        "excerpt": {
            "start": excerpt_start,
            "end": excerpt_end,
            "duration": excerpt_end - excerpt_start,
        },
        "notes": [asdict(note) for note in selected],
        "velocities": velocities,
        "original_timbre": [asdict(item) for item in original_features],
        "v3": compare_audio(reference, baseline, audio.sample_rate),
        "v4": compare_audio(reference, enhanced, audio.sample_rate),
        "timbre_tracking": _timbre_tracking(
            reference,
            baseline,
            enhanced,
            audio.sample_rate,
            relative_onsets,
        ),
    }


def run_bass_velocity_poc(config: BassVelocityPocConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    audio = read_mono(config.audio_path)
    notes = read_notes(config.midi_path)
    excluded = tuple((start, end) for _, start, end in config.validation_segments)
    training_notes = [
        note for note in notes if not any(start <= note.start < end for start, end in excluded)
    ]
    raw_layers, target_level = build_root_layers(
        audio,
        training_notes,
        config.root_notes,
        sample_duration=config.sample_duration,
        maximum_variants=3,
        minimum_duration=1.25,
    )
    v3_layers = hybridize_round_robin_layers(
        raw_layers,
        attack_duration=0.09,
        transition_duration=0.05,
        maximum_attack_spectral_distance=0.25,
    )
    v4_layers = build_velocity_root_layers(
        audio,
        notes,
        v3_layers,
        excluded_ranges=excluded,
        maximum_velocity_layers=3,
        maximum_round_robins=2,
        maximum_attack_spectral_distance=config.maximum_attack_spectral_distance,
    )
    calibration = {
        root: collect_attack_candidates(audio, notes, root, excluded_ranges=excluded)
        for root in config.root_notes
    }
    write_velocity_samples(config.output_dir / "Samples", v4_layers)
    sfz = render_velocity_sfz(
        v4_layers,
        low_key=min(note.note for note in notes),
        high_key=max(note.note for note in notes),
        release=config.release,
    )
    sfz_path = config.output_dir / "Bass-Velocity-v4.sfz"
    sfz_path.write_text(sfz, encoding="utf-8")
    validations = [
        _validation(
            label,
            start,
            end,
            audio,
            notes,
            v3_layers,
            v4_layers,
            calibration,
            config.root_notes,
            config.output_dir,
            config.release,
        )
        for label, start, end in config.validation_segments
    ]
    report: dict[str, object] = {
        "input": {"audio": str(config.audio_path), "midi": str(config.midi_path)},
        "instrument": {
            "target_onset_level": target_level,
            "sfz": str(sfz_path),
            "excluded_validation_ranges": excluded,
            "roots": [
                {
                    "root_note": root.root_note,
                    "candidate_count": len(calibration[root.root_note]),
                    "layers": [
                        {
                            "velocity_range": [layer.low_velocity, layer.high_velocity],
                            "center_velocity": layer.center_velocity,
                            "source_scores": layer.source_scores,
                            "variants": [
                                {
                                    "name": item.name,
                                    "source_note": asdict(item.source_note),
                                    "detected_onset": item.detected_onset,
                                }
                                for item in layer.variants
                            ],
                        }
                        for layer in root.layers
                    ],
                }
                for root in v4_layers
            ],
        },
        "validations": validations,
    }
    (config.output_dir / "comparison-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
