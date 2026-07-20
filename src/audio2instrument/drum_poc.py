from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
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
class DrumPieceConfig:
    name: str
    audio_path: Path
    midi_path: Path
    sample_duration: float
    first_target_key: int


@dataclass(frozen=True, slots=True)
class DrumKitPocConfig:
    pieces: tuple[DrumPieceConfig, ...]
    output_dir: Path
    validation_start: float = 156.0
    validation_end: float = 160.0
    validation_tail: float = 0.90


@dataclass(frozen=True, slots=True)
class DrumSample:
    piece: str
    source_pitch: int
    layer: int
    rr: int
    source_time: float
    source_level: float
    samples: np.ndarray
    filename: str


def drum_layer_ranges(count: int) -> tuple[tuple[int, int], ...]:
    if count == 1:
        return ((1, 127),)
    if count == 2:
        return ((1, 80), (81, 127))
    if count == 3:
        return ((1, 48), (49, 92), (93, 127))
    raise ValueError("drum velocity layers must be 1, 2, or 3")


def custom_key_map(source_pitches: list[int], first_target_key: int) -> dict[int, int]:
    ordered = sorted(set(source_pitches))
    if not ordered:
        raise ValueError("at least one source pitch is required")
    last = first_target_key + len(ordered) - 1
    if not 0 <= first_target_key <= last <= 127:
        raise ValueError("custom drum key map exceeds MIDI range")
    return {pitch: first_target_key + index for index, pitch in enumerate(ordered)}


def infer_layer(level: float, training_levels: list[float], layer_count: int) -> int:
    if not training_levels:
        return 1
    percentile = float(np.mean(np.asarray(training_levels) <= level))
    if layer_count == 1:
        return 1
    if layer_count == 2:
        return 1 if percentile < 0.58 else 2
    if layer_count == 3:
        return 1 if percentile < 0.33 else 2 if percentile < 0.70 else 3
    raise ValueError("drum velocity layers must be 1, 2, or 3")


def _edge_fade(samples: np.ndarray, rate: int) -> np.ndarray:
    result = np.asarray(samples, dtype=np.float64).copy()
    fade_in = min(len(result), round(0.001 * rate))
    fade_out = min(len(result), round(0.030 * rate))
    if fade_in:
        result[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
    if fade_out:
        result[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)
    return result


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if len(samples) >= length:
        return samples[:length].copy()
    return np.pad(samples, (0, length - len(samples)))


def _match_rms(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return estimate * (rms(reference) / max(rms(estimate), 1e-12))


def _render(events: list[tuple[float, np.ndarray]], rate: int, duration: float) -> np.ndarray:
    output = np.zeros(max(1, round(duration * rate)), dtype=np.float64)
    for start, sample in events:
        first = round(start * rate)
        if first >= len(output):
            continue
        last = min(len(output), first + len(sample))
        output[first:last] += sample[: last - first]
    return output


def _attack_feature(samples: np.ndarray, rate: int) -> np.ndarray:
    count = min(len(samples), round(0.18 * rate))
    values = np.asarray(samples[:count], dtype=np.float64)
    if len(values) < count:
        values = np.pad(values, (0, count - len(values)))
    magnitude = np.abs(np.fft.rfft(values * np.hanning(len(values))))
    result = np.log1p(magnitude)
    return result / max(np.linalg.norm(result), 1e-12)


def _extract_candidate(
    audio: AudioData,
    note: NoteEvent,
    duration: float,
) -> tuple[float, np.ndarray, float, float]:
    onset = detect_onset_near(
        audio,
        note.start,
        search_before=0.045,
        search_after=0.065,
        window=0.010,
        hop=0.001,
    )
    sample = slice_audio(audio, onset, onset + duration)
    expected = round(duration * audio.sample_rate)
    if len(sample) < expected:
        sample = np.pad(sample, (0, expected - len(sample)))
    post = rms(sample[: round(min(0.12, duration) * audio.sample_rate)])
    pre = rms(
        slice_audio(
            audio,
            max(0.0, onset - 0.05),
            max(0.0, onset - 0.006),
        )
    )
    return onset, _edge_fade(sample, audio.sample_rate), post, post / max(pre, 1e-5)


def _collect_articulation_samples(
    piece: DrumPieceConfig,
    source_pitch: int,
    audio: AudioData,
    notes: list[NoteEvent],
    excluded: tuple[float, float],
    output_dir: Path,
) -> tuple[
    list[DrumSample],
    list[float],
    float,
    list[tuple[np.ndarray, np.ndarray, float]],
]:
    training = [
        note
        for note in notes
        if note.note == source_pitch and not (excluded[0] <= note.start < excluded[1])
    ]
    candidates = []
    oracle_pool = []
    offsets = []
    for note in training:
        onset, sample, level, quality = _extract_candidate(
            audio,
            note,
            piece.sample_duration,
        )
        if level < 1e-4:
            continue
        oracle_pool.append((_attack_feature(sample, audio.sample_rate), sample, note.start))
        if quality >= 1.08:
            candidates.append((level, quality, note, onset, sample))
            offsets.append(onset - note.start)
    if not candidates:
        for note in training:
            onset, sample, level, _ = _extract_candidate(
                audio,
                note,
                piece.sample_duration,
            )
            candidates.append((level, 1.0, note, onset, sample))
            offsets.append(onset - note.start)
    if not candidates:
        raise ValueError(f"no candidates for {piece.name} source pitch {source_pitch}")

    candidates.sort(key=lambda item: item[0])
    layer_count = 3 if len(candidates) >= 9 else 2 if len(candidates) >= 4 else 1
    if layer_count == 3:
        quantiles = (0.18, 0.50, 0.82)
    elif layer_count == 2:
        quantiles = (0.30, 0.75)
    else:
        quantiles = (0.50,)
    levels = [item[0] for item in candidates]
    sample_dir = output_dir / instrument_sample_directory("Drums", piece.name)
    sample_dir.mkdir(parents=True, exist_ok=True)
    selected: list[DrumSample] = []
    for layer_index, quantile in enumerate(quantiles, start=1):
        target = float(np.quantile(levels, quantile))
        ranked = sorted(candidates, key=lambda item: (abs(item[0] - target), -item[1]))
        chosen = []
        for item in ranked:
            if any(abs(item[2].start - previous[2].start) < 1.5 for previous in chosen):
                continue
            chosen.append(item)
            if len(chosen) == 3:
                break
        for rr, item in enumerate(chosen, start=1):
            level, _, note, _, sample = item
            filename = f"{piece.name}_src{source_pitch}_vel{layer_index}_rr{rr}.wav"
            write_audio(sample_dir / filename, sample, audio.sample_rate)
            selected.append(
                DrumSample(
                    piece.name,
                    source_pitch,
                    layer_index,
                    rr,
                    note.start,
                    level,
                    sample,
                    f"{piece.name}/{filename}",
                )
            )
    median_offset = float(np.median(offsets)) if offsets else 0.0
    return selected, levels, median_offset, oracle_pool


def run_drum_kit_poc(config: DrumKitPocConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    validation = (config.validation_start, config.validation_end)
    sfz_lines = [
        "<control>",
        f"default_path={sfz_default_path('Drums')}",
        "",
        "<global>",
        "amp_veltrack=0",
        "",
    ]
    key_map_lines = ["instrument,source_pitch,target_key"]
    validation_midi: list[NoteEvent] = []
    report_pieces = []
    reference_parts: dict[str, np.ndarray] = {}
    automatic_parts: dict[str, np.ndarray] = {}
    oracle_parts: dict[str, np.ndarray] = {}
    sample_rate: int | None = None

    for piece in config.pieces:
        audio = read_mono(piece.audio_path)
        notes = read_notes(piece.midi_path)
        if sample_rate is None:
            sample_rate = audio.sample_rate
        elif sample_rate != audio.sample_rate:
            raise ValueError("all drum pieces must use the same sample rate")
        pitches = sorted(set(note.note for note in notes))
        target_keys = custom_key_map(pitches, piece.first_target_key)
        articulation_data: dict[int, dict[str, object]] = {}
        all_selected: list[DrumSample] = []

        for source_pitch in pitches:
            selected, levels, offset, oracle_pool = _collect_articulation_samples(
                piece,
                source_pitch,
                audio,
                notes,
                validation,
                config.output_dir,
            )
            by_layer: dict[int, list[DrumSample]] = defaultdict(list)
            for sample in selected:
                by_layer[sample.layer].append(sample)
            layer_count = len(by_layer)
            ranges = drum_layer_ranges(layer_count)
            target_key = target_keys[source_pitch]
            key_map_lines.append(f"{piece.name},{source_pitch},{target_key}")
            for layer_index, (low, high) in enumerate(ranges, start=1):
                variants = by_layer[layer_index]
                for position, sample in enumerate(variants, start=1):
                    sfz_lines.extend(
                        [
                            "<region>",
                            f"sample={sample.filename}",
                            f"key={target_key}",
                            f"lovel={low}",
                            f"hivel={high}",
                            f"seq_length={len(variants)}",
                            f"seq_position={position}",
                            "loop_mode=one_shot",
                            "",
                        ]
                    )
            articulation_data[source_pitch] = {
                "target_key": target_key,
                "levels": levels,
                "offset": offset,
                "by_layer": by_layer,
                "ranges": ranges,
                "layer_count": layer_count,
                "oracle_pool": oracle_pool,
            }
            all_selected.extend(selected)

        selected_events = [
            note
            for note in notes
            if config.validation_start <= note.start < config.validation_end
        ]
        automatic_events = []
        oracle_events = []
        counters: dict[tuple[int, int], int] = defaultdict(int)
        event_records = []
        for note in selected_events:
            data = articulation_data[note.note]
            onset, original_hit, level, _ = _extract_candidate(
                audio,
                note,
                piece.sample_duration,
            )
            local = onset - config.validation_start
            layer = infer_layer(level, data["levels"], data["layer_count"])
            variants = data["by_layer"][layer]
            counter_key = (note.note, layer)
            position = counters[counter_key] % len(variants)
            counters[counter_key] += 1
            chosen = variants[position]
            automatic_events.append((local, chosen.samples))
            low, high = data["ranges"][layer - 1]
            velocity = int(round((low + high) / 2))
            validation_midi.append(
                NoteEvent(local, local + 0.08, data["target_key"], velocity, 9)
            )

            target_feature = _attack_feature(original_hit, audio.sample_rate)
            pool = data["oracle_pool"]
            if pool:
                oracle_choice = min(
                    pool,
                    key=lambda item: float(np.linalg.norm(target_feature - item[0])),
                )
                oracle_sample = oracle_choice[1].copy()
                attack_count = round(min(0.12, piece.sample_duration) * audio.sample_rate)
                oracle_sample *= rms(original_hit[:attack_count]) / max(
                    rms(oracle_sample[:attack_count]),
                    1e-12,
                )
                oracle_events.append((local, oracle_sample))
                oracle_source = oracle_choice[2]
                oracle_distance = float(np.linalg.norm(target_feature - oracle_choice[0]))
            else:
                oracle_source = None
                oracle_distance = None

            event_records.append(
                {
                    "source_pitch": note.note,
                    "target_key": data["target_key"],
                    "midi_time": note.start,
                    "detected_audio_time": onset,
                    "layer": layer,
                    "rr": chosen.rr,
                    "sample": f"Samples/Drums/{chosen.filename}",
                    "oracle_source_time": oracle_source,
                    "oracle_feature_distance": oracle_distance,
                }
            )

        duration = config.validation_end - config.validation_start + config.validation_tail
        reference = slice_audio(
            audio,
            config.validation_start,
            config.validation_end + config.validation_tail,
        )
        automatic = _render(automatic_events, audio.sample_rate, duration)
        oracle = _render(oracle_events, audio.sample_rate, duration)
        automatic = _match_rms(reference, automatic) if rms(automatic) else automatic
        oracle = _match_rms(reference, oracle) if rms(oracle) else oracle
        reference_parts[piece.name] = reference
        automatic_parts[piece.name] = automatic
        oracle_parts[piece.name] = oracle

        stem_dir = config.output_dir / "Stems" / piece.name
        write_audio(stem_dir / "01-original.wav", reference, audio.sample_rate)
        write_audio(stem_dir / "02-automatic.wav", automatic, audio.sample_rate)
        write_audio(stem_dir / "03-oracle-nearest-hit.wav", oracle, audio.sample_rate)
        silence = np.zeros(round(0.25 * audio.sample_rate), dtype=np.float64)
        write_audio(
            stem_dir / "04-original-auto-oracle.wav",
            np.concatenate([reference, silence, automatic, silence, oracle]),
            audio.sample_rate,
        )
        report_pieces.append(
            {
                "piece": piece.name,
                "sample_directory": f"Samples/Drums/{piece.name}",
                "source_pitch_to_target_key": {
                    str(pitch): target_keys[pitch] for pitch in pitches
                },
                "selected_samples": [
                    {
                        "source_pitch": item.source_pitch,
                        "layer": item.layer,
                        "rr": item.rr,
                        "source_time": item.source_time,
                        "source_level": item.source_level,
                        "sample": f"Samples/Drums/{item.filename}",
                    }
                    for item in all_selected
                ],
                "validation_event_count": len(selected_events),
                "events": event_records,
                "automatic_metrics": compare_audio(
                    reference,
                    automatic,
                    audio.sample_rate,
                ),
                "oracle_metrics": compare_audio(reference, oracle, audio.sample_rate),
            }
        )

    assert sample_rate is not None
    count = max(len(item) for item in reference_parts.values())
    reference_sum = sum(
        (_fit_length(item, count) for item in reference_parts.values()),
        np.zeros(count),
    )
    automatic_sum = sum(
        (_fit_length(item, count) for item in automatic_parts.values()),
        np.zeros(count),
    )
    oracle_sum = sum(
        (_fit_length(item, count) for item in oracle_parts.values()),
        np.zeros(count),
    )
    automatic_sum = _match_rms(reference_sum, automatic_sum)
    oracle_sum = _match_rms(reference_sum, oracle_sum)
    write_audio(config.output_dir / "01-original-drums-sum.wav", reference_sum, sample_rate)
    write_audio(config.output_dir / "02-automatic-drums.wav", automatic_sum, sample_rate)
    write_audio(config.output_dir / "03-oracle-nearest-hit.wav", oracle_sum, sample_rate)
    silence = np.zeros(round(0.45 * sample_rate), dtype=np.float64)
    write_audio(
        config.output_dir / "04-original-auto-oracle.wav",
        np.concatenate([reference_sum, silence, automatic_sum, silence, oracle_sum]),
        sample_rate,
    )
    (config.output_dir / "Drums-RoundRobin-Velocity.sfz").write_text(
        "\n".join(sfz_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "Drums-key-map.csv").write_text(
        "\n".join(key_map_lines) + "\n",
        encoding="utf-8",
    )
    write_notes(
        config.output_dir / "Drums-validation.mid",
        sorted(validation_midi, key=lambda item: (item.start, item.note)),
    )
    report: dict[str, object] = {
        "instrument": "Drums",
        "sample_directory": "Samples/Drums",
        "validation": {
            "start": config.validation_start,
            "end": config.validation_end,
            "tail": config.validation_tail,
        },
        "pieces": report_pieces,
        "automatic_metrics": compare_audio(reference_sum, automatic_sum, sample_rate),
        "oracle_metrics": compare_audio(reference_sum, oracle_sum, sample_rate),
        "oracle_warning": (
            "The nearest-hit oracle uses held-out audio features and is not MIDI-only playback."
        ),
    }
    (config.output_dir / "comparison-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
