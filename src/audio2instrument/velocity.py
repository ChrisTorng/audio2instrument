from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from audio2instrument.audio import AudioData, detect_onset_near, rms, slice_audio, write_audio
from audio2instrument.expressive import (
    RenderEvent,
    RootLayer,
    SampleVariant,
    reject_ambiguous_same_pitch_retrigger,
    render_variant,
)
from audio2instrument.midi import NoteEvent


@dataclass(frozen=True, slots=True)
class TimbreFeatures:
    level: float
    attack_slope_db_per_second: float
    spectral_centroid_hz: float
    high_frequency_ratio: float


@dataclass(frozen=True, slots=True)
class AttackCandidate:
    note: NoteEvent
    detected_onset: float
    samples: np.ndarray
    features: TimbreFeatures
    intensity_score: float = 0.0


@dataclass(frozen=True, slots=True)
class VelocityLayer:
    low_velocity: int
    high_velocity: int
    center_velocity: int
    variants: tuple[SampleVariant, ...]
    source_scores: tuple[float, ...]

    def validate(self) -> None:
        if not 1 <= self.low_velocity <= self.high_velocity <= 127:
            raise ValueError("velocity layer range must be between 1 and 127")
        if not self.low_velocity <= self.center_velocity <= self.high_velocity:
            raise ValueError("center velocity must fall inside the layer")
        if not self.variants:
            raise ValueError("velocity layer needs at least one variant")


@dataclass(frozen=True, slots=True)
class VelocityRootLayer:
    root_note: int
    layers: tuple[VelocityLayer, ...]

    def validate(self) -> None:
        if not self.layers:
            raise ValueError("root needs at least one velocity layer")
        previous = 0
        for layer in self.layers:
            layer.validate()
            if layer.low_velocity != previous + 1:
                raise ValueError("velocity layers must be contiguous")
            if any(item.root_note != self.root_note for item in layer.variants):
                raise ValueError("all variants must use the root note")
            previous = layer.high_velocity
        if previous != 127:
            raise ValueError("velocity layers must cover through velocity 127")


def extract_timbre_features(samples: np.ndarray, sample_rate: int) -> TimbreFeatures:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    count = max(64, round(0.20 * sample_rate))
    values = np.asarray(samples[:count], dtype=np.float64)
    if len(values) < count:
        values = np.pad(values, (0, count - len(values)))
    values -= np.mean(values)
    level = rms(values[round(0.025 * sample_rate) : round(0.18 * sample_rate)])

    frame = max(16, round(0.010 * sample_rate))
    hop = max(1, round(0.005 * sample_rate))
    last = max(frame, round(0.080 * sample_rate))
    envelope = [rms(values[index : index + frame]) for index in range(0, last - frame + 1, hop)]
    db = 20.0 * np.log10(np.asarray(envelope) + 1e-8)
    times = np.arange(len(db), dtype=np.float64) * hop / sample_rate
    slope = float(np.polyfit(times, db, 1)[0]) if len(db) > 1 else 0.0

    first = round(0.020 * sample_rate)
    end = round(0.180 * sample_rate)
    segment = values[first:end]
    segment *= np.hanning(len(segment))
    power = np.abs(np.fft.rfft(segment)) ** 2 + 1e-15
    frequencies = np.fft.rfftfreq(len(segment), 1.0 / sample_rate)
    total = float(np.sum(power))
    centroid = float(np.sum(frequencies * power) / total)
    high_frequency_ratio = float(np.sum(power[frequencies >= 700.0]) / total)
    return TimbreFeatures(level, slope, centroid, high_frequency_ratio)


def valid_bass_attack(features: TimbreFeatures) -> bool:
    return (
        features.level >= 0.02
        and features.spectral_centroid_hz <= 1000.0
        and features.high_frequency_ratio <= 0.20
    )


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    lower, upper = np.percentile(values, [25, 75], axis=0)
    scale = upper - lower
    scale = np.where(scale < 1e-6, 1.0, scale)
    return np.clip((values - median) / scale, -2.5, 2.5)


def score_attack_candidates(candidates: list[AttackCandidate]) -> list[AttackCandidate]:
    if not candidates:
        return []
    matrix = np.asarray(
        [
            [
                np.log(max(item.features.level, 1e-8)),
                np.log(max(item.features.spectral_centroid_hz, 1.0)),
                np.log(item.features.high_frequency_ratio + 1e-4),
                item.features.attack_slope_db_per_second / 100.0,
            ]
            for item in candidates
        ],
        dtype=np.float64,
    )
    z = _robust_z(matrix)
    scores = 0.55 * z[:, 0] + 0.25 * z[:, 1] + 0.15 * z[:, 2] + 0.05 * z[:, 3]
    return [
        AttackCandidate(
            note=item.note,
            detected_onset=item.detected_onset,
            samples=item.samples,
            features=item.features,
            intensity_score=float(score),
        )
        for item, score in zip(candidates, scores, strict=True)
    ]


def collect_attack_candidates(
    audio: AudioData,
    notes: list[NoteEvent],
    root_note: int,
    *,
    excluded_ranges: tuple[tuple[float, float], ...] = (),
    attack_duration: float = 0.22,
    minimum_note_duration: float = 0.14,
) -> list[AttackCandidate]:
    ordered = sorted(notes, key=lambda item: (item.start, item.note))
    result: list[AttackCandidate] = []
    for index, note in enumerate(ordered):
        if note.note != root_note or note.duration < minimum_note_duration:
            continue
        if any(start <= note.start < end for start, end in excluded_ranges):
            continue
        if reject_ambiguous_same_pitch_retrigger(ordered, index):
            continue
        onset = detect_onset_near(
            audio,
            note.start,
            search_before=0.05,
            search_after=0.08,
            window=0.012,
            hop=0.0015,
        )
        samples = slice_audio(audio, onset, min(audio.duration, onset + attack_duration))
        expected = round(attack_duration * audio.sample_rate)
        if len(samples) < expected:
            samples = np.pad(samples, (0, expected - len(samples)))
        features = extract_timbre_features(samples, audio.sample_rate)
        if valid_bass_attack(features):
            result.append(AttackCandidate(note, onset, samples, features))
    return score_attack_candidates(result)


def velocity_layout(layer_count: int) -> tuple[tuple[int, int, int], ...]:
    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    if layer_count == 1:
        return ((1, 127, 64),)
    if layer_count == 2:
        return ((1, 72, 44), (73, 127, 104))
    return ((1, 52, 32), (53, 96, 76), (97, 127, 116))


def _select_near_quantile(
    candidates: list[AttackCandidate],
    quantile: float,
    *,
    maximum_variants: int,
    minimum_time_separation: float,
) -> list[AttackCandidate]:
    scores = np.asarray([item.intensity_score for item in candidates], dtype=np.float64)
    target = float(np.quantile(scores, quantile))
    ranked = sorted(
        candidates,
        key=lambda item: (abs(item.intensity_score - target), -item.note.duration),
    )
    selected: list[AttackCandidate] = []
    for item in ranked:
        if any(abs(item.note.start - other.note.start) < minimum_time_separation for other in selected):
            continue
        selected.append(item)
        if len(selected) == maximum_variants:
            break
    return selected


def _normalized_attack_feature(samples: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
    count = max(64, min(len(samples), round(duration * sample_rate)))
    segment = np.asarray(samples[:count], dtype=np.float64)
    if len(segment) < count:
        segment = np.pad(segment, (0, count - len(segment)))
    feature = np.log1p(np.abs(np.fft.rfft(segment * np.hanning(len(segment)))))
    return feature / max(np.linalg.norm(feature), 1e-12)


def _attack_to_canonical(
    canonical: SampleVariant,
    candidate: AttackCandidate,
    *,
    attack_duration: float,
    transition_duration: float,
    name: str,
    maximum_attack_spectral_distance: float,
) -> SampleVariant:
    rate = canonical.sample_rate
    length = len(canonical.samples)
    alternate = np.asarray(candidate.samples, dtype=np.float64)
    if len(alternate) < length:
        alternate = np.pad(alternate, (0, length - len(alternate)))
    else:
        alternate = alternate[:length]
    attack_end = min(length, round(attack_duration * rate))
    transition_end = min(length, attack_end + round(transition_duration * rate))
    hybrid = canonical.samples.copy()
    alt_attack = alternate[:attack_end].copy()
    alt_level = rms(alt_attack)
    canonical_level = rms(canonical.samples[:attack_end])
    if alt_level > 0:
        alt_attack *= canonical_level / alt_level
    canonical_feature = _normalized_attack_feature(canonical.samples, rate, attack_duration)
    alternate_feature = _normalized_attack_feature(alt_attack, rate, attack_duration)
    distance = float(np.linalg.norm(canonical_feature - alternate_feature))
    blend = min(1.0, maximum_attack_spectral_distance / max(distance, 1e-12))
    hybrid[:attack_end] = canonical.samples[:attack_end] * (1.0 - blend) + alt_attack * blend
    count = transition_end - attack_end
    if count:
        fade_alt = np.cos(np.linspace(0.0, np.pi / 2.0, count, endpoint=True))
        fade_body = np.sin(np.linspace(0.0, np.pi / 2.0, count, endpoint=True))
        transition = alternate[attack_end:transition_end].copy()
        transition_level = rms(transition)
        body_level = rms(canonical.samples[attack_end:transition_end])
        if transition_level > 0:
            transition *= body_level / transition_level
        blended_transition = (
            canonical.samples[attack_end:transition_end] * (1.0 - blend) + transition * blend
        )
        hybrid[attack_end:transition_end] = blended_transition * fade_alt + canonical.samples[
            attack_end:transition_end
        ] * fade_body
    return SampleVariant(
        samples=hybrid,
        sample_rate=rate,
        root_note=canonical.root_note,
        source_note=candidate.note,
        detected_onset=candidate.detected_onset,
        source_level=candidate.features.level,
        loop=canonical.loop,
        name=name,
    )


def build_velocity_root_layers(
    audio: AudioData,
    notes: list[NoteEvent],
    canonical_layers: list[RootLayer],
    *,
    excluded_ranges: tuple[tuple[float, float], ...] = (),
    maximum_velocity_layers: int = 3,
    maximum_round_robins: int = 2,
    attack_duration: float = 0.14,
    transition_duration: float = 0.05,
    maximum_attack_spectral_distance: float = 0.20,
) -> list[VelocityRootLayer]:
    result: list[VelocityRootLayer] = []
    quantiles_by_count = {1: (0.5,), 2: (0.25, 0.75), 3: (0.12, 0.5, 0.88)}
    for root in canonical_layers:
        root.validate()
        canonical = root.variants[0]
        candidates = collect_attack_candidates(
            audio,
            notes,
            root.root_note,
            excluded_ranges=excluded_ranges,
        )
        if len(candidates) >= 8 and maximum_velocity_layers >= 3:
            count = 3
        elif len(candidates) >= 3 and maximum_velocity_layers >= 2:
            count = 2
        else:
            count = 1
        layout = velocity_layout(count)
        velocity_layers: list[VelocityLayer] = []
        for layer_index, ((low, high, center), quantile) in enumerate(
            zip(layout, quantiles_by_count[count], strict=True), start=1
        ):
            selected = _select_near_quantile(
                candidates,
                quantile,
                maximum_variants=maximum_round_robins,
                minimum_time_separation=4.0,
            )
            if not selected:
                selected = [
                    AttackCandidate(
                        note=canonical.source_note,
                        detected_onset=canonical.detected_onset,
                        samples=canonical.samples[: round(0.22 * canonical.sample_rate)],
                        features=extract_timbre_features(canonical.samples, canonical.sample_rate),
                        intensity_score=0.0,
                    )
                ]
            variants = tuple(
                _attack_to_canonical(
                    canonical,
                    item,
                    attack_duration=attack_duration,
                    transition_duration=transition_duration,
                    name=f"Bass_{root.root_note}_vel{layer_index}_rr{variant_index}.wav",
                    maximum_attack_spectral_distance=maximum_attack_spectral_distance,
                )
                for variant_index, item in enumerate(selected, start=1)
            )
            velocity_layers.append(
                VelocityLayer(
                    low_velocity=low,
                    high_velocity=high,
                    center_velocity=center,
                    variants=variants,
                    source_scores=tuple(item.intensity_score for item in selected),
                )
            )
        output = VelocityRootLayer(root.root_note, tuple(velocity_layers))
        output.validate()
        result.append(output)
    return result


def choose_velocity_layer(root: VelocityRootLayer, velocity: int) -> VelocityLayer:
    root.validate()
    for layer in root.layers:
        if layer.low_velocity <= velocity <= layer.high_velocity:
            return layer
    raise ValueError(f"no velocity layer covers {velocity}")


def infer_velocity_from_candidates(features: TimbreFeatures, candidates: list[AttackCandidate]) -> int:
    if not candidates:
        return 64
    scored = score_attack_candidates(
        candidates
        + [
            AttackCandidate(
                note=NoteEvent(0.0, 1.0, 0, 64, 0),
                detected_onset=0.0,
                samples=np.zeros(1, dtype=np.float64),
                features=features,
            )
        ]
    )
    target = scored[-1].intensity_score
    reference = np.asarray([item.intensity_score for item in scored[:-1]], dtype=np.float64)
    percentile = float(np.mean(reference <= target))
    return int(np.clip(round(16.0 + 96.0 * percentile), 16, 112))


def render_velocity_sequence(
    roots: list[VelocityRootLayer],
    events: list[RenderEvent],
    *,
    release: float = 0.08,
    round_robin: bool = True,
) -> np.ndarray:
    if not roots or not events:
        return np.zeros(0, dtype=np.float64)
    by_root = {item.root_note: item for item in roots}
    root_notes = sorted(by_root)
    rates = {
        variant.sample_rate
        for root in roots
        for layer in root.layers
        for variant in layer.variants
    }
    if len(rates) != 1:
        raise ValueError("all samples must use one sample rate")
    sample_rate = rates.pop()
    output = np.zeros(
        max(1, round(max(item.start + item.duration + release for item in events) * sample_rate)),
        dtype=np.float64,
    )
    counters: dict[tuple[int, int], int] = {}
    for event in events:
        root_note = min(root_notes, key=lambda value: (abs(value - event.note), value))
        root = by_root[root_note]
        layer = choose_velocity_layer(root, event.velocity)
        key = (root_note, layer.center_velocity)
        position = counters.get(key, 0) % len(layer.variants) if round_robin else 0
        counters[key] = counters.get(key, 0) + 1
        voice = render_variant(layer.variants[position], event, release=release)
        first = round(event.start * sample_rate)
        last = min(len(output), first + len(voice))
        output[first:last] += voice[: last - first]
    return output


def render_velocity_sfz(
    roots: list[VelocityRootLayer],
    *,
    low_key: int,
    high_key: int,
    release: float = 0.08,
    sample_prefix: str = "Samples/",
) -> str:
    ordered = sorted(roots, key=lambda item: item.root_note)
    ranges: dict[int, tuple[int, int]] = {}
    next_low = low_key
    for index, root in enumerate(ordered):
        high = high_key if index == len(ordered) - 1 else (
            root.root_note + ordered[index + 1].root_note
        ) // 2
        ranges[root.root_note] = (next_low, high)
        next_low = high + 1
    lines = [
        "<control>",
        f"default_path={sample_prefix}",
        "",
        "<global>",
        f"ampeg_release={release:.4f}",
        "amp_veltrack=100",
        "",
    ]
    for root in ordered:
        low, high = ranges[root.root_note]
        for layer in root.layers:
            for position, variant in enumerate(layer.variants, start=1):
                lines.extend(
                    [
                        "<region>",
                        f"sample={variant.name}",
                        f"pitch_keycenter={root.root_note}",
                        f"lokey={low}",
                        f"hikey={high}",
                        f"lovel={layer.low_velocity}",
                        f"hivel={layer.high_velocity}",
                        f"seq_length={len(layer.variants)}",
                        f"seq_position={position}",
                        "loop_mode=loop_sustain",
                        f"loop_start={variant.loop.start}",
                        f"loop_end={variant.loop.end}",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def write_velocity_samples(output_dir: str | Path, roots: list[VelocityRootLayer]) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for root in roots:
        for layer in root.layers:
            for variant in layer.variants:
                write_audio(target / variant.name, variant.samples, variant.sample_rate)
