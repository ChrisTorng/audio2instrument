from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample

from audio2instrument.audio import AudioData, apply_edge_fades, detect_onset_near, rms, slice_audio
from audio2instrument.loop import LoopPoints, find_loop_points
from audio2instrument.midi import NoteEvent
from audio2instrument.render import RenderNote, SampleInstrument, render_voice


@dataclass(frozen=True, slots=True)
class SampleVariant:
    samples: np.ndarray
    sample_rate: int
    root_note: int
    source_note: NoteEvent
    detected_onset: float
    source_level: float
    loop: LoopPoints
    name: str


@dataclass(frozen=True, slots=True)
class RootLayer:
    root_note: int
    variants: tuple[SampleVariant, ...]

    def validate(self) -> None:
        if not self.variants:
            raise ValueError("a root layer needs at least one variant")
        if any(item.root_note != self.root_note for item in self.variants):
            raise ValueError("all variants must use the layer root note")
        if len({item.sample_rate for item in self.variants}) != 1:
            raise ValueError("all variants in one layer must use the same sample rate")


@dataclass(frozen=True, slots=True)
class RenderEvent:
    start: float
    duration: float
    note: int
    velocity: int = 64


def reject_ambiguous_same_pitch_retrigger(
    ordered_notes: list[NoteEvent], index: int, *, tolerance: float = 0.08
) -> bool:
    if index <= 0:
        return False
    previous = ordered_notes[index - 1]
    current = ordered_notes[index]
    return previous.note == current.note and abs(previous.end - current.start) <= tolerance


def select_round_robin_candidates(
    notes: list[NoteEvent],
    root_note: int,
    *,
    minimum_duration: float = 1.25,
    maximum_variants: int = 3,
    minimum_time_separation: float = 8.0,
) -> list[NoteEvent]:
    if maximum_variants <= 0:
        raise ValueError("maximum_variants must be positive")
    ordered = sorted(notes, key=lambda item: (item.start, item.note))
    ranked: list[tuple[float, NoteEvent]] = []
    for index, note in enumerate(ordered):
        if note.note != root_note or note.duration < minimum_duration:
            continue
        if reject_ambiguous_same_pitch_retrigger(ordered, index):
            continue
        previous = ordered[index - 1] if index else None
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        score = note.duration
        score += 0.25 if previous is None or previous.note != root_note else 0.0
        score += 0.10 if following is None or following.note != root_note else 0.0
        ranked.append((score, note))
    ranked.sort(key=lambda item: (-item[0], item[1].start))
    selected: list[NoteEvent] = []
    for _, note in ranked:
        if any(abs(note.start - other.start) < minimum_time_separation for other in selected):
            continue
        selected.append(note)
        if len(selected) == maximum_variants:
            break
    return sorted(selected, key=lambda item: item.start)


def onset_level(
    samples: np.ndarray,
    sample_rate: int,
    *,
    offset: float = 0.03,
    duration: float = 0.18,
) -> float:
    first = max(0, round(offset * sample_rate))
    last = min(len(samples), round((offset + duration) * sample_rate))
    return rms(samples[first:last])


def build_root_layers(
    audio: AudioData,
    notes: list[NoteEvent],
    root_notes: list[int] | tuple[int, ...],
    *,
    sample_duration: float = 1.35,
    maximum_variants: int = 3,
    minimum_duration: float = 1.25,
) -> tuple[list[RootLayer], float]:
    if sample_duration <= 0:
        raise ValueError("sample_duration must be positive")
    raw: list[tuple[int, NoteEvent, float, np.ndarray, float, LoopPoints]] = []
    for root in sorted(set(root_notes)):
        candidates = select_round_robin_candidates(
            notes,
            root,
            minimum_duration=max(minimum_duration, sample_duration + 0.02),
            maximum_variants=maximum_variants,
        )
        for note in candidates:
            onset = detect_onset_near(audio, note.start)
            sample = slice_audio(audio, onset, min(audio.duration, onset + sample_duration))
            expected = round(sample_duration * audio.sample_rate)
            if len(sample) < expected:
                sample = np.pad(sample, (0, expected - len(sample)))
            sample = apply_edge_fades(sample, audio.sample_rate, fade_in=0.003, fade_out=0.012)
            level = onset_level(sample, audio.sample_rate)
            loop = find_loop_points(
                sample,
                audio.sample_rate,
                search_start=0.78,
                search_end=min(sample_duration - 0.06, 1.30),
                min_duration=0.24,
                max_duration=0.58,
                crossfade=0.03,
            )
            raw.append((root, note, onset, sample, level, loop))
    if not raw:
        raise ValueError("no usable sample candidates found")
    positive = [item[4] for item in raw if item[4] > 0]
    target_level = float(np.median(positive)) if positive else 1.0
    layers: list[RootLayer] = []
    for root in sorted(set(item[0] for item in raw)):
        variants = []
        for index, (_, note, onset, sample, level, loop) in enumerate(
            [item for item in raw if item[0] == root], start=1
        ):
            variants.append(
                SampleVariant(
                    samples=sample * (target_level / max(level, 1e-12)),
                    sample_rate=audio.sample_rate,
                    root_note=root,
                    source_note=note,
                    detected_onset=onset,
                    source_level=level,
                    loop=loop,
                    name=f"Bass_{root}_rr{index}.wav",
                )
            )
        layer = RootLayer(root, tuple(variants))
        layer.validate()
        layers.append(layer)
    return layers, target_level


def equal_power_root_weights(
    root_notes: list[int] | tuple[int, ...], target_note: int
) -> dict[int, float]:
    roots = sorted(set(root_notes))
    if not roots:
        raise ValueError("at least one root note is required")
    if target_note <= roots[0]:
        return {roots[0]: 1.0}
    if target_note >= roots[-1]:
        return {roots[-1]: 1.0}
    for left, right in zip(roots, roots[1:]):
        if target_note == left:
            return {left: 1.0}
        if target_note == right:
            return {right: 1.0}
        if left < target_note < right:
            position = (target_note - left) / (right - left)
            return {
                left: float(np.cos(position * np.pi / 2.0)),
                right: float(np.sin(position * np.pi / 2.0)),
            }
    raise RuntimeError("unreachable root interpolation state")


def render_variant(variant: SampleVariant, event: RenderEvent, *, release: float) -> np.ndarray:
    instrument = SampleInstrument(
        samples=variant.samples,
        sample_rate=variant.sample_rate,
        root_note=variant.root_note,
        release=release,
        loop_start=variant.loop.start,
        loop_end=variant.loop.end,
        loop_crossfade=variant.loop.crossfade,
    )
    return render_voice(
        instrument,
        RenderNote(start=0.0, duration=event.duration, note=event.note, velocity=event.velocity),
    )


def _render_with_weights(
    layers: list[RootLayer],
    events: list[RenderEvent],
    weights_for,
    *,
    release: float,
    round_robin: bool,
) -> np.ndarray:
    if not layers or not events:
        return np.zeros(0, dtype=np.float64)
    rates = {variant.sample_rate for layer in layers for variant in layer.variants}
    if len(rates) != 1:
        raise ValueError("all samples must use one sample rate")
    sample_rate = rates.pop()
    by_root = {layer.root_note: layer for layer in layers}
    roots = sorted(by_root)
    output = np.zeros(
        max(1, round(max(event.start + event.duration + release for event in events) * sample_rate)),
        dtype=np.float64,
    )
    counters = {root: 0 for root in roots}
    for event in events:
        for root, weight in weights_for(roots, event.note).items():
            layer = by_root[root]
            position = counters[root] % len(layer.variants) if round_robin else 0
            counters[root] += 1
            voice = render_variant(layer.variants[position], event, release=release) * weight
            first = round(event.start * sample_rate)
            last = min(len(output), first + len(voice))
            output[first:last] += voice[: last - first]
    return output


def render_expressive_sequence(
    layers: list[RootLayer],
    events: list[RenderEvent],
    *,
    release: float = 0.08,
    crossfade_roots: bool = True,
    round_robin: bool = True,
) -> np.ndarray:
    roots = sorted(layer.root_note for layer in layers)

    def choose(_: list[int], note: int) -> dict[int, float]:
        if crossfade_roots:
            return equal_power_root_weights(roots, note)
        nearest = min(roots, key=lambda root: (abs(root - note), root))
        return {nearest: 1.0}

    return _render_with_weights(
        layers, events, choose, release=release, round_robin=round_robin
    )


def infer_velocities(levels: list[float], *, reference_velocity: int = 64) -> list[int]:
    if not levels:
        return []
    positive = [value for value in levels if value > 0]
    reference = float(np.median(positive)) if positive else 1.0
    return [int(np.clip(round(reference_velocity * value / reference), 1, 127)) for value in levels]


def _attack_feature(samples: np.ndarray, sample_rate: int, duration: float) -> np.ndarray:
    first = min(len(samples), round(0.02 * sample_rate))
    last = min(len(samples), max(first + 1, round(duration * sample_rate)))
    segment = samples[first:last]
    if len(segment) < 64:
        segment = np.pad(segment, (0, 64 - len(segment)))
    feature = np.log1p(np.abs(np.fft.rfft(segment * np.hanning(len(segment)))))
    return feature / max(np.linalg.norm(feature), 1e-12)


def hybridize_round_robin_layers(
    layers: list[RootLayer],
    *,
    attack_duration: float = 0.09,
    transition_duration: float = 0.05,
    maximum_attack_spectral_distance: float = 0.25,
) -> list[RootLayer]:
    if attack_duration < 0 or transition_duration <= 0:
        raise ValueError("attack and transition durations must be valid")
    if maximum_attack_spectral_distance <= 0:
        raise ValueError("maximum_attack_spectral_distance must be positive")
    result = []
    for layer in layers:
        layer.validate()
        canonical = layer.variants[0]
        rate = canonical.sample_rate
        attack_count = round(attack_duration * rate)
        transition_count = round(transition_duration * rate)
        canonical_feature = _attack_feature(canonical.samples, rate, attack_duration)
        variants = [canonical]
        for alternate in layer.variants[1:]:
            length = min(len(canonical.samples), len(alternate.samples))
            hybrid = canonical.samples[:length].copy()
            distance = float(
                np.linalg.norm(
                    canonical_feature - _attack_feature(alternate.samples, rate, attack_duration)
                )
            )
            mix = min(1.0, maximum_attack_spectral_distance / max(distance, 1e-12))
            attack_end = min(length, attack_count)
            transition_end = min(length, attack_end + transition_count)
            mixed = canonical.samples[:attack_end] * (1.0 - mix) + alternate.samples[:attack_end] * mix
            mixed_rms = rms(mixed)
            if mixed_rms > 0:
                mixed *= rms(canonical.samples[:attack_end]) / mixed_rms
            hybrid[:attack_end] = mixed
            count = transition_end - attack_end
            if count:
                fade_alt = np.cos(np.linspace(0.0, np.pi / 2.0, count, endpoint=True))
                fade_canonical = np.sin(np.linspace(0.0, np.pi / 2.0, count, endpoint=True))
                blended = (
                    canonical.samples[attack_end:transition_end] * (1.0 - mix)
                    + alternate.samples[attack_end:transition_end] * mix
                )
                hybrid[attack_end:transition_end] = (
                    blended * fade_alt
                    + canonical.samples[attack_end:transition_end] * fade_canonical
                )
            variants.append(
                SampleVariant(
                    samples=hybrid,
                    sample_rate=rate,
                    root_note=layer.root_note,
                    source_note=alternate.source_note,
                    detected_onset=alternate.detected_onset,
                    source_level=alternate.source_level,
                    loop=canonical.loop,
                    name=alternate.name,
                )
            )
        result.append(RootLayer(layer.root_note, tuple(variants)))
    return result


def _shift(samples: np.ndarray, semitones: int) -> np.ndarray:
    factor = 2.0 ** (-semitones / 12.0)
    return np.asarray(resample(samples, max(1, round(len(samples) * factor))), dtype=np.float64)


def _window_feature(samples: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    first = min(len(samples), max(0, round(start * rate)))
    last = min(len(samples), max(first + 1, round(end * rate)))
    segment = samples[first:last]
    if len(segment) < 64:
        segment = np.pad(segment, (0, 64 - len(segment)))
    feature = np.log1p(np.abs(np.fft.rfft(segment * np.hanning(len(segment)))))
    return feature / max(np.linalg.norm(feature), 1e-12)


def root_pair_similarity(left: RootLayer, right: RootLayer) -> float:
    left.validate()
    right.validate()
    target = round((left.root_note + right.root_note) / 2)
    left_audio = _shift(left.variants[0].samples, target - left.root_note)
    right_audio = _shift(right.variants[0].samples, target - right.root_note)
    count = min(len(left_audio), len(right_audio))
    rate = left.variants[0].sample_rate
    attack = np.dot(
        _window_feature(left_audio[:count], rate, 0.02, 0.25),
        _window_feature(right_audio[:count], rate, 0.02, 0.25),
    )
    body = np.dot(
        _window_feature(left_audio[:count], rate, 0.45, 0.85),
        _window_feature(right_audio[:count], rate, 0.45, 0.85),
    )
    return float(0.35 * attack + 0.65 * body)


def compatible_root_pairs(
    layers: list[RootLayer], *, threshold: float = 0.94
) -> dict[tuple[int, int], float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    ordered = sorted(layers, key=lambda layer: layer.root_note)
    pairs = {}
    for left, right in zip(ordered, ordered[1:]):
        if right.root_note - left.root_note < 2:
            continue
        similarity = root_pair_similarity(left, right)
        if similarity >= threshold:
            pairs[(left.root_note, right.root_note)] = similarity
    return pairs


def adaptive_root_weights(
    root_notes: list[int] | tuple[int, ...],
    target_note: int,
    compatible_pairs: set[tuple[int, int]],
) -> dict[int, float]:
    roots = sorted(set(root_notes))
    if target_note <= roots[0]:
        return {roots[0]: 1.0}
    if target_note >= roots[-1]:
        return {roots[-1]: 1.0}
    for left, right in zip(roots, roots[1:]):
        if target_note == left:
            return {left: 1.0}
        if target_note == right:
            return {right: 1.0}
        if left < target_note < right:
            if (left, right) in compatible_pairs:
                return equal_power_root_weights(roots, target_note)
            nearest = min((left, right), key=lambda root: (abs(root - target_note), root))
            return {nearest: 1.0}
    raise RuntimeError("unreachable adaptive interpolation state")


def render_adaptive_sequence(
    layers: list[RootLayer],
    events: list[RenderEvent],
    *,
    compatible_pairs: set[tuple[int, int]],
    release: float = 0.08,
    round_robin: bool = True,
) -> np.ndarray:
    def choose(roots: list[int], note: int) -> dict[int, float]:
        return adaptive_root_weights(roots, note, compatible_pairs)

    return _render_with_weights(
        layers, events, choose, release=release, round_robin=round_robin
    )


def onset_spectral_diversity(
    samples: np.ndarray,
    sample_rate: int,
    onset_times: list[float],
    *,
    window: float = 0.12,
) -> float:
    spectra = []
    count = max(64, round(window * sample_rate))
    for onset in onset_times:
        first = max(0, round(onset * sample_rate))
        segment = samples[first : first + count]
        if len(segment) < count:
            segment = np.pad(segment, (0, count - len(segment)))
        feature = np.log1p(np.abs(np.fft.rfft(segment * np.hanning(count))))
        spectra.append(feature / max(np.linalg.norm(feature), 1e-12))
    if len(spectra) < 2:
        return 0.0
    return float(
        np.mean(
            [
                np.linalg.norm(spectra[left] - spectra[right])
                for left in range(len(spectra))
                for right in range(left + 1, len(spectra))
            ]
        )
    )


def render_sfz_round_robin_crossfade(
    layers: list[RootLayer],
    *,
    low_key: int,
    high_key: int,
    release: float = 0.08,
    sample_prefix: str = "Samples/",
) -> str:
    ordered = sorted(layers, key=lambda item: item.root_note)
    pairs = {
        (left.root_note, right.root_note)
        for left, right in zip(ordered, ordered[1:])
    }
    return render_sfz_adaptive(
        layers,
        low_key=low_key,
        high_key=high_key,
        compatible_pairs=pairs,
        release=release,
        sample_prefix=sample_prefix,
    )


def render_sfz_adaptive(
    layers: list[RootLayer],
    *,
    low_key: int,
    high_key: int,
    compatible_pairs: set[tuple[int, int]],
    release: float = 0.08,
    sample_prefix: str = "Samples/",
) -> str:
    roots = sorted(layer.root_note for layer in layers)
    by_root = {layer.root_note: layer for layer in layers}
    hard_ranges = {}
    next_low = low_key
    for index, root in enumerate(roots):
        high = high_key if index == len(roots) - 1 else (root + roots[index + 1]) // 2
        hard_ranges[root] = (next_low, high)
        next_low = high + 1
    lines = [
        "<control>",
        f"default_path={sample_prefix}",
        "",
        "<global>",
        "xf_keycurve=power",
        f"ampeg_release={release:.6f}",
        "",
    ]
    for index, root in enumerate(roots):
        previous = roots[index - 1] if index else None
        following = roots[index + 1] if index + 1 < len(roots) else None
        lokey, hikey = hard_ranges[root]
        fade_in = previous is not None and (previous, root) in compatible_pairs
        fade_out = following is not None and (root, following) in compatible_pairs
        if fade_in:
            lokey = previous
        if fade_out:
            hikey = following
        for position, variant in enumerate(by_root[root].variants, start=1):
            opcodes = [
                "<region>",
                f"sample={variant.name}",
                f"pitch_keycenter={root}",
                f"lokey={lokey}",
                f"hikey={hikey}",
                "loop_mode=loop_sustain",
                f"loop_start={variant.loop.start}",
                f"loop_end={variant.loop.end}",
            ]
            if fade_in:
                opcodes += [f"xfin_lokey={previous}", f"xfin_hikey={root}"]
            if fade_out:
                opcodes += [f"xfout_lokey={root}", f"xfout_hikey={following}"]
            if len(by_root[root].variants) > 1:
                opcodes += [
                    f"seq_length={len(by_root[root].variants)}",
                    f"seq_position={position}",
                ]
            lines.append(" ".join(opcodes))
    return "\n".join(lines) + "\n"


def write_samples(directory: str | Path, layers: list[RootLayer]) -> None:
    import soundfile as sf

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        for variant in layer.variants:
            sf.write(
                target / variant.name,
                np.asarray(variant.samples, dtype=np.float32),
                variant.sample_rate,
                subtype="PCM_24",
            )
