from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample


@dataclass(frozen=True, slots=True)
class RenderNote:
    start: float
    duration: float
    note: int
    velocity: int = 64
    gain: float = 1.0


@dataclass(frozen=True, slots=True)
class SampleInstrument:
    samples: np.ndarray
    sample_rate: int
    root_note: int
    release: float = 0.12
    loop_start: int | None = None
    loop_end: int | None = None
    loop_crossfade: int = 0

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if len(self.samples) == 0:
            raise ValueError("samples must not be empty")
        if not 0 <= self.root_note <= 127:
            raise ValueError("root_note must be between 0 and 127")
        if self.release < 0:
            raise ValueError("release must be non-negative")
        if (self.loop_start is None) != (self.loop_end is None):
            raise ValueError("loop_start and loop_end must be specified together")
        if self.loop_start is not None and self.loop_end is not None:
            if not 0 <= self.loop_start < self.loop_end <= len(self.samples):
                raise ValueError("loop points must fall inside the sample")
            loop_length = self.loop_end - self.loop_start
            if not 0 <= self.loop_crossfade < loop_length:
                raise ValueError("loop_crossfade must be shorter than the loop")


@dataclass(frozen=True, slots=True)
class SampleRegion:
    instrument: SampleInstrument
    low_key: int = 0
    high_key: int = 127
    low_velocity: int = 1
    high_velocity: int = 127

    def validate(self) -> None:
        self.instrument.validate()
        if not 0 <= self.low_key <= self.high_key <= 127:
            raise ValueError("region key range must be between 0 and 127")
        if not 1 <= self.low_velocity <= self.high_velocity <= 127:
            raise ValueError("region velocity range must be between 1 and 127")

    def matches(self, note: RenderNote) -> bool:
        return (
            self.low_key <= note.note <= self.high_key
            and self.low_velocity <= note.velocity <= self.high_velocity
        )


def select_region(regions: list[SampleRegion], note: RenderNote) -> SampleRegion:
    if not regions:
        raise ValueError("at least one sample region is required")
    for region in regions:
        region.validate()
    matches = [region for region in regions if region.matches(note)]
    if not matches:
        raise ValueError(f"no sample region covers MIDI note {note.note} velocity {note.velocity}")
    return min(matches, key=lambda region: abs(region.instrument.root_note - note.note))


def _pitch_shift_resample(samples: np.ndarray, semitones: int) -> tuple[np.ndarray, float]:
    duration_factor = 2.0 ** (-semitones / 12.0)
    new_length = max(1, round(len(samples) * duration_factor))
    return np.asarray(resample(samples, new_length), dtype=np.float64), duration_factor


def _append_looped(
    pitched: np.ndarray,
    required_length: int,
    loop_start: int,
    loop_end: int,
    crossfade: int,
) -> np.ndarray:
    if required_length <= len(pitched):
        return pitched[:required_length].copy()
    prefix = pitched[:loop_end].copy()
    loop = pitched[loop_start:loop_end].copy()
    if len(loop) == 0:
        return np.pad(prefix, (0, max(0, required_length - len(prefix))))[:required_length]

    output = prefix
    while len(output) < required_length:
        if crossfade <= 0 or len(output) < crossfade or len(loop) <= crossfade:
            output = np.concatenate([output, loop])
            continue
        reference = output[-crossfade:]
        incoming = loop[:crossfade]
        reference_rms = np.sqrt(np.mean(reference * reference) + 1e-15)
        incoming_rms = np.sqrt(np.mean(incoming * incoming) + 1e-15)
        gain = float(np.clip(reference_rms / incoming_rms, 0.5, 1.25))
        scaled_loop = loop * gain
        fade_out = np.cos(np.linspace(0.0, np.pi / 2.0, crossfade, endpoint=False))
        fade_in = np.sin(np.linspace(0.0, np.pi / 2.0, crossfade, endpoint=False))
        blend = reference * fade_out + scaled_loop[:crossfade] * fade_in
        output = np.concatenate([output[:-crossfade], blend, scaled_loop[crossfade:]])
        loop = scaled_loop
    return output[:required_length]


def render_voice(instrument: SampleInstrument, note: RenderNote) -> np.ndarray:
    instrument.validate()
    if note.duration <= 0:
        raise ValueError("note duration must be positive")
    if not 0 <= note.velocity <= 127:
        raise ValueError("velocity must be between 0 and 127")

    semitones = note.note - instrument.root_note
    pitched, duration_factor = _pitch_shift_resample(instrument.samples, semitones)
    sustain_samples = round(note.duration * instrument.sample_rate)
    release_samples = round(instrument.release * instrument.sample_rate)
    required = max(1, sustain_samples + release_samples)

    if instrument.loop_start is not None and instrument.loop_end is not None:
        loop_start = round(instrument.loop_start * duration_factor)
        loop_end = round(instrument.loop_end * duration_factor)
        crossfade = round(instrument.loop_crossfade * duration_factor)
        voice = _append_looped(pitched, required, loop_start, loop_end, crossfade)
    else:
        voice = np.pad(pitched, (0, max(0, required - len(pitched))))[:required]

    if release_samples:
        release_start = min(sustain_samples, len(voice))
        release_end = min(len(voice), release_start + release_samples)
        count = release_end - release_start
        if count:
            # Cosine release starts at the current value and reaches zero smoothly.
            envelope = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, count)))
            voice[release_start:release_end] *= envelope
        if release_end < len(voice):
            voice[release_end:] = 0.0

    velocity_gain = note.velocity / 64.0 if note.velocity else 0.0
    return voice * note.gain * velocity_gain


def render_sequence(
    instrument: SampleInstrument,
    notes: list[RenderNote],
    *,
    tail: float = 0.0,
) -> np.ndarray:
    if not notes:
        return np.zeros(0, dtype=np.float64)
    if any(note.start < 0 for note in notes):
        raise ValueError("note start times must be non-negative")
    end_time = max(note.start + note.duration + instrument.release for note in notes) + tail
    output = np.zeros(max(1, round(end_time * instrument.sample_rate)), dtype=np.float64)
    for note in notes:
        voice = render_voice(instrument, note)
        first = round(note.start * instrument.sample_rate)
        last = min(len(output), first + len(voice))
        output[first:last] += voice[: last - first]
    return output


def render_multisample_sequence(
    regions: list[SampleRegion],
    notes: list[RenderNote],
    *,
    tail: float = 0.0,
) -> np.ndarray:
    if not notes:
        return np.zeros(0, dtype=np.float64)
    if any(note.start < 0 for note in notes):
        raise ValueError("note start times must be non-negative")
    selected = [(note, select_region(regions, note)) for note in notes]
    sample_rates = {region.instrument.sample_rate for _, region in selected}
    if len(sample_rates) != 1:
        raise ValueError("all selected sample regions must use the same sample rate")
    sample_rate = sample_rates.pop()
    end_time = max(
        note.start + note.duration + region.instrument.release for note, region in selected
    ) + tail
    output = np.zeros(max(1, round(end_time * sample_rate)), dtype=np.float64)
    for note, region in selected:
        voice = render_voice(region.instrument, note)
        first = round(note.start * sample_rate)
        last = min(len(output), first + len(voice))
        output[first:last] += voice[: last - first]
    return output
