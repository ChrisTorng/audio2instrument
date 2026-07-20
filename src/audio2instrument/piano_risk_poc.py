from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audio2instrument.audio import AudioData, detect_onset_near, rms
from audio2instrument.metrics import compare_audio
from audio2instrument.midi import NoteEvent, read_notes
from audio2instrument.piano_risk import (
    fit_note_spectral_gains,
    harmonic_mask_sample,
    log_spectral_distance,
    overlap_count,
    render_exact_key_sfz,
    render_one_shot_bank,
    score_aligned_onset,
    select_isolated_source,
)


@dataclass(frozen=True, slots=True)
class PianoRiskPocConfig:
    audio_path: Path
    midi_path: Path
    soundfont_path: Path
    output_dir: Path
    calibration_midi_onset: float = 29.122
    heldout_midi_onset: float = 29.352
    next_midi_onset: float = 30.222
    pitches: tuple[int, ...] = (49, 71, 76, 80)
    source_duration: float = 1.25


def _write_audio(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(samples, dtype=np.float32), sample_rate, subtype="PCM_24")


def _gain_match(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return estimate * (rms(reference) / max(rms(estimate), 1e-12))


def _render_fluidsynth(
    events: list[NoteEvent],
    sample_rate: int,
    *,
    origin: float,
    duration: float,
    soundfont_path: Path,
) -> np.ndarray:
    try:
        import fluidsynth
    except ImportError as error:
        raise RuntimeError(
            "The piano risk baseline requires pyfluidsynth and the system FluidSynth library"
        ) from error
    synth = fluidsynth.Synth(samplerate=float(sample_rate))
    soundfont_id = synth.sfload(str(soundfont_path))
    if soundfont_id < 0:
        synth.delete()
        raise RuntimeError(f"could not load SoundFont: {soundfont_path}")
    synth.program_select(0, soundfont_id, 0, 0)
    timeline: list[tuple[float, int, int, int]] = []
    for event in events:
        timeline.append((event.start - origin, 1, event.note, event.velocity))
        timeline.append((event.end - origin, 0, event.note, 0))
    timeline.sort(key=lambda item: (item[0], item[1]))

    chunks: list[np.ndarray] = []
    cursor = 0
    for event_time, is_on, pitch, velocity in timeline:
        target = max(cursor, round(max(0.0, event_time) * sample_rate))
        if target > cursor:
            block = np.asarray(synth.get_samples(target - cursor), dtype=np.float64)
            chunks.append(block.reshape(-1, 2) / 32768.0)
            cursor = target
        if is_on:
            synth.noteon(0, pitch, velocity)
        else:
            synth.noteoff(0, pitch)
    total = round(duration * sample_rate)
    if total > cursor:
        block = np.asarray(synth.get_samples(total - cursor), dtype=np.float64)
        chunks.append(block.reshape(-1, 2) / 32768.0)
    synth.delete()
    if not chunks:
        return np.zeros((total, 2), dtype=np.float64)
    return np.concatenate(chunks, axis=0)[:total]


def _metrics(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> dict[str, float]:
    standard = compare_audio(reference.mean(axis=1), estimate.mean(axis=1), sample_rate)
    standard["log_spectral_distance_db"] = log_spectral_distance(
        reference, estimate, sample_rate
    )
    return standard


def run_piano_risk_poc(config: PianoRiskPocConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(
        config.audio_path,
        always_2d=True,
        dtype="float64",
    )
    notes = read_notes(config.midi_path)
    mono = AudioData(audio.mean(axis=1), sample_rate)

    calibration_audio_onset = detect_onset_near(mono, config.calibration_midi_onset)
    heldout_audio_onset = score_aligned_onset(
        config.calibration_midi_onset,
        calibration_audio_onset,
        config.heldout_midi_onset,
    )
    heldout_end = score_aligned_onset(
        config.calibration_midi_onset,
        calibration_audio_onset,
        config.next_midi_onset,
    )
    excluded = ((config.calibration_midi_onset - 0.2, config.next_midi_onset + 0.2),)

    raw_samples: dict[int, np.ndarray] = {}
    harmonic_samples: dict[int, np.ndarray] = {}
    source_report: dict[int, object] = {}
    sample_count = round(config.source_duration * sample_rate)
    for pitch in config.pitches:
        source = select_isolated_source(notes, pitch, excluded_ranges=excluded)
        source_onset = detect_onset_near(mono, source.start)
        first = round(source_onset * sample_rate)
        sample = audio[first : first + sample_count].copy()
        if len(sample) < sample_count:
            sample = np.pad(sample, ((0, sample_count - len(sample)), (0, 0)))
        raw_samples[pitch] = sample
        harmonic_samples[pitch] = harmonic_mask_sample(sample, sample_rate, pitch)
        source_report[pitch] = {
            "midi": asdict(source),
            "detected_onset": source_onset,
            "overlap_count": overlap_count(notes, source),
        }

    for bank in (raw_samples, harmonic_samples):
        levels = {
            pitch: rms(sample[: round(0.18 * sample_rate)])
            for pitch, sample in bank.items()
        }
        target_level = float(np.median(list(levels.values())))
        for pitch, level in levels.items():
            bank[pitch] *= target_level / max(level, 1e-12)

    selected_events = [
        note
        for note in notes
        if config.calibration_midi_onset - 0.03 <= note.start < config.next_midi_onset + 0.03
        and note.note in config.pitches
    ]
    calibration_events = [
        note for note in selected_events if note.start < config.heldout_midi_onset - 0.05
    ]
    heldout_events = [
        note for note in selected_events if note.start >= config.heldout_midi_onset - 0.05
    ]
    calibration_duration = heldout_audio_onset - calibration_audio_onset
    heldout_duration = heldout_end - heldout_audio_onset
    calibration_reference = audio[
        round(calibration_audio_onset * sample_rate) : round(heldout_audio_onset * sample_rate)
    ]
    heldout_reference = audio[
        round(heldout_audio_onset * sample_rate) : round(heldout_end * sample_rate)
    ]

    raw_render = render_one_shot_bank(
        heldout_events,
        raw_samples,
        sample_rate,
        origin=heldout_audio_onset,
        duration=heldout_duration,
    )
    harmonic_render = render_one_shot_bank(
        heldout_events,
        harmonic_samples,
        sample_rate,
        origin=heldout_audio_onset,
        duration=heldout_duration,
    )
    note_gains = fit_note_spectral_gains(
        calibration_reference,
        calibration_events,
        harmonic_samples,
        sample_rate,
        origin=calibration_audio_onset,
        duration=calibration_duration,
    )
    weighted_render = render_one_shot_bank(
        heldout_events,
        harmonic_samples,
        sample_rate,
        origin=heldout_audio_onset,
        duration=heldout_duration,
        gains=note_gains,
    )
    gm_render = _render_fluidsynth(
        heldout_events,
        sample_rate,
        origin=heldout_audio_onset,
        duration=heldout_duration,
        soundfont_path=config.soundfont_path,
    )

    versions = {
        "gm": _gain_match(heldout_reference, gm_render),
        "raw_sampler": _gain_match(heldout_reference, raw_render),
        "harmonic_sampler": _gain_match(heldout_reference, harmonic_render),
        "nnls_sampler": _gain_match(heldout_reference, weighted_render),
    }
    _write_audio(config.output_dir / "01-original-heldout.wav", heldout_reference, sample_rate)
    for index, (name, samples) in enumerate(versions.items(), start=2):
        _write_audio(config.output_dir / f"{index:02d}-{name}.wav", samples, sample_rate)
    silence = np.zeros((round(0.45 * sample_rate), 2), dtype=np.float64)
    comparison = [heldout_reference]
    for samples in versions.values():
        comparison.extend((silence, samples))
    _write_audio(
        config.output_dir / "00-original-gm-raw-harmonic-nnls.wav",
        np.concatenate(comparison, axis=0),
        sample_rate,
    )

    sample_names: dict[int, str] = {}
    for pitch, samples in harmonic_samples.items():
        filename = f"Piano_{pitch}_harmonic.wav"
        sample_names[pitch] = filename
        _write_audio(config.output_dir / "Samples" / filename, samples, sample_rate)
    (config.output_dir / "Piano-Exact-Keys-Risk-POC.sfz").write_text(
        render_exact_key_sfz(sample_names),
        encoding="utf-8",
    )

    result_metrics = {
        name: _metrics(heldout_reference, samples, sample_rate)
        for name, samples in versions.items()
    }
    best_spectral = min(
        result_metrics,
        key=lambda name: result_metrics[name]["spectral_convergence"],
    )
    best_log_spectral = min(
        result_metrics,
        key=lambda name: result_metrics[name]["log_spectral_distance_db"],
    )
    report: dict[str, object] = {
        "segment": {
            "calibration_midi_onset": config.calibration_midi_onset,
            "calibration_audio_onset": calibration_audio_onset,
            "heldout_midi_onset": config.heldout_midi_onset,
            "heldout_audio_onset": heldout_audio_onset,
            "heldout_end": heldout_end,
            "pitches": config.pitches,
        },
        "sources": source_report,
        "calibration_note_gains": note_gains,
        "metrics": result_metrics,
        "decision": {
            "best_by_spectral_convergence": best_spectral,
            "best_by_log_spectral_distance": best_log_spectral,
            "sampler_beats_gm_by_spectral_convergence": min(
                result_metrics[name]["spectral_convergence"]
                for name in versions
                if name != "gm"
            )
            < result_metrics["gm"]["spectral_convergence"],
            "interpretation": (
                "Exact-pitch isolated samples preserve source colour in log spectra, but simple "
                "addition and per-note gain fitting do not reliably beat the generic piano on the "
                "held-out chord. Treat arbitrary polyphonic piano reconstruction as unsupported."
            ),
        },
    }
    (config.output_dir / "comparison-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
