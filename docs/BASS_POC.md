# Bass proof of concept

## Test material

The development test uses an externally supplied 21-track educational multitrack and a separately
supplied MuScriptor MIDI bundle. Neither archive nor any extracted media is committed.

The selected source is the effected electric-bass track. It is the lowest-risk first instrument
because the passage is monophonic, has stable pitch, and contains three adjacent notes longer than
three seconds.

## Selected passage

The pipeline searches after 110 seconds and selects MIDI notes 45, 44, and 43. It detects the
actual acoustic attacks near the MIDI onsets, then crops the passage from the first attack through
the following note boundary.

The first note supplies a 1.35-second root sample. A deterministic two-stage search finds loop
boundaries by comparing normalized waveform shape, local slope, and RMS level. The offline renderer
uses an equal-power crossfade and gain-matches each successive loop iteration, allowing the sample
to sustain beyond its original duration without a hard discontinuity.

## Interpretation of metrics

The generated report includes:

- `spectral_convergence`: lower is better; this is the primary initial objective metric.
- `correlation`: phase-sensitive and expected to be low after resampling and looping.
- `snr_db`: also phase-sensitive; it is not a perceptual quality score.
- RMS and peak error: basic level and error diagnostics.

A/B listening remains required. The third WAV plays the original excerpt, 500 ms of silence, then
the reconstruction.

## Next technical increments

1. Add several root samples across the instrument range.
2. Infer velocity from local onset/body energy and create velocity zones.
3. Separate attack, body, and release layers.
4. Compare sfizz rendering with the internal deterministic renderer.
5. Add per-note sample selection and automatic rejection of contaminated candidates.
