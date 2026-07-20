# StemForge downstream comparison

This comparison holds the official source stems and supplied MuScriptor MIDI constant. It isolates the downstream synthesis and mixing approaches instead of conflating them with stem separation or transcription quality.

## Compared paths

- **audio2instrument:** source-stem-derived note samples, drum one-shots, guitar chord samples, and Synth phrase instruments.
- **StemForge-style baseline:** FluidSynth with the TimGM6mb GM SoundFont, corresponding to StemForge's documented MIDI preview and Mix workflow.

StemForge BasicPitch was not executed in the validation runtime because its model/runtime was unavailable. The comparison therefore does not claim to measure StemForge transcription quality.

## Original-song volume automation

Both renderers receive a source-derived, time-varying level curve for each track:

- 200 ms RMS analysis window
- 50 ms hop
- active-region global level alignment
- local gain limited to -18 dB through +12 dB
- median and Gaussian smoothing
- source-silence gate

This prevents a single global RMS adjustment from hiding arrangement-level dynamics.

## Synth representation

Synth sounds are separate instruments identified by sound state and source section, not by pitch register:

- `Synth_IntroPulse`
- `Synth_Build`
- `Synth_LeadA`
- `Synth_LeadB`

Each group has an independent SFZ, sample directory, score MIDI, and phrase-trigger MIDI. MIDI pitch is not used to decide which Synth instrument owns an event.

## Result summary

For the five-track full-song mix, the stem-derived audio2instrument route produced a lower sampled log-spectral distance and slightly better source-volume-envelope tracking than the GM/SoundFont baseline. The advantage was strongest for drums and source-phrase Synth material. GM rendering remained competitive for Bass and preserved the Piano and electric-guitar activity envelopes more consistently where the stem-derived libraries had sparse pitch or articulation coverage.

The comparison supports a hybrid architecture: stem-derived instruments where source coverage is sufficient, and a fallback renderer or retained residual where it is not.
