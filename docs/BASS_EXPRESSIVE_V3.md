# Bass expressive proof of concept

V3 tests two common sampler improvements against held-out material:

1. Sequential round robins for repeated notes.
2. Equal-power keyboard crossfades between neighboring root samples.

The raw round-robin experiment switched complete samples and produced more variation than the source.
The implemented version keeps only the first 90 ms of an alternate attack, transitions for 50 ms, and
then returns to the canonical sustain body and loop. Alternate attacks are blended toward the canonical
attack whenever their normalized log-spectrum distance exceeds 0.25.

Keyboard crossfades are not enabled blindly. Adjacent roots are pitch-shifted to a common target and
compared in attack and sustain windows. A pair must reach 0.94 similarity before SFZ `xfin` / `xfout`
opcodes are emitted. In the current bass material, 35-43 scored about 0.907 and 45-47 about 0.903, so
both pairs remain hard-zoned. Mixing these non-phase-aligned samples increased spectral error.

The generated SFZ uses `seq_length` and `seq_position` for round robins. Compatible root pairs, when
present, use `xf_keycurve=power` with keyboard fade ranges. The internal renderer follows the same
selection rules used to generate the SFZ.

Validation outputs compare:

- Original audio.
- Hard-zone, single-variant baseline.
- V3 adaptive instrument.

The repetition validation also reports normalized onset-spectrum diversity. This metric is more useful
than waveform correlation for machine-gun repetition because intentionally different attacks cannot be
sample-aligned with the original waveform.
