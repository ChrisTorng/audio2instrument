# Bass velocity-layer proof of concept

V4 adds velocity-dependent attack timbre while retaining the stable sustain body and loop from V3.
Short notes are used only as attack donors: 140 ms of their onset is grafted onto a canonical long-note
sample, followed by a 50 ms equal-power transition back to the canonical sustain.

For each root note, candidates are filtered to reject silence, high-frequency separation artifacts, and
ambiguous same-pitch retriggers. A robust within-pitch score combines onset level, spectral centroid,
high-frequency ratio, and attack slope. Roots with enough observations receive soft, medium, and hard
layers; sparse roots receive two or one layer. Each layer may contain two round-robin attacks.

The three validation passages are excluded from candidate selection. V3 and V4 receive the same inferred
MIDI velocities, so the comparison isolates velocity-dependent timbre rather than amplitude automation.
Velocity is inferred from both level and brightness and is conservatively restricted to 16-112.

The evaluation reports ordinary audio metrics plus per-note timbre tracking. Waveform and full-spectrum
metrics may worsen because V4 deliberately changes attack waveforms. The load-bearing criteria are whether
spectral-centroid and high-frequency-ratio variation follow the held-out recording more closely.
