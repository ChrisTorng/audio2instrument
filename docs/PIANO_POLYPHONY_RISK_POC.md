# Piano polyphony risk proof of concept

This experiment deliberately tests the highest-risk assumption in the project: whether notes sampled
from other isolated occurrences can be added together to reconstruct a processed polyphonic piano
chord.

The default validation uses two consecutive occurrences of the same four-note chord. The first chord
provides only score/audio alignment and optional non-negative per-note gain calibration. The second
chord is held out. Exact-pitch source notes are selected from other parts of the song.

Generated comparisons are:

1. a generic General MIDI piano rendered through FluidSynth;
2. direct addition of isolated source-note samples;
3. score-informed harmonic masking before sample addition;
4. non-negative per-note spectral weights learned on the first chord and applied to the second.

A negative result is expected to be informative. Failure after exact MIDI, exact-pitch isolated source
notes, and held-out repetition indicates that room effects, compression, resonance, pedal state, and
phase interactions are not reducible to independent per-note gains. It does not invalidate the proven
monophonic and percussion use cases, but it rules out a universal claim for sample-based reconstruction.

The FluidSynth baseline is optional and requires the `pyfluidsynth` Python package, the system
FluidSynth library, and a General MIDI SoundFont. No SoundFont or test media is committed.
