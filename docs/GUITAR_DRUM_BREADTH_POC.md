# Electric guitar and drum breadth validation

This stage stops refining bass and tests two different instrument classes.

## Output layout

All generated SFZ files may share one parent directory. Samples are therefore instrument-scoped:

- `Samples/ElectricGuitar/Notes/`
- `Samples/ElectricGuitar/Chords/`
- `Samples/Drums/Kick/`
- `Samples/Drums/Snare/`
- `Samples/Drums/HiHat/`
- `Samples/Drums/Tom/`
- `Samples/Drums/Tambourine/`
- `Samples/Drums/Shaker/`
- `Samples/Drums/Conga/`

## Electric guitar risk test

The held-out passage repeats MIDI notes 57, 61, and 64 as a short effected electric-guitar chord. One version sums three exact-pitch isolated samples. A second version replays a complete chord one-shot from another song location.

The chord one-shot is modestly closer than independent note addition. This supports chord or phrase sampling for effected polyphonic guitar and rejects a general promise that arbitrary guitar chords can be recreated by summing isolated notes.

## Drum kit test

Each direct drum stem is treated separately. MuScriptor pitch values inside a stem are preserved as distinct articulations, then remapped to non-conflicting custom keys. Each articulation receives up to three velocity layers and three round-robin samples.

Two held-out renders are produced:

- Automatic: articulation from MIDI pitch, velocity layer from attack level, deterministic round robin.
- Oracle nearest hit: chooses the closest training attack spectrum using held-out audio. This is an upper bound and is not MIDI-only playback.

The gap between automatic and oracle rendering measures the main current drum risk: reusable one-shots exist, but automatic articulation and velocity labeling is less reliable than the audio extraction itself.
