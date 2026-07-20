# Full-song cover and Synth breadth validation

This validation renders the complete MIDI performances for the reconstructed Bass, Electric Guitar, Piano, and Drum instruments, then adds a Synth experiment.

## Instrument-scoped sample layout

All SFZ files may share one parent directory. Samples remain isolated by instrument:

- `Samples/Bass/`
- `Samples/Piano/`
- `Samples/ElectricGuitar/`
- `Samples/Drums/`
- `Samples/Synth/Low/`
- `Samples/Synth/Lead/`
- `Samples/Synth/Phrases/`

## Full-song coverage

- Bass: all 497 MIDI notes are covered by the existing multisample map.
- Drums: all 1,949 direct-stem hits map to an articulation key.
- Electric Guitar: 493 of 1,210 notes use one of the three observed exact pitches; 31 exact chord groups use the chord one-shot. The remaining notes require nearest-root pitch shifting.
- Piano: 230 of 1,374 notes use one of the four observed exact pitches. The remaining 1,144 notes require nearest-root pitch shifting.

The full-song cover is therefore a coverage test, not evidence that the current Piano and Electric Guitar instruments generalize across their complete ranges.

## Synth result

The transcribed Synth MIDI contains 191 events. Audio evidence retained only 85 events and rejected 106 events because the source audio had insufficient level or target-harmonic energy. A large block of rejected events occurs while the source stem is nearly silent.

The note-level Synth sampler remains weak because the source contains at least two registers/timbres, shared effects, and a combined Synth/SE stem. A phrase representation is highly accurate for the observed source sections, but it is not a general-purpose note instrument.

Observed clean-window results:

- Note sampler: correlation approximately -0.04 and SNR approximately -2.66 dB.
- Phrase sampler upper bound: correlation approximately 0.9997 and SNR approximately 32.6 dB.

## Main technical conclusion

The weakest broadly improvable stage is now event transcription and state labeling rather than sample cutting. The same failure mode appears in both Synth and Drums:

- false-positive or missing MIDI events;
- articulation labels that do not uniquely describe the sound;
- flat or unreliable velocity;
- no confidence score;
- no representation for open/closed, muted/sustained, phrase, or layered states.

Future extraction should treat MIDI as a proposal and validate every event against source-audio evidence before instrument construction.
