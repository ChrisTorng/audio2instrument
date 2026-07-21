# Strict single-event sampling

A note sample is not accepted merely because the aligned MIDI contains one note at its start. Earlier notes, sustain, room tails, stem leakage, and later attacks can remain in the audio window.

## Hard rules

For note and drum regions:

1. Locate the acoustic onset near the scored onset.
2. Reject simultaneous scored onsets for note-level extraction.
3. End the sample before both the next scored onset and the next acoustically detected onset.
4. Reserve a safety margin before the limiting event.
5. Remove pre-existing background energy using pre-onset spectral subtraction.
6. For pitched notes, retain target-harmonic residual energy and reject excessive non-target energy.
7. Detect attacks again in the exported sample. Note and drum samples must contain exactly one.
8. Prefer a clean 80–300 ms source attack over a long contaminated recording.

A chord or phrase can contain multiple notes only when it is explicitly labeled and mapped as a chord or phrase instrument. It must never be placed in a note-level SFZ directory.

## Short attacks and playable notes

When the recording does not contain a clean sustain body, the deterministic source attack can be followed by a generated single-pitch decay. Such a region must record:

- the source attack file and duration;
- the synthetic-tail flag;
- the synthesis method and parameters;
- the final single-event audit result.

The appended tail is synthetic completion, not recovered source audio.

## Directory convention

```text
Instrument.sfz
Samples/
└── Instrument/
    ├── Attacks/
    ├── HybridSingleNotes/
    ├── Chords/
    └── Phrases/
```

Drum pieces receive their own subdirectories below `Samples/Drums/`.

## Required audit fields

Each selected sample should report at least:

- scored onset;
- detected acoustic onset;
- exported duration;
- next-onset distance;
- detected onset count;
- target/non-target spectral ratio for pitched notes;
- acceptance or rejection reason;
- extracted, synthetic, fallback, chord, or phrase provenance.

The release package should include audition renders and a CSV identifying their playback order so every accepted sample can be checked without loading the SFZ.
