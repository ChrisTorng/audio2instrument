# audio2instrument

Experimental, deterministic tooling for turning an isolated instrument track plus aligned MIDI into
playable sample instruments and reproducible reconstruction renders.

The project does **not** assume that every source can be represented by one traditional multisample
instrument. Depending on the material, it can use note samples, drum one-shots, chord samples,
phrase samples, source-volume automation, or a hybrid fallback.

## Repository policy

The repository contains source code, tests, and documentation only. Source audio, MIDI supplied for
experiments, extracted samples, generated renders, reports, model files, and downloaded archives are
excluded by `.gitignore`.

Public documentation uses generic names such as `instrument.wav`, `performance.mid`, and
`project-output/`. Internal experiment titles, publisher filenames, song names, and workstation paths
must not be committed.

## Current capabilities

- Inventory and pair audio/MIDI tracks inside archives.
- Parse tempo-aware MIDI note events.
- Correct MIDI events with acoustic onset and audio-confidence evidence.
- Extract root samples and map them across observed pitch ranges.
- Find sustain-loop candidates and render note-off release envelopes.
- Build multisample SFZ instruments with velocity layers and round robin.
- Build articulation-aware drum one-shot instruments.
- Compare isolated-note addition with chord or phrase sampling for polyphonic sources.
- Route synthesizer material by explicit sound-state sections rather than pitch range.
- Extract time-varying loudness envelopes from the source track and apply them to reconstructions.
- Keep samples separated by instrument so multiple SFZ files can share one parent directory.
- Produce deterministic offline renders and objective comparison reports.

## Representation strategy

The preferred representation is selected from the evidence available in the source:

| Source material | Preferred representation |
|---|---|
| Monophonic pitched instrument | Multisample SFZ |
| Repeated drum or percussion hits | Velocity/round-robin one-shots |
| Distorted or strongly processed chord | Chord sample |
| Sequenced or automated sound | Phrase sample or sound-state instrument |
| Unreliable MIDI event | Reject, regenerate, or preserve source residual |
| Missing pitch or articulation coverage | Generic-instrument fallback or synthetic completion |

A generated instrument is only expected to be reliable inside the pitch, velocity, articulation, and
sound-state coverage supported by the source material. Large extrapolations must be reported rather
than silently presented as recovered source audio.

## Output layout

SFZ files may be stored together while samples remain isolated by instrument:

```text
project-output/
├── Bass.sfz
├── ElectricGuitar.sfz
├── Drums.sfz
├── SynthLead.sfz
└── Samples/
    ├── Bass/
    ├── ElectricGuitar/
    ├── Drums/
    └── SynthLead/
```

## Example workflow

Input media remain outside the repository:

```powershell
audio2instrument bass-multisample-poc `
  --audio "D:\AudioProject\stems\bass.wav" `
  --midi "D:\AudioProject\midi\bass.mid" `
  --out "D:\AudioProject\generated\bass-instrument"
```

A typical generated directory contains an SFZ file, instrument-scoped samples, validation MIDI,
comparison audio, and a machine-readable report:

```text
bass-instrument/
├── Bass.sfz
├── validation.mid
├── comparison-report.json
├── validation/
│   ├── original.wav
│   └── reconstructed.wav
└── Samples/
    └── Bass/
        ├── root-01.wav
        └── root-02.wav
```

The SFZ can be opened in a player such as sfizz. The bundled offline renderers keep tests and
experiments independent of an installed audio plug-in.

## AI-assisted synthetic completion

Generative audio can potentially fill regions that cannot be extracted reliably, including:

- missing root pitches or velocity layers;
- alternate attacks and drum articulations;
- clean sustain bodies for notes that are too short or contaminated;
- release tails, fret/key noise, room tails, and other residual layers;
- a replacement phrase when MIDI evidence is incomplete but the intended part is known.

The strongest conditioning order is generally:

1. a clean source-audio example from the same sound state;
2. timing, pitch, duration, velocity, and articulation constraints from MIDI or an event graph;
3. a human-written text description of the intended sound;
4. optional references for room, amplifier, effects, or production style.

Human-provided descriptions are therefore useful. A description should state the instrument or
synthesis method, register, articulation, dynamics, duration, envelope, brightness, effects, stereo
perspective, and musical role. For example:

```text
Short muted electric-guitar power chord, medium pick attack, low register, tight palm mute,
moderate amplifier saturation, fast decay, dry close-miked recording, no rhythmic accompaniment.
```

Generated content must be labeled **synthetic**, retain its prompt/model/seed provenance, and be
validated against held-out source audio. It is not evidence that missing information was recovered
from the recording. For identity-critical material, deterministic samples or preserved source
residuals remain preferable.

AI completion is currently a roadmap component, not a replacement for the deterministic extraction
pipeline. Candidate backends include text-to-audio, audio-to-audio/inpainting, and pitch/loudness-
conditioned neural synthesis.

## StemForge interoperability

StemForge and audio2instrument address adjacent stages:

- StemForge can provide separation, MIDI extraction, generic SoundFont preview, generation, and mix
  infrastructure.
- audio2instrument focuses on constructing source-specific note, drum, chord, and phrase instruments
  from the separated tracks and aligned event data.

A practical combined workflow is:

```text
song
→ separation and MIDI extraction
→ audio-validated event graph
→ source-specific extraction where coverage is reliable
→ generic or AI-generated completion where coverage is missing
→ source-volume automation
→ final mix and export
```

## Development

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m pytest
```

GitHub Actions uploads Ruff and Pytest diagnostics for each pull request.

## Current limitations

- Audio and MIDI alignment can still fail on masked or legato onsets.
- Polyphonic recordings are not guaranteed to decompose into independently reusable notes.
- Recorded effects, compression, ambience, and physical resonances may be non-additive.
- MIDI often lacks pedal, articulation, pick/bow direction, mute state, and phrase identity.
- Drum quality is limited more by articulation classification and sample selection than extraction.
- Generated or fallback content may preserve musical function without preserving the exact source
  identity.
- Waveform correlation is usually not meaningful after sample replacement or pitch shifting;
  spectral, envelope, event-coverage, and listening evaluations are required together.

## License

MIT. Third-party audio, MIDI, prompts, model weights, and generated assets retain their respective
licenses and are not covered by this repository's MIT license.