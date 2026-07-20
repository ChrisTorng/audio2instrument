# audio2instrument

Experimental, deterministic pipeline for reconstructing a playable sample instrument from an
instrument audio track plus aligned MIDI. The first proof of concept targets monophonic electric
bass and emits an SFZ definition, extracted sample, adjusted MIDI, and offline reconstruction
render.

## Repository policy

The repository contains source code, tests, and documentation only. Copyrighted source audio,
MIDI files, extracted samples, generated renders, reports, model files, and downloaded archives
are intentionally excluded by `.gitignore`.

## Implemented pipeline

1. Inventory and match WAV/MIDI tracks inside ZIP archives.
2. Parse tempo-aware MIDI note events.
3. Detect acoustic onsets near MIDI note-on events.
4. Select a run of long, monophonic notes.
5. Extract a short root sample and find phase-compatible sustain-loop boundaries.
6. Generate an SFZ mapping with key range, sustain loop, loop crossfade, and release.
7. Render the MIDI excerpt offline with pitch shifting, sustain looping, and note-off release.
8. Export original/reconstructed/A-B WAV files and objective comparison metrics.

## Development on English Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Bass proof of concept

The input WAV and MIDI are deliberately external to the repository:

```powershell
audio2instrument bass-poc `
  --audio "D:\PrivateMedia\FADER-19_Bass_EFFECTED.wav" `
  --midi "D:\PrivateMedia\FADER-19_Bass_EFFECTED.mid" `
  --out "D:\PrivateMedia\bass-poc" `
  --search-start 110
```

Generated output:

```text
bass-poc/
├── 01-original-bass-excerpt.wav
├── 02-reconstructed-bass-excerpt.wav
├── 03-ab-original-then-reconstructed.wav
├── Bass.sfz
├── bass-poc.mid
├── comparison-report.json
└── Samples/
    └── Bass_45.wav
```

`Bass.sfz` can be opened in an SFZ player such as sfizz. The bundled offline renderer exists so
that the proof of concept and tests do not depend on an installed plug-in.

## Current limitations

- The proof of concept uses one velocity layer and one root sample.
- It assumes a monophonic track and uses audio-informed onset correction.
- Sustain-loop quality depends on the source note having a stable body.
- Recorded effects and room sound remain baked into the extracted sample.
- Objective waveform correlation is not expected to be high after pitch-shifting a single sample;
  spectral similarity and listening tests are more meaningful for this stage.

## License

MIT. Third-party audio and MIDI supplied by users or publishers retain their own copyrights and
are not covered by this repository's license.
