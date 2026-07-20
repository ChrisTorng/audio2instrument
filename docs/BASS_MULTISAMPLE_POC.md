# Bass multisample proof of concept

The second proof of concept removes the single-root limitation without committing any test media.
It selects the strongest long-note candidates by MIDI structure, extracts up to five root samples,
normalizes their onset level, finds an independent sustain loop for each sample, and maps the roots
across the observed MIDI range by nearest-note boundaries.

The renderer infers MIDI velocity from local onset amplitude after the samples have been normalized
to a common reference level. This keeps dynamics in the MIDI rather than baking every level change
into per-note gain automation.

Two validations are generated:

- `validation-same`: an upper bound containing the three long notes that supplied three roots.
- `validation-held-out`: a different performance passage, used to test whether the instrument can
  replay later instances rather than merely copying the sampled occurrence.

The same-segment result is expected to score better and is not evidence of generalization. The
held-out result is the more meaningful listening test.
