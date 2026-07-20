from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio2instrument.synth_confidence import correct_midi_from_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m audio2instrument.synth_confidence_cli",
        description="Reject transcribed MIDI events that lack source-audio evidence.",
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-level", type=float, default=0.001)
    parser.add_argument("--minimum-harmonic-ratio", type=float, default=0.03)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = correct_midi_from_audio(
        args.audio,
        args.midi,
        args.out,
        minimum_level=args.minimum_level,
        minimum_harmonic_ratio=args.minimum_harmonic_ratio,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
