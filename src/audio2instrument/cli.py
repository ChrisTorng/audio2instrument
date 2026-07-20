from __future__ import annotations

import argparse
from pathlib import Path

from audio2instrument import __version__
from audio2instrument.drum_poc import DrumKitPocConfig, DrumPieceConfig, run_drum_kit_poc
from audio2instrument.expressive_poc import BassExpressivePocConfig, run_bass_expressive_poc
from audio2instrument.guitar_risk_poc import (
    ElectricGuitarRiskConfig,
    run_electric_guitar_risk_poc,
)
from audio2instrument.inventory import inventory_json
from audio2instrument.multisample_poc import BassMultisamplePocConfig, run_bass_multisample_poc
from audio2instrument.piano_risk_poc import PianoRiskPocConfig, run_piano_risk_poc
from audio2instrument.poc import BassPocConfig, run_bass_poc
from audio2instrument.velocity_poc import BassVelocityPocConfig, run_bass_velocity_poc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio2instrument")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    inventory = subparsers.add_parser("inventory", help="Match audio and MIDI members in ZIPs")
    inventory.add_argument("--audio-zip", type=Path, required=True)
    inventory.add_argument("--midi-zip", type=Path, required=True)

    bass = subparsers.add_parser(
        "bass-poc", help="Build and render the three-note bass proof of concept"
    )
    bass.add_argument("--audio", type=Path, required=True)
    bass.add_argument("--midi", type=Path, required=True)
    bass.add_argument("--out", type=Path, required=True)
    bass.add_argument("--search-start", type=float, default=100.0)

    multisample = subparsers.add_parser(
        "bass-multisample-poc",
        help="Build a five-root bass SFZ and render same-segment and held-out validations",
    )
    multisample.add_argument("--audio", type=Path, required=True)
    multisample.add_argument("--midi", type=Path, required=True)
    multisample.add_argument("--out", type=Path, required=True)

    expressive = subparsers.add_parser(
        "bass-expressive-poc",
        help="Add attack-only round robins and similarity-gated key crossfades",
    )
    expressive.add_argument("--audio", type=Path, required=True)
    expressive.add_argument("--midi", type=Path, required=True)
    expressive.add_argument("--out", type=Path, required=True)

    velocity = subparsers.add_parser(
        "bass-velocity-poc",
        help="Build velocity-dependent bass attack layers and held-out validations",
    )
    velocity.add_argument("--audio", type=Path, required=True)
    velocity.add_argument("--midi", type=Path, required=True)
    velocity.add_argument("--out", type=Path, required=True)

    piano = subparsers.add_parser(
        "piano-risk-poc",
        help="Test isolated-note addition against a held-out four-note piano chord",
    )
    piano.add_argument("--audio", type=Path, required=True)
    piano.add_argument("--midi", type=Path, required=True)
    piano.add_argument("--soundfont", type=Path, required=True)
    piano.add_argument("--out", type=Path, required=True)

    guitar = subparsers.add_parser(
        "electric-guitar-risk-poc",
        help="Compare isolated-note addition with a held-out electric-guitar chord one-shot",
    )
    guitar.add_argument("--audio", type=Path, required=True)
    guitar.add_argument("--midi", type=Path, required=True)
    guitar.add_argument("--out", type=Path, required=True)

    drums = subparsers.add_parser(
        "drums-poc",
        help="Build articulation-aware drum one-shots with velocity and round robin",
    )
    drums.add_argument("--audio-dir", type=Path, required=True)
    drums.add_argument("--midi-dir", type=Path, required=True)
    drums.add_argument("--out", type=Path, required=True)
    return parser


def _drum_pieces(audio_dir: Path, midi_dir: Path) -> tuple[DrumPieceConfig, ...]:
    layout = (
        ("Kick", 0.80, 35),
        ("Snare", 0.65, 37),
        ("HiHat", 0.45, 42),
        ("Tom", 0.80, 47),
        ("Tambourine", 0.50, 51),
        ("Shaker", 0.28, 53),
        ("Conga", 0.55, 56),
    )
    return tuple(
        DrumPieceConfig(
            name=name,
            audio_path=audio_dir / f"{name}.wav",
            midi_path=midi_dir / f"{name}.mid",
            sample_duration=duration,
            first_target_key=first_key,
        )
        for name, duration, first_key in layout
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        print(inventory_json(args.audio_zip, args.midi_zip))
    elif args.command == "bass-poc":
        import json

        report = run_bass_poc(
            BassPocConfig(
                audio_path=args.audio,
                midi_path=args.midi,
                output_dir=args.out,
                search_start=args.search_start,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "bass-multisample-poc":
        import json

        report = run_bass_multisample_poc(
            BassMultisamplePocConfig(
                audio_path=args.audio,
                midi_path=args.midi,
                output_dir=args.out,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "bass-expressive-poc":
        import json

        report = run_bass_expressive_poc(
            BassExpressivePocConfig(
                audio_path=args.audio,
                midi_path=args.midi,
                output_dir=args.out,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "bass-velocity-poc":
        import json

        report = run_bass_velocity_poc(
            BassVelocityPocConfig(
                audio_path=args.audio,
                midi_path=args.midi,
                output_dir=args.out,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "piano-risk-poc":
        import json

        report = run_piano_risk_poc(
            PianoRiskPocConfig(
                audio_path=args.audio,
                midi_path=args.midi,
                soundfont_path=args.soundfont,
                output_dir=args.out,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "electric-guitar-risk-poc":
        import json

        report = run_electric_guitar_risk_poc(
            ElectricGuitarRiskConfig(
                audio_path=args.audio,
                midi_path=args.midi,
                output_dir=args.out,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "drums-poc":
        import json

        report = run_drum_kit_poc(
            DrumKitPocConfig(
                pieces=_drum_pieces(args.audio_dir, args.midi_dir),
                output_dir=args.out,
            )
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
