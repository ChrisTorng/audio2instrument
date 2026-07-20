from __future__ import annotations

import argparse
from pathlib import Path

from audio2instrument import __version__
from audio2instrument.inventory import inventory_json
from audio2instrument.poc import BassPocConfig, run_bass_poc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio2instrument")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    inventory = subparsers.add_parser("inventory", help="Match audio and MIDI members in ZIPs")
    inventory.add_argument("--audio-zip", type=Path, required=True)
    inventory.add_argument("--midi-zip", type=Path, required=True)

    bass = subparsers.add_parser("bass-poc", help="Build and render the three-note bass proof of concept")
    bass.add_argument("--audio", type=Path, required=True)
    bass.add_argument("--midi", type=Path, required=True)
    bass.add_argument("--out", type=Path, required=True)
    bass.add_argument("--search-start", type=float, default=100.0)
    return parser


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
