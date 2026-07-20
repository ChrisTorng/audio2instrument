from __future__ import annotations

import argparse
from pathlib import Path

from audio2instrument import __version__
from audio2instrument.inventory import inventory_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio2instrument")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    inventory = subparsers.add_parser("inventory", help="Match audio and MIDI members in ZIPs")
    inventory.add_argument("--audio-zip", type=Path, required=True)
    inventory.add_argument("--midi-zip", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        print(inventory_json(args.audio_zip, args.midi_zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
