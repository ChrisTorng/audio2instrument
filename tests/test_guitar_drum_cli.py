from audio2instrument.cli import build_parser


def test_electric_guitar_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "electric-guitar-risk-poc",
            "--audio",
            "eg.wav",
            "--midi",
            "eg.mid",
            "--out",
            "out",
        ]
    )
    assert args.command == "electric-guitar-risk-poc"


def test_drum_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "drums-poc",
            "--audio-dir",
            "audio",
            "--midi-dir",
            "midi",
            "--out",
            "out",
        ]
    )
    assert args.command == "drums-poc"
