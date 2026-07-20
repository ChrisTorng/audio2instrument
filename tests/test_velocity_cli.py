from audio2instrument.cli import build_parser


def test_bass_velocity_poc_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "bass-velocity-poc",
            "--audio",
            "bass.wav",
            "--midi",
            "bass.mid",
            "--out",
            "out",
        ]
    )
    assert args.command == "bass-velocity-poc"
