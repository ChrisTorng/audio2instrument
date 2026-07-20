from audio2instrument.cli import build_parser


def test_piano_risk_poc_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "piano-risk-poc",
            "--audio",
            "piano.wav",
            "--midi",
            "piano.mid",
            "--soundfont",
            "piano.sf2",
            "--out",
            "out",
        ]
    )
    assert args.command == "piano-risk-poc"
