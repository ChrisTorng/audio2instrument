from audio2instrument import __version__
from audio2instrument.cli import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_smoke() -> None:
    assert main([]) == 0


def test_bass_poc_command_is_registered() -> None:
    from audio2instrument.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "bass-poc",
            "--audio",
            "bass.wav",
            "--midi",
            "bass.mid",
            "--out",
            "out",
        ]
    )
    assert args.command == "bass-poc"


def test_bass_multisample_poc_command_is_registered() -> None:
    from audio2instrument.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "bass-multisample-poc",
            "--audio",
            "bass.wav",
            "--midi",
            "bass.mid",
            "--out",
            "out",
        ]
    )
    assert args.command == "bass-multisample-poc"


def test_bass_expressive_poc_command_is_registered() -> None:
    from audio2instrument.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "bass-expressive-poc",
            "--audio",
            "bass.wav",
            "--midi",
            "bass.mid",
            "--out",
            "out",
        ]
    )
    assert args.command == "bass-expressive-poc"
