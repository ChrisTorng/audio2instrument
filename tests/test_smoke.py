from audio2instrument import __version__
from audio2instrument.cli import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_smoke() -> None:
    assert main([]) == 0
