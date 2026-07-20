from audio2instrument.synth_confidence_cli import build_parser


def test_synth_confidence_command_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--audio",
            "synth.wav",
            "--midi",
            "synth.mid",
            "--out",
            "corrected.mid",
        ]
    )
    assert args.audio.name == "synth.wav"
    assert args.midi.name == "synth.mid"
    assert args.out.name == "corrected.mid"
    assert args.minimum_level == 0.001
    assert args.minimum_harmonic_ratio == 0.03
