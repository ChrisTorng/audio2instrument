import pytest

from audio2instrument.sample_layout import instrument_sample_directory, safe_component, sfz_default_path


def test_instrument_folders_are_not_flat() -> None:
    assert instrument_sample_directory("Electric Guitar", "Notes").as_posix() == "Samples/Electric_Guitar/Notes"
    assert sfz_default_path("Drums", "Kick") == "Samples/Drums/Kick/"


def test_invalid_component_is_rejected() -> None:
    with pytest.raises(ValueError):
        safe_component("***")
