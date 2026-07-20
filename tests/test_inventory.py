from pathlib import Path
from zipfile import ZipFile

from audio2instrument.inventory import normalize_track_key, pair_archives


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_normalize_track_key_removes_effected_suffix() -> None:
    assert normalize_track_key("folder/FADER-19_Bass_EFFECTED.wav") == "fader-19_bass"


def test_pair_archives_matches_by_normalized_basename(tmp_path: Path) -> None:
    audio_zip = tmp_path / "audio.zip"
    midi_zip = tmp_path / "midi.zip"
    _write_zip(audio_zip, {"audio/FADER-19_Bass_EFFECTED.wav": b"audio", "audio/FADER-9_Piano_EFFECTED.wav": b"audio"})
    _write_zip(midi_zip, {"midi/FADER-19_Bass_EFFECTED.mid": b"midi", "midi/OnlyMidi.mid": b"midi"})
    pairs = pair_archives(audio_zip, midi_zip)
    by_key = {pair.key: pair for pair in pairs}
    assert by_key["fader-19_bass"].complete
    assert by_key["fader-9_piano"].midi_member is None
    assert by_key["onlymidi"].audio_member is None
