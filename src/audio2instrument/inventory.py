from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

_AUDIO_SUFFIXES = {".wav", ".flac", ".aif", ".aiff"}
_MIDI_SUFFIXES = {".mid", ".midi"}
_EFFECTED_SUFFIX_RE = re.compile(r"_EFFECTED$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TrackPair:
    key: str
    audio_member: str | None
    midi_member: str | None

    @property
    def complete(self) -> bool:
        return self.audio_member is not None and self.midi_member is not None


def normalize_track_key(path: str) -> str:
    stem = PurePosixPath(path).stem
    stem = _EFFECTED_SUFFIX_RE.sub("", stem)
    stem = re.sub(r"\s+", "_", stem.strip())
    return stem.casefold()


def list_media_members(zip_path: str | Path, suffixes: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            suffix = PurePosixPath(member).suffix.casefold()
            if suffix not in suffixes:
                continue
            key = normalize_track_key(member)
            if key in result:
                raise ValueError(f"Duplicate normalized track key {key!r} in {zip_path}")
            result[key] = member
    return result


def pair_archives(audio_zip: str | Path, midi_zip: str | Path) -> list[TrackPair]:
    audio = list_media_members(audio_zip, _AUDIO_SUFFIXES)
    midi = list_media_members(midi_zip, _MIDI_SUFFIXES)
    keys = sorted(audio.keys() | midi.keys())
    return [TrackPair(key=key, audio_member=audio.get(key), midi_member=midi.get(key)) for key in keys]


def inventory_json(audio_zip: str | Path, midi_zip: str | Path) -> str:
    pairs = pair_archives(audio_zip, midi_zip)
    payload = {
        "audio_zip": str(audio_zip),
        "midi_zip": str(midi_zip),
        "pair_count": sum(pair.complete for pair in pairs),
        "tracks": [asdict(pair) | {"complete": pair.complete} for pair in pairs],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
