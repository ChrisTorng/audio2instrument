from __future__ import annotations

from pathlib import Path
import re

_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_component(value: str) -> str:
    """Return a portable non-empty folder component for generated sample libraries."""
    result = _COMPONENT.sub("_", value.strip()).strip("._-")
    if not result:
        raise ValueError("sample folder component must contain a letter or number")
    return result


def instrument_sample_directory(instrument: str, *parts: str) -> Path:
    """Build Samples/<Instrument>/... so multiple SFZ files can share one parent folder."""
    path = Path("Samples") / safe_component(instrument)
    for part in parts:
        path /= safe_component(part)
    return path


def sfz_default_path(instrument: str, *parts: str) -> str:
    return instrument_sample_directory(instrument, *parts).as_posix() + "/"
