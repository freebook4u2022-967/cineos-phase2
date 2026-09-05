"""Strict file and shot validation helpers."""

import hashlib
from pathlib import Path

from .exceptions import ValidationError


def file_hash(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValidationError(f"missing or empty output: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reusable_output(path: str | Path, expected_hash: str | None) -> bool:
    """Only permit resume reuse when the recorded hash still matches.

    Missing, empty, unreadable, or otherwise invalid artifacts are normal recovery
    inputs during resume, not fatal validator errors. Return ``False`` so the
    orchestrator can safely regenerate them while ``file_hash`` remains strict for
    newly accepted renderer outputs.
    """
    if not expected_hash:
        return False
    try:
        return file_hash(path) == expected_hash
    except (OSError, ValidationError):
        return False


class ShotValidator:
    """Default validator checks existence; production callers inject stricter policy."""

    def validate(self, path: str | Path, _shot: dict | None = None) -> dict:
        return {"approved": True, "checksum": file_hash(path), "warnings": []}
