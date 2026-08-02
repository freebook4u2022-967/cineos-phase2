"""Release artifact checksum and content-policy helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .exceptions import ReleaseError

FORBIDDEN_SUFFIXES = {".ckpt", ".safetensors", ".pt", ".pth", ".ttf", ".otf", ".mov"}


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(checksums: dict[str, str], root: Path) -> tuple[str, ...]:
    return tuple(
        name
        for name, expected in sorted(checksums.items())
        if not (root / name).is_file() or checksum(root / name) != expected
    )


def enforce_content_policy(paths: list[Path]) -> None:
    forbidden = [
        str(path) for path in paths if path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise ReleaseError("forbidden bundled content: " + ", ".join(forbidden))
