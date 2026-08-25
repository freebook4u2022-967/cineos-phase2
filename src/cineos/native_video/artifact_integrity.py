"""Cryptographic integrity boundary for CINEOS native video artifacts.

Native continuity state must never silently outlive or diverge from the rendered
artifact that produced it. This module provides a small, reusable provenance
record and fail-closed verification helpers for render-time recording, checkpoint
resume, cache reuse, and future distributed artifact stores.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ArtifactIntegrityError(RuntimeError):
    """Raised when a native artifact cannot satisfy recorded provenance."""


class ArtifactProvenanceCarrier(Protocol):
    """Structural contract implemented by durable continuity anchors."""

    native_artifact_sha256: str | None
    native_artifact_bytes: int | None


@dataclass(frozen=True, slots=True)
class NativeArtifactProvenance:
    """Content-addressed identity of a completed native render artifact."""

    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal digest")
        if self.byte_size <= 0:
            raise ValueError("byte_size must be positive")


def provenance_for(path: str | Path) -> NativeArtifactProvenance:
    """Compute provenance for a real, non-empty artifact using streaming I/O."""
    artifact = Path(path)
    if not artifact.is_file():
        raise ArtifactIntegrityError(f"native artifact does not exist: {artifact}")

    byte_size = artifact.stat().st_size
    if byte_size <= 0:
        raise ArtifactIntegrityError(f"native artifact is empty: {artifact}")

    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return NativeArtifactProvenance(digest.hexdigest(), byte_size)


def verify_provenance(
    path: str | Path,
    *,
    sha256: str,
    byte_size: int,
) -> NativeArtifactProvenance:
    """Recompute and verify an artifact against durable expected provenance."""
    expected = NativeArtifactProvenance(str(sha256), int(byte_size))
    actual = provenance_for(path)
    if actual.byte_size != expected.byte_size:
        raise ArtifactIntegrityError(
            "native artifact byte size does not match durable continuity provenance"
        )
    if actual.sha256 != expected.sha256:
        raise ArtifactIntegrityError(
            "native artifact sha256 does not match durable continuity provenance"
        )
    return actual


def verify_continuity_artifact(
    carrier: ArtifactProvenanceCarrier,
    path: str | Path,
    *,
    require_provenance: bool = True,
) -> NativeArtifactProvenance | None:
    """Verify a render file against the provenance stored on a continuity anchor.

    Older checkpoints may predate artifact provenance. Callers performing strict
    production resume should keep ``require_provenance=True`` (the default). A
    migration/audit tool may opt out and receive ``None`` for legacy anchors.
    """
    digest = carrier.native_artifact_sha256
    byte_size = carrier.native_artifact_bytes
    if digest is None or byte_size is None:
        if require_provenance:
            raise ArtifactIntegrityError(
                "continuity anchor has no native artifact provenance"
            )
        return None
    return verify_provenance(path, sha256=digest, byte_size=byte_size)
