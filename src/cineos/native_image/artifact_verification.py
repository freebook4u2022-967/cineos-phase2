"""Cryptographic verification for released CINEOS native model artifacts.

A model manifest is only useful at production time if the bytes loaded by the runtime
actually match the component digests recorded in that manifest.  This module provides
a small, dependency-free verification boundary that can be used by release tooling and
production composition roots before any learned artifact is trusted.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .model_manifest import ModelManifestError, NativeModelManifest

_DEFAULT_CHUNK_BYTES = 1024 * 1024


class ModelArtifactVerificationError(ModelManifestError):
    """Raised when released model bytes do not match their signed manifest contract."""


@dataclass(frozen=True, slots=True)
class VerifiedModelComponent:
    """Measured integrity evidence for one native model component."""

    name: str
    path: str
    size_bytes: int
    expected_sha256: str
    actual_sha256: str


@dataclass(frozen=True, slots=True)
class ModelArtifactAttestation:
    """Deterministic verification evidence for all components in one manifest."""

    manifest_sha256: str
    components: tuple[VerifiedModelComponent, ...]

    @property
    def fingerprint(self) -> str:
        payload = {
            "manifest_sha256": self.manifest_sha256,
            "components": [asdict(component) for component in self.components],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return metadata that must remain stable throughout one verification read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256_regular_file(
    path: str | Path, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES
) -> tuple[str, int]:
    """Hash one stable regular file and return digest plus descriptor-backed size."""

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    source = Path(path)
    if source.is_symlink():
        raise ModelArtifactVerificationError(
            f"model artifact must not be a symbolic link: {source}"
        )
    if not source.exists():
        raise ModelArtifactVerificationError(f"model artifact is missing: {source}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ModelArtifactVerificationError(
            f"unable to open model artifact securely: {source}"
        ) from exc

    digest = hashlib.sha256()
    bytes_read = 0
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ModelArtifactVerificationError(
                    f"model artifact is not a regular file: {source}"
                )
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                bytes_read += len(chunk)
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except ModelArtifactVerificationError:
        raise
    except OSError as exc:
        raise ModelArtifactVerificationError(
            f"unable to read model artifact: {source}"
        ) from exc

    if _file_identity(before) != _file_identity(after):
        raise ModelArtifactVerificationError(
            f"model artifact changed while being verified: {source}"
        )
    if bytes_read == 0:
        raise ModelArtifactVerificationError(f"model artifact is empty: {source}")
    if bytes_read != before.st_size:
        raise ModelArtifactVerificationError(
            f"model artifact size changed while being verified: {source}"
        )
    return digest.hexdigest(), bytes_read


def sha256_file(path: str | Path, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> str:
    """Stream a stable regular file through SHA-256 without loading weights into RAM.

    Production verification rejects symbolic links and, where the platform supports
    ``O_NOFOLLOW``, also refuses a link introduced between validation and open. The
    open file descriptor is checked before and after hashing so an in-place rewrite
    cannot silently produce integrity evidence from bytes that changed mid-read.
    """

    digest, _ = _sha256_regular_file(path, chunk_bytes=chunk_bytes)
    return digest


def verify_component_artifacts(
    manifest: NativeModelManifest,
    artifact_paths: Mapping[str, str | Path],
    *,
    require_exact_set: bool = True,
) -> ModelArtifactAttestation:
    """Verify every component file against the active native model manifest.

    By default the supplied artifact mapping must match the manifest component set
    exactly.  This prevents a production configuration typo from silently verifying
    only a subset of the release or from carrying an unexpected extra learned artifact
    outside the versioned model contract.
    """

    manifest.validate()
    expected = {component.name: component for component in manifest.components}
    provided = {str(name): Path(path) for name, path in artifact_paths.items()}

    missing = sorted(set(expected).difference(provided))
    if missing:
        raise ModelArtifactVerificationError(
            "missing model component artifact(s): " + ", ".join(missing)
        )
    if require_exact_set:
        extras = sorted(set(provided).difference(expected))
        if extras:
            raise ModelArtifactVerificationError(
                "unexpected model component artifact(s): " + ", ".join(extras)
            )

    verified: list[VerifiedModelComponent] = []
    for name in sorted(expected):
        component = expected[name]
        source = provided[name]
        actual, size = _sha256_regular_file(source)
        if actual != component.artifact_sha256:
            raise ModelArtifactVerificationError(
                f"model artifact SHA-256 mismatch for component {name}"
            )
        verified.append(
            VerifiedModelComponent(
                name=name,
                path=str(source),
                size_bytes=size,
                expected_sha256=component.artifact_sha256,
                actual_sha256=actual,
            )
        )

    return ModelArtifactAttestation(
        manifest_sha256=manifest.manifest_sha256,
        components=tuple(verified),
    )


__all__ = [
    "ModelArtifactAttestation",
    "ModelArtifactVerificationError",
    "VerifiedModelComponent",
    "sha256_file",
    "verify_component_artifacts",
]
