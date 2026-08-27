"""Cryptographic verification for released CINEOS native model artifacts.

A model manifest is only useful at production time if the bytes loaded by the runtime
actually match the component digests recorded in that manifest.  This module provides
a small, dependency-free verification boundary that can be used by release tooling and
production composition roots before any learned artifact is trusted.
"""

from __future__ import annotations

import hashlib
import json
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


def sha256_file(path: str | Path, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> str:
    """Stream a regular file through SHA-256 without loading model weights into RAM."""

    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    source = Path(path)
    if not source.exists():
        raise ModelArtifactVerificationError(f"model artifact is missing: {source}")
    if not source.is_file():
        raise ModelArtifactVerificationError(
            f"model artifact is not a regular file: {source}"
        )

    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ModelArtifactVerificationError(
            f"unable to read model artifact: {source}"
        ) from exc

    if size == 0:
        raise ModelArtifactVerificationError(f"model artifact is empty: {source}")
    return digest.hexdigest()


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
        actual = sha256_file(source)
        if actual != component.artifact_sha256:
            raise ModelArtifactVerificationError(
                f"model artifact SHA-256 mismatch for component {name}"
            )
        try:
            size = source.stat().st_size
        except OSError as exc:
            raise ModelArtifactVerificationError(
                f"unable to stat verified model artifact: {source}"
            ) from exc
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
