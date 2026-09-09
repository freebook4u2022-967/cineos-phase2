"""Deterministic byte-level attestation for external pretrained model snapshots.

CINEOS may execute legally usable external pretrained foundations, but production
receipts must prove which local bytes were actually made available for execution.
This module hashes a resolved model snapshot without claiming ownership of weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class ModelSnapshotAttestationError(RuntimeError):
    """Raised when a model snapshot cannot be attested or no longer matches."""


@dataclass(frozen=True, slots=True)
class ModelSnapshotFile:
    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ModelSnapshotAttestation:
    model_id: str
    revision: str
    files: tuple[ModelSnapshotFile, ...]
    snapshot_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "cineos-model-snapshot-attestation/0.1",
            "model_id": self.model_id,
            "revision": self.revision,
            "files": [item.to_dict() for item in self.files],
            "snapshot_sha256": self.snapshot_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attest_model_snapshot(
    snapshot_dir: str | Path, *, model_id: str, revision: str
) -> ModelSnapshotAttestation:
    """Hash every regular file in a resolved immutable model snapshot.

    Symlinks are rejected rather than followed so an attestation cannot silently bind
    mutable bytes outside the snapshot root. Ordering is canonical by POSIX relative
    path, making the aggregate digest deterministic across filesystem enumeration.
    """

    root = Path(snapshot_dir)
    if not model_id.strip() or not revision.strip():
        raise ModelSnapshotAttestationError("model_id and revision must be non-empty")
    if not root.is_dir():
        raise ModelSnapshotAttestationError("model snapshot directory does not exist")

    entries: list[ModelSnapshotFile] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ModelSnapshotAttestationError(
                f"model snapshot contains unsupported symlink: {path.relative_to(root).as_posix()}"
            )
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            entries.append(ModelSnapshotFile(relative, path.stat().st_size, _sha256(path)))
    if not entries:
        raise ModelSnapshotAttestationError("model snapshot contains no files")

    canonical = json.dumps(
        {
            "model_id": model_id,
            "revision": revision,
            "files": [item.to_dict() for item in entries],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ModelSnapshotAttestation(
        model_id=model_id,
        revision=revision,
        files=tuple(entries),
        snapshot_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def verify_model_snapshot(
    snapshot_dir: str | Path, expected: ModelSnapshotAttestation
) -> ModelSnapshotAttestation:
    """Re-attest immediately before execution and fail closed on byte drift."""

    observed = attest_model_snapshot(
        snapshot_dir, model_id=expected.model_id, revision=expected.revision
    )
    if observed != expected:
        raise ModelSnapshotAttestationError(
            "model snapshot bytes do not match the approved production attestation"
        )
    return observed


__all__ = [
    "ModelSnapshotAttestation",
    "ModelSnapshotAttestationError",
    "ModelSnapshotFile",
    "attest_model_snapshot",
    "verify_model_snapshot",
]
