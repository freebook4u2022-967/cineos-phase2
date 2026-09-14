"""Cryptographic integrity evidence for acquired pretrained-foundation snapshots.

CINEOS may use transparently declared external pretrained foundations, but a pinned
repository revision alone does not prove which local bytes were present when a
production render executed. This module hashes the actual snapshot file contents in
stable relative-path order and supports fail-closed verification before and after a
GPU benchmark.

Hugging Face cache snapshots commonly contain symlinks into the cache blob store.
Files are therefore opened normally (following the symlink) so the digest covers the
bytes the model loader will read, while the manifest records that the snapshot entry
was a symlink. Absolute cache paths are deliberately excluded from the signed tree so
manifests remain portable across runners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FoundationSnapshotIntegrityError(RuntimeError):
    """Raised when foundation snapshot integrity cannot be established."""


_SCHEMA = "cineos-foundation-snapshot-integrity/0.1"
_CHUNK_SIZE = 8 * 1024 * 1024


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise FoundationSnapshotIntegrityError(
            f"cannot read foundation snapshot file: {path}"
        ) from exc
    return size, digest.hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FoundationSnapshotIntegrity:
    """Portable byte-level identity for one acquired foundation snapshot."""

    profile_id: str
    model_id: str
    revision: str
    files: tuple[dict[str, Any], ...]
    file_count: int
    total_bytes: int
    tree_sha256: str
    schema: str = _SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "tree_sha256": self.tree_sha256,
            "files": [dict(item) for item in self.files],
        }


def inspect_foundation_snapshot(
    snapshot_path: str | Path,
    *,
    profile_id: str,
    model_id: str,
    revision: str,
) -> FoundationSnapshotIntegrity:
    """Hash every regular snapshot entry and return deterministic integrity evidence."""

    if not all(
        isinstance(value, str) and value.strip()
        for value in (profile_id, model_id, revision)
    ):
        raise FoundationSnapshotIntegrityError(
            "profile_id, model_id and revision must be non-empty strings"
        )

    root = Path(snapshot_path)
    try:
        if not root.is_dir():
            raise FoundationSnapshotIntegrityError(
                f"foundation snapshot path is not a directory: {root}"
            )
        entries = sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    except OSError as exc:
        raise FoundationSnapshotIntegrityError(
            f"cannot enumerate foundation snapshot: {root}"
        ) from exc

    if not entries:
        raise FoundationSnapshotIntegrityError("foundation snapshot contains no files")

    files: list[dict[str, Any]] = []
    total_bytes = 0
    for path in entries:
        relative_path = path.relative_to(root).as_posix()
        size, digest = _sha256_file(path)
        total_bytes += size
        files.append(
            {
                "path": relative_path,
                "size_bytes": size,
                "sha256": digest,
                "symlink": path.is_symlink(),
            }
        )

    signed_payload = {
        "schema": _SCHEMA,
        "profile_id": profile_id,
        "model_id": model_id,
        "revision": revision,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }
    tree_sha256 = _canonical_digest(signed_payload)
    return FoundationSnapshotIntegrity(
        profile_id=profile_id,
        model_id=model_id,
        revision=revision,
        files=tuple(files),
        file_count=len(files),
        total_bytes=total_bytes,
        tree_sha256=tree_sha256,
    )


def write_foundation_snapshot_manifest(
    snapshot_path: str | Path,
    manifest_path: str | Path,
    *,
    profile_id: str,
    model_id: str,
    revision: str,
) -> FoundationSnapshotIntegrity:
    evidence = inspect_foundation_snapshot(
        snapshot_path,
        profile_id=profile_id,
        model_id=model_id,
        revision=revision,
    )
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def verify_foundation_snapshot_manifest(
    snapshot_path: str | Path,
    manifest_path: str | Path,
) -> FoundationSnapshotIntegrity:
    """Re-hash a snapshot and fail if any provenance or content field has drifted."""

    manifest_file = Path(manifest_path)
    try:
        expected = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FoundationSnapshotIntegrityError(
            f"cannot read foundation snapshot integrity manifest: {manifest_file}"
        ) from exc
    if not isinstance(expected, dict) or expected.get("schema") != _SCHEMA:
        raise FoundationSnapshotIntegrityError(
            "foundation snapshot integrity manifest has unsupported schema"
        )

    profile_id = expected.get("profile_id")
    model_id = expected.get("model_id")
    revision = expected.get("revision")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (profile_id, model_id, revision)
    ):
        raise FoundationSnapshotIntegrityError(
            "foundation snapshot integrity manifest is missing provenance identity"
        )

    observed = inspect_foundation_snapshot(
        snapshot_path,
        profile_id=profile_id,
        model_id=model_id,
        revision=revision,
    )
    observed_payload = observed.to_dict()
    if observed_payload != expected:
        raise FoundationSnapshotIntegrityError(
            "foundation snapshot bytes or provenance changed after acquisition"
        )
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify byte-level integrity evidence for an acquired foundation snapshot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--snapshot", required=True)
    create.add_argument("--manifest", required=True)
    create.add_argument("--profile-id", required=True)
    create.add_argument("--model-id", required=True)
    create.add_argument("--revision", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--snapshot", required=True)
    verify.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "create":
        evidence = write_foundation_snapshot_manifest(
            args.snapshot,
            args.manifest,
            profile_id=args.profile_id,
            model_id=args.model_id,
            revision=args.revision,
        )
    else:
        evidence = verify_foundation_snapshot_manifest(args.snapshot, args.manifest)
    print(json.dumps(evidence.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FoundationSnapshotIntegrity",
    "FoundationSnapshotIntegrityError",
    "inspect_foundation_snapshot",
    "main",
    "verify_foundation_snapshot_manifest",
    "write_foundation_snapshot_manifest",
]
