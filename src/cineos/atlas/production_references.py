"""First-party production loader for immutable visual conditioning references.

The real GPU path needs approved identity/reference assets without turning a normal
production asset boundary into an "injected" test runtime. This module provides a
small fail-closed loader whose manifest and every referenced file are SHA-256 bound.
Remote URLs and unpinned files are intentionally rejected for benchmark evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

REFERENCE_MANIFEST_SCHEMA = "cineos-approved-reference-manifest/0.1"
REFERENCE_RUNTIME_SCHEMA = "cineos-production-reference-runtime/0.1"


class ProductionReferenceError(RuntimeError):
    """Raised when approved production reference evidence is incomplete or stale."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProductionReferenceError(
            f"cannot read approved reference: {path}"
        ) from exc
    return digest.hexdigest()


class ProductionReferenceLoader:
    """Resolve only locally pinned, hash-verified production reference images.

    Manifest format::

        {
          "schema": "cineos-approved-reference-manifest/0.1",
          "references": [
            {"reference_id": "hero-front", "path": "refs/hero.png",
             "sha256": "<64 hex>"}
          ]
        }

    Relative paths are resolved against the manifest directory. The manifest is
    hashed from its exact bytes, and each image is re-hashed immediately before it
    is opened so a post-approval file swap fails closed.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        try:
            manifest_bytes = self.manifest_path.read_bytes()
        except OSError as exc:
            raise ProductionReferenceError(
                f"cannot read approved reference manifest: {self.manifest_path}"
            ) from exc
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        try:
            payload = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProductionReferenceError(
                "approved reference manifest must be UTF-8 JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ProductionReferenceError(
                "approved reference manifest must be an object"
            )
        if payload.get("schema") != REFERENCE_MANIFEST_SCHEMA:
            raise ProductionReferenceError(
                f"unsupported approved reference manifest schema: {payload.get('schema')!r}"
            )
        raw_references = payload.get("references")
        if not isinstance(raw_references, Sequence) or isinstance(
            raw_references, (str, bytes)
        ):
            raise ProductionReferenceError(
                "approved reference manifest requires a references array"
            )

        root = self.manifest_path.parent
        references: dict[str, tuple[Path, str]] = {}
        for index, raw in enumerate(raw_references):
            if not isinstance(raw, Mapping):
                raise ProductionReferenceError(
                    f"approved reference entry {index} must be an object"
                )
            reference_id = raw.get("reference_id")
            raw_path = raw.get("path")
            sha256 = raw.get("sha256")
            if not isinstance(reference_id, str) or not reference_id.strip():
                raise ProductionReferenceError(
                    f"approved reference entry {index} has invalid reference_id"
                )
            reference_id = reference_id.strip()
            if reference_id in references:
                raise ProductionReferenceError(
                    f"duplicate approved reference_id: {reference_id!r}"
                )
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ProductionReferenceError(
                    f"approved reference {reference_id!r} has invalid path"
                )
            if (
                not isinstance(sha256, str)
                or len(sha256) != 64
                or any(
                    character not in "0123456789abcdef" for character in sha256.lower()
                )
            ):
                raise ProductionReferenceError(
                    f"approved reference {reference_id!r} requires a 64-character SHA-256"
                )
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            references[reference_id] = (path, sha256.lower())

        if not references:
            raise ProductionReferenceError(
                "approved reference manifest must contain at least one reference"
            )
        self._references = references

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(self._references)

    def validate_reference_ids(self, reference_ids: Iterable[str]) -> None:
        missing = sorted(
            {item for item in reference_ids if item not in self._references}
        )
        if missing:
            raise ProductionReferenceError(
                "approved reference manifest is missing requested ids: "
                + ", ".join(missing)
            )

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": REFERENCE_RUNTIME_SCHEMA,
            "loader": "cineos.atlas.production_references.ProductionReferenceLoader",
            "manifest_sha256": self.manifest_sha256,
            "reference_count": len(self._references),
        }

    def __call__(self, reference_id: str) -> Any:
        try:
            path, expected_sha256 = self._references[reference_id]
        except KeyError as exc:
            raise ProductionReferenceError(
                f"reference_id is not approved by the production manifest: {reference_id!r}"
            ) from exc
        if not path.is_file():
            raise ProductionReferenceError(
                f"approved reference is not a readable file: {path}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ProductionReferenceError(
                f"approved reference hash changed after approval: {reference_id!r}"
            )
        try:
            image_module = import_module("PIL.Image")
        except ImportError as exc:
            raise ProductionReferenceError(
                "production image references require Pillow from the video extra"
            ) from exc
        try:
            with image_module.open(path) as image:
                return image.convert("RGB").copy()
        except Exception as exc:
            raise ProductionReferenceError(
                f"approved reference is not a decodable image: {path}"
            ) from exc


__all__ = [
    "ProductionReferenceError",
    "ProductionReferenceLoader",
    "REFERENCE_MANIFEST_SCHEMA",
    "REFERENCE_RUNTIME_SCHEMA",
]
