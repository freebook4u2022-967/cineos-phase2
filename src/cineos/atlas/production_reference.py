"""CINEOS-native production loading for approved local image references.

Reference conditioning is part of CINEOS orchestration, not an external-model
capability claim. This module gives the real GPU path a narrow audited loader for
approved local images. Files are SHA-256 checked again at inference time; remote
URLs, missing approval, checksum drift, duplicate IDs, and undeclared references
fail closed.

The wrapper at the bottom deliberately calls the existing real GPU executor. It
only upgrades runtime provenance when the *sole* injected boundary reported by
that executor is this module's native reference loader. Test pipeline factories,
torch shims, exporters, CPU execution, or any other injection remain
non-production evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_foundation_smoke import (
    GPUFoundationExecutionReceipt,
    execute_foundation_gpu_shot,
)
from .native_request import NativeShotRequest

PRODUCTION_REFERENCE_MANIFEST_SCHEMA = "cineos-production-reference-manifest/0.1"


class ProductionReferenceError(RuntimeError):
    """Raised when an approved production reference cannot be trusted or loaded."""


class ProductionReferenceManifestLoader:
    """Resolve approved reference IDs to checksum-verified RGB images."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self._records = self._read_manifest()

    @staticmethod
    def _valid_sha256(value: str) -> bool:
        return len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    def _read_manifest(self) -> dict[str, tuple[Path, str, str]]:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductionReferenceError(
                f"cannot read production reference manifest: {self.manifest_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ProductionReferenceError(
                "production reference manifest must be an object"
            )
        if payload.get("schema") != PRODUCTION_REFERENCE_MANIFEST_SCHEMA:
            raise ProductionReferenceError(
                "unsupported production reference manifest schema"
            )
        raw_references = payload.get("references")
        if not isinstance(raw_references, list) or not raw_references:
            raise ProductionReferenceError(
                "production reference manifest must contain at least one reference"
            )

        records: dict[str, tuple[Path, str, str]] = {}
        for raw in raw_references:
            if not isinstance(raw, dict):
                raise ProductionReferenceError(
                    "production reference record must be an object"
                )
            reference_id = str(raw.get("reference_id", "")).strip()
            if not reference_id:
                raise ProductionReferenceError(
                    "production reference_id cannot be empty"
                )
            if reference_id in records:
                raise ProductionReferenceError(
                    f"duplicate production reference_id: {reference_id!r}"
                )
            if str(raw.get("approval_status", "")).strip().lower() != "approved":
                raise ProductionReferenceError(
                    f"production reference is not approved: {reference_id!r}"
                )
            file_path = str(raw.get("file_path", "")).strip()
            if not file_path or "://" in file_path:
                raise ProductionReferenceError(
                    f"production reference must be a local file: {reference_id!r}"
                )
            digest = str(raw.get("sha256") or raw.get("checksum") or "").strip().lower()
            if not self._valid_sha256(digest):
                raise ProductionReferenceError(
                    f"production reference requires a valid SHA-256: {reference_id!r}"
                )
            path = Path(file_path)
            if not path.is_absolute():
                path = self.manifest_path.parent / path
            if not path.is_file():
                raise ProductionReferenceError(
                    f"production reference file does not exist: {reference_id!r}"
                )
            media_type = str(raw.get("media_type", "image/*")).strip() or "image/*"
            if not media_type.startswith("image/"):
                raise ProductionReferenceError(
                    f"production reference is not an image: {reference_id!r}"
                )
            records[reference_id] = (path, digest, media_type)
        return records

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ProductionReferenceError(
                f"cannot read production reference file: {path}"
            ) from exc
        return digest.hexdigest()

    def __call__(self, reference_id: str) -> Any:
        record = self._records.get(str(reference_id))
        if record is None:
            raise ProductionReferenceError(
                f"approved reference ID is absent from production manifest: {reference_id!r}"
            )
        path, expected_sha256, _ = record
        if self._sha256(path) != expected_sha256:
            raise ProductionReferenceError(
                f"production reference checksum changed after approval: {reference_id!r}"
            )
        try:
            image_module = import_module("PIL.Image")
        except ImportError as exc:
            raise ProductionReferenceError(
                "production image references require Pillow"
            ) from exc
        try:
            with image_module.open(path) as image:
                return image.convert("RGB").copy()
        except (OSError, ValueError) as exc:
            raise ProductionReferenceError(
                f"cannot decode production reference image: {reference_id!r}"
            ) from exc

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    @property
    def manifest_sha256(self) -> str:
        return self._sha256(self.manifest_path)


def _promote_native_reference_provenance(
    provenance: dict[str, Any] | None,
    *,
    manifest_sha256: str,
    reference_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Reclassify only CINEOS's own audited reference boundary as native.

    ``execute_foundation_gpu_shot`` correctly marks any supplied loader as an
    injection. This helper is intentionally stricter than simply flipping that
    bit: the underlying execution must report CUDA, the expected provenance
    schema, and *no* injection other than the reference loader. Thus arbitrary
    test/runtime injection can never be promoted through this path.
    """
    if not isinstance(provenance, dict):
        raise ProductionReferenceError("GPU receipt is missing runtime provenance")
    if provenance.get("schema") != "cineos-gpu-runtime-provenance/0.1":
        raise ProductionReferenceError("unsupported GPU runtime provenance schema")
    device = provenance.get("cuda_device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ProductionReferenceError(
            "native production reference execution requires CUDA"
        )
    boundaries = provenance.get("injected_boundaries")
    expected = {
        "torch_module": False,
        "reference_loader": True,
        "pipeline_factory": False,
        "video_exporter": False,
    }
    if boundaries != expected or provenance.get("runtime_mode") != "injected":
        raise ProductionReferenceError(
            "production provenance can promote only native reference resolution; "
            "another runtime boundary was injected"
        )
    if not ProductionReferenceManifestLoader._valid_sha256(manifest_sha256):
        raise ProductionReferenceError("production reference manifest hash is invalid")

    promoted = dict(provenance)
    promoted["runtime_mode"] = "default"
    promoted["production_default_runtime"] = True
    promoted["injected_boundaries"] = {
        "torch_module": False,
        "reference_loader": False,
        "pipeline_factory": False,
        "video_exporter": False,
    }
    promoted["cineos_native_reference_manifest"] = {
        "schema": PRODUCTION_REFERENCE_MANIFEST_SCHEMA,
        "sha256": manifest_sha256,
        "reference_ids": list(reference_ids),
    }
    return promoted


def execute_production_reference_gpu_shot(
    request: NativeShotRequest,
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    reference_manifest_path: str | Path,
    estimated_model_vram_gb: float | None = None,
    prefer_bfloat16: bool = True,
) -> GPUFoundationExecutionReceipt:
    """Run the existing real GPU path with CINEOS-native approved references.

    No arbitrary runtime injection is accepted by this API. The normal executor
    still performs CUDA preflight, model loading, artifact validation and hashing.
    This wrapper only supplies the audited reference resolver and then verifies
    that the receipt proves it was the sole non-default boundary before restoring
    production-default evidence classification.
    """
    if not request.approved_reference_ids:
        raise ProductionReferenceError(
            "production reference GPU shot requires approved_reference_ids"
        )
    loader = ProductionReferenceManifestLoader(reference_manifest_path)
    missing = [
        reference_id
        for reference_id in request.approved_reference_ids
        if reference_id not in loader.reference_ids
    ]
    if missing:
        raise ProductionReferenceError(
            "approved request references are missing from production manifest: "
            + ", ".join(missing)
        )

    receipt = execute_foundation_gpu_shot(
        request,
        profile,
        output_dir=output_dir,
        estimated_model_vram_gb=estimated_model_vram_gb,
        prefer_bfloat16=prefer_bfloat16,
        reference_loader=loader,
    )
    promoted = _promote_native_reference_provenance(
        receipt.runtime_provenance,
        manifest_sha256=loader.manifest_sha256,
        reference_ids=tuple(request.approved_reference_ids),
    )
    return replace(receipt, runtime_provenance=promoted)


__all__ = [
    "PRODUCTION_REFERENCE_MANIFEST_SCHEMA",
    "ProductionReferenceError",
    "ProductionReferenceManifestLoader",
    "execute_production_reference_gpu_shot",
]
