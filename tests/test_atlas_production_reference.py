"""Regression tests for CINEOS-native production reference resolution."""

from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image

from cineos.atlas.production_reference import (
    PRODUCTION_REFERENCE_MANIFEST_SCHEMA,
    ProductionReferenceError,
    ProductionReferenceManifestLoader,
    _promote_native_reference_provenance,
)


def _manifest(tmp_path, *, approval_status="approved"):
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (8, 8), (12, 34, 56)).save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "references.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": PRODUCTION_REFERENCE_MANIFEST_SCHEMA,
                "references": [
                    {
                        "reference_id": "hero-front",
                        "file_path": "hero.png",
                        "sha256": digest,
                        "approval_status": approval_status,
                        "media_type": "image/png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, image_path, digest


def test_native_reference_loader_verifies_and_decodes_approved_image(tmp_path) -> None:
    manifest_path, _, _ = _manifest(tmp_path)
    loader = ProductionReferenceManifestLoader(manifest_path)

    image = loader("hero-front")

    assert image.mode == "RGB"
    assert image.size == (8, 8)
    assert loader.reference_ids == ("hero-front",)


def test_native_reference_loader_rejects_bytes_changed_after_approval(tmp_path) -> None:
    manifest_path, image_path, _ = _manifest(tmp_path)
    loader = ProductionReferenceManifestLoader(manifest_path)
    image_path.write_bytes(b"tampered")

    with pytest.raises(ProductionReferenceError, match="checksum changed"):
        loader("hero-front")


def test_native_reference_loader_rejects_unapproved_manifest_record(tmp_path) -> None:
    manifest_path, _, _ = _manifest(tmp_path, approval_status="pending")

    with pytest.raises(ProductionReferenceError, match="not approved"):
        ProductionReferenceManifestLoader(manifest_path)


def test_native_reference_provenance_promotes_only_reference_loader_boundary() -> None:
    provenance = {
        "schema": "cineos-gpu-runtime-provenance/0.1",
        "runtime_mode": "injected",
        "production_default_runtime": False,
        "cuda_device": "cuda:0",
        "dtype": "bfloat16",
        "injected_boundaries": {
            "torch_module": False,
            "reference_loader": True,
            "pipeline_factory": False,
            "video_exporter": False,
        },
    }

    promoted = _promote_native_reference_provenance(
        provenance,
        manifest_sha256="a" * 64,
        reference_ids=("hero-front",),
    )

    assert promoted["runtime_mode"] == "default"
    assert promoted["production_default_runtime"] is True
    assert promoted["injected_boundaries"]["reference_loader"] is False
    assert promoted["cineos_native_reference_manifest"] == {
        "schema": PRODUCTION_REFERENCE_MANIFEST_SCHEMA,
        "sha256": "a" * 64,
        "reference_ids": ["hero-front"],
    }


def test_native_reference_provenance_rejects_other_injected_boundary() -> None:
    provenance = {
        "schema": "cineos-gpu-runtime-provenance/0.1",
        "runtime_mode": "injected",
        "production_default_runtime": False,
        "cuda_device": "cuda:0",
        "dtype": "bfloat16",
        "injected_boundaries": {
            "torch_module": False,
            "reference_loader": True,
            "pipeline_factory": True,
            "video_exporter": False,
        },
    }

    with pytest.raises(
        ProductionReferenceError, match="only native reference resolution"
    ):
        _promote_native_reference_provenance(
            provenance,
            manifest_sha256="a" * 64,
            reference_ids=("hero-front",),
        )
