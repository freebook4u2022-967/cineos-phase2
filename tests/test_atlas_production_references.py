import hashlib
import json

import pytest

from cineos.atlas.production_references import (
    ProductionReferenceError,
    ProductionReferenceLoader,
    bind_production_reference_runtime,
)


def _manifest(tmp_path, *, reference_id="hero-front"):
    image = tmp_path / "hero.png"
    image.write_bytes(b"approved-reference-bytes")
    payload = {
        "schema": "cineos-approved-reference-manifest/0.1",
        "references": [
            {
                "reference_id": reference_id,
                "path": image.name,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest = tmp_path / "references.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def _runtime(*, reference_injected=True, pipeline_injected=False):
    injected = {
        "torch_module": False,
        "reference_loader": reference_injected,
        "pipeline_factory": pipeline_injected,
        "video_exporter": False,
    }
    return {
        "schema": "cineos-gpu-runtime-provenance/0.1",
        "runtime_mode": "injected" if any(injected.values()) else "default",
        "production_default_runtime": not any(injected.values()),
        "cuda_device": "cuda:0",
        "injected_boundaries": injected,
    }


def test_first_party_hash_bound_loader_is_standard_production_boundary(tmp_path):
    loader = ProductionReferenceLoader(_manifest(tmp_path))

    runtime = bind_production_reference_runtime(_runtime(), loader)

    assert runtime["runtime_mode"] == "default"
    assert runtime["production_default_runtime"] is True
    assert runtime["injected_boundaries"]["reference_loader"] is False
    assert runtime["reference_assets"]["manifest_sha256"] == loader.manifest_sha256
    assert runtime["reference_assets"]["reference_count"] == 1


def test_arbitrary_reference_callable_remains_injected():
    runtime = bind_production_reference_runtime(
        _runtime(), lambda _reference_id: object()
    )

    assert runtime["runtime_mode"] == "injected"
    assert runtime["production_default_runtime"] is False
    assert runtime["injected_boundaries"]["reference_loader"] is True
    assert "reference_assets" not in runtime


def test_first_party_loader_cannot_hide_other_injected_boundaries(tmp_path):
    loader = ProductionReferenceLoader(_manifest(tmp_path))

    runtime = bind_production_reference_runtime(
        _runtime(pipeline_injected=True), loader
    )

    assert runtime["runtime_mode"] == "injected"
    assert runtime["production_default_runtime"] is False
    assert runtime["injected_boundaries"]["reference_loader"] is False
    assert runtime["injected_boundaries"]["pipeline_factory"] is True


def test_manifest_must_cover_every_requested_reference(tmp_path):
    loader = ProductionReferenceLoader(_manifest(tmp_path))

    with pytest.raises(ProductionReferenceError, match="partner-front"):
        loader.validate_reference_ids(("hero-front", "partner-front"))


def test_manifest_rejects_duplicate_reference_ids(tmp_path):
    image = tmp_path / "hero.png"
    image.write_bytes(b"approved-reference-bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "cineos-approved-reference-manifest/0.1",
                "references": [
                    {"reference_id": "hero", "path": image.name, "sha256": digest},
                    {"reference_id": "hero", "path": image.name, "sha256": digest},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ProductionReferenceError, match="duplicate approved reference_id"
    ):
        ProductionReferenceLoader(manifest)
