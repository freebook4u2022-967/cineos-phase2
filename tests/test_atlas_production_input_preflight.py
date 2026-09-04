from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_input_preflight import (
    ProductionInputPreflightError,
    preflight_production_inputs,
)
from cineos.atlas.production_references import ProductionReferenceLoader


def _requests(count: int = 5, *, seed_offset: int = 0) -> tuple[NativeShotRequest, ...]:
    requests: list[NativeShotRequest] = []
    for index in range(count):
        shot_id = f"shot-{index + 1}"
        previous = None if index == 0 else f"shot-{index}"
        request = NativeShotRequest(
            shot_id=shot_id,
            scene_id="scene-1",
            camera={},
            characters=[{"character_id": "hero"}],
            environment=None,
            wardrobe=[],
            props=[],
            continuity={"previous_shot": previous},
            performance={},
            approved_reference_ids=["hero-front"],
            deterministic_seed=100 + index + seed_offset,
            renderer_requirements={},
        )
        request.refresh_hash()
        requests.append(request)
    return tuple(requests)


def _bundle_sha256(requests: tuple[NativeShotRequest, ...]) -> str:
    payload = json.dumps(
        [request.to_dict() for request in requests],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest(tmp_path: Path, *, approved_hash: str | None = None) -> Path:
    asset = tmp_path / "hero.ref"
    asset.write_bytes(b"approved-identity-payload")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "cineos-approved-reference-manifest/0.1",
                "references": [
                    {
                        "reference_id": "hero-front",
                        "path": asset.name,
                        "sha256": approved_hash or digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_preflight_validates_connected_graph_and_reference_before_model_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoded: list[str] = []

    def fake_decode(self: ProductionReferenceLoader, reference_id: str) -> object:
        decoded.append(reference_id)
        return object()

    monkeypatch.setattr(ProductionReferenceLoader, "__call__", fake_decode)
    requests = _requests()
    result = preflight_production_inputs(requests, _manifest(tmp_path))

    assert result["schema"] == "cineos-production-input-preflight/0.2"
    assert result["shot_count"] == 5
    assert result["request_bundle_sha256"] == _bundle_sha256(requests)
    assert result["request_content_hashes"] == [
        request.content_hash for request in requests
    ]
    assert result["reference_count"] == 1
    assert result["validated"] is True
    assert decoded == ["hero-front"]


def test_preflight_receipt_changes_when_valid_request_bundle_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ProductionReferenceLoader,
        "__call__",
        lambda self, reference_id: object(),
    )
    manifest = _manifest(tmp_path)
    original_requests = _requests()
    changed_requests = _requests(seed_offset=1)

    original = preflight_production_inputs(original_requests, manifest)
    changed = preflight_production_inputs(changed_requests, manifest)

    assert original["reference_manifest_sha256"] == changed["reference_manifest_sha256"]
    assert original["request_bundle_sha256"] != changed["request_bundle_sha256"]
    assert original["request_content_hashes"] != changed["request_content_hashes"]


def test_preflight_rejects_non_production_shot_count_before_reference_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoded = False

    def unexpected_decode(self: ProductionReferenceLoader, reference_id: str) -> object:
        nonlocal decoded
        decoded = True
        return object()

    monkeypatch.setattr(ProductionReferenceLoader, "__call__", unexpected_decode)
    with pytest.raises(ProductionInputPreflightError, match="between 5 and 10 shots"):
        preflight_production_inputs(_requests(4), _manifest(tmp_path))
    assert decoded is False


def test_preflight_rejects_changed_approved_reference_hash(tmp_path: Path) -> None:
    stale_hash = "0" * 64
    with pytest.raises(
        ProductionInputPreflightError, match="hash changed after approval"
    ):
        preflight_production_inputs(
            _requests(),
            _manifest(tmp_path, approved_hash=stale_hash),
        )


def test_gpu_workflow_preflights_before_foundation_snapshot_download() -> None:
    workflow = Path(".github/workflows/gpu-connected-production.yml").read_text(
        encoding="utf-8"
    )
    preflight = workflow.index(
        "Preflight connected production inputs before model acquisition"
    )
    snapshot = workflow.index(
        "Prefetch and verify immutable foundation and QC snapshots"
    )
    benchmark = workflow.index(
        "Run real connected GPU benchmark with production visual QC"
    )

    assert preflight < snapshot < benchmark
    assert "python -m cineos.atlas.production_input_preflight" in workflow
