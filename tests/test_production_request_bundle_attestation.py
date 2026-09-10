import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cineos.atlas.production_request_bundle_attestation as bundle_attestation
from cineos.atlas.production_request_bundle_attestation import (
    ProductionRequestBundleAttestationError,
    validate_request_bundle_preflight,
    verify_production_request_bundle_attestation,
    write_production_request_bundle_attestation,
)


def _request(shot_id: str, payload: str):
    return SimpleNamespace(
        shot_id=shot_id,
        content_hash=hashlib.sha256(payload.encode()).hexdigest(),
    )


def _requests():
    return tuple(_request(f"shot-{index}", f"payload-{index}") for index in range(5))


def _bundle_hash(requests) -> str:
    payload = [
        {"shot_id": request.shot_id, "content_hash": request.content_hash}
        for request in requests
    ]
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_preflight(root: Path, requests) -> Path:
    path = root / "connected-benchmark-fixture-preflight.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cineos-connected-benchmark-fixture-preflight/0.2",
                "validated": True,
                "shot_count": len(requests),
                "ordered_shot_ids": [request.shot_id for request in requests],
                "normalized_request_bundle_sha256": _bundle_hash(requests),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _quality_payload(requests, *, order=None):
    ordered = list(requests if order is None else (requests[index] for index in order))
    return {
        "attestation_sha256": "a" * 64,
        "connected_benchmark": {
            "shots": [
                {
                    "result": {
                        "shot_id": request.shot_id,
                        "request_hash": request.content_hash,
                    }
                }
                for request in ordered
            ]
        },
    }


def _write_quality(root: Path, payload) -> Path:
    path = root / "quality-first-production-attestation.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_inference_gate_accepts_exact_preflight_bundle(monkeypatch, tmp_path):
    requests = _requests()
    preflight = _write_preflight(tmp_path, requests)
    monkeypatch.setattr(bundle_attestation, "load_native_requests", lambda _: requests)

    result = validate_request_bundle_preflight("requests.json", preflight)

    assert result["validated"] is True
    assert result["shot_count"] == 5
    assert result["ordered_shot_ids"] == [request.shot_id for request in requests]
    assert result["normalized_request_bundle_sha256"] == _bundle_hash(requests)


def test_inference_gate_rejects_request_content_changed_after_preflight(
    monkeypatch, tmp_path
):
    requests = _requests()
    preflight = _write_preflight(tmp_path, requests)
    changed = list(requests)
    changed[2] = _request("shot-2", "substituted-payload")
    monkeypatch.setattr(
        bundle_attestation, "load_native_requests", lambda _: tuple(changed)
    )

    with pytest.raises(
        ProductionRequestBundleAttestationError,
        match="content changed after connected benchmark preflight",
    ):
        validate_request_bundle_preflight("requests.json", preflight)


def test_final_attestation_binds_preflight_to_accepted_render_receipts(
    monkeypatch, tmp_path
):
    requests = _requests()
    preflight = _write_preflight(tmp_path, requests)
    quality_payload = _quality_payload(requests)
    quality = _write_quality(tmp_path, quality_payload)
    monkeypatch.setattr(bundle_attestation, "load_native_requests", lambda _: requests)
    monkeypatch.setattr(
        bundle_attestation,
        "verify_quality_first_production_attestation",
        lambda _: quality_payload,
    )

    destination = write_production_request_bundle_attestation(
        tmp_path,
        requests_path="requests.json",
        preflight_path=preflight,
        quality_attestation_path=quality,
    )
    payload = verify_production_request_bundle_attestation(destination)

    assert payload["schema"] == "cineos-production-request-bundle-attestation/0.1"
    assert payload["normalized_request_bundle_sha256"] == _bundle_hash(requests)
    assert payload["ordered_shot_ids"] == [request.shot_id for request in requests]
    assert payload["quality_attestation_root_sha256"] == "a" * 64


def test_final_attestation_rejects_reordered_accepted_renders(monkeypatch, tmp_path):
    requests = _requests()
    preflight = _write_preflight(tmp_path, requests)
    quality_payload = _quality_payload(requests, order=(0, 2, 1, 3, 4))
    quality = _write_quality(tmp_path, quality_payload)
    monkeypatch.setattr(bundle_attestation, "load_native_requests", lambda _: requests)
    monkeypatch.setattr(
        bundle_attestation,
        "verify_quality_first_production_attestation",
        lambda _: quality_payload,
    )

    with pytest.raises(
        ProductionRequestBundleAttestationError,
        match="accepted render order does not match",
    ):
        write_production_request_bundle_attestation(
            tmp_path,
            requests_path="requests.json",
            preflight_path=preflight,
            quality_attestation_path=quality,
        )


def test_verifier_rejects_preflight_sidecar_substitution(monkeypatch, tmp_path):
    requests = _requests()
    preflight = _write_preflight(tmp_path, requests)
    quality_payload = _quality_payload(requests)
    quality = _write_quality(tmp_path, quality_payload)
    monkeypatch.setattr(bundle_attestation, "load_native_requests", lambda _: requests)
    monkeypatch.setattr(
        bundle_attestation,
        "verify_quality_first_production_attestation",
        lambda _: quality_payload,
    )
    destination = write_production_request_bundle_attestation(
        tmp_path,
        requests_path="requests.json",
        preflight_path=preflight,
        quality_attestation_path=quality,
    )

    preflight.write_text(preflight.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(
        ProductionRequestBundleAttestationError,
        match="preflight digest does not match",
    ):
        verify_production_request_bundle_attestation(destination)
