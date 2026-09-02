import hashlib
import json
from types import SimpleNamespace

import pytest

from cineos.atlas import gpu_benchmark_cli as cli
from cineos.atlas.gpu_benchmark_cli import (
    GPUProductionBenchmarkCLIError,
    load_native_requests,
    run_production_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_multi_reference import ProductionReferenceBoardAdapter


def _request(index: int, reference_ids=None) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-cli",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=list(reference_ids or ["lead-approved-reference"]),
        deterministic_seed=4000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
    )
    request.refresh_hash()
    return request


def _reference_manifest(tmp_path, reference_ids=("lead-approved-reference",)):
    references = []
    for index, reference_id in enumerate(reference_ids):
        image = tmp_path / f"reference-{index}.png"
        image.write_bytes(f"approved-image-{index}".encode())
        references.append(
            {
                "reference_id": reference_id,
                "path": image.name,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "cineos-approved-reference-manifest/0.1",
                "references": references,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _quality_receipt(*, gpu=True, quality=True):
    return SimpleNamespace(
        production_gpu_evidence=gpu,
        production_quality_evidence=quality,
        evidence_tier=(
            "production-gpu-quality-gated" if gpu and quality else "research"
        ),
    )


def test_load_native_requests_recomputes_and_preserves_valid_hashes(tmp_path):
    source = tmp_path / "requests.json"
    requests = [_request(index) for index in range(5)]
    source.write_text(
        json.dumps({"shots": [request.to_dict() for request in requests]}),
        encoding="utf-8",
    )

    loaded = load_native_requests(source)

    assert len(loaded) == 5
    assert [request.content_hash for request in loaded] == [
        request.content_hash for request in requests
    ]


def test_load_native_requests_rejects_stale_hash(tmp_path):
    source = tmp_path / "requests.json"
    payload = _request(0).to_dict()
    payload["camera"]["movement"] = "changed-after-hash"
    source.write_text(json.dumps([payload]), encoding="utf-8")

    with pytest.raises(GPUProductionBenchmarkCLIError, match="content_hash is stale"):
        load_native_requests(source)


def test_load_native_requests_rejects_non_array_manifest(tmp_path):
    source = tmp_path / "requests.json"
    source.write_text(json.dumps({"not_shots": []}), encoding="utf-8")

    with pytest.raises(GPUProductionBenchmarkCLIError, match="shots array"):
        load_native_requests(source)


def test_production_runner_requires_reference_manifest_before_qc_model_load(tmp_path):
    with pytest.raises(GPUProductionBenchmarkCLIError, match="reference-manifest"):
        run_production_benchmark(
            "production-evidence",
            [_request(index) for index in range(5)],
            output_dir=tmp_path / "renders",
        )


def test_production_runner_rejects_manifest_missing_requested_identity(tmp_path):
    manifest = _reference_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["references"][0]["reference_id"] = "somebody-else"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(GPUProductionBenchmarkCLIError, match="lead-approved-reference"):
        run_production_benchmark(
            "production-evidence",
            [_request(index) for index in range(5)],
            output_dir=tmp_path / "renders",
            reference_manifest=manifest,
        )


def test_production_runner_routes_through_quality_retry_boundary(monkeypatch, tmp_path):
    evaluator = object()
    fake_receipt = _quality_receipt()
    captured = {}
    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: evaluator,
    )

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_receipt

    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        fake_run,
    )
    requests = [_request(index) for index in range(5)]
    manifest = _reference_manifest(tmp_path)

    receipt = run_production_benchmark(
        "production-evidence",
        requests,
        output_dir=tmp_path / "renders",
        reference_manifest=manifest,
    )

    assert receipt is fake_receipt
    assert captured["args"][:2] == ("production-evidence", requests)
    assert captured["kwargs"]["quality_evaluator"] is evaluator
    assert captured["kwargs"]["reference_manifest"] == manifest
    assert captured["kwargs"]["output_dir"] == tmp_path / "renders"


def test_production_runner_enables_audited_multi_reference_adapter():
    reference_ids = ("lead-approved-reference", "partner-approved-reference")
    requests = [_request(index, reference_ids) for index in range(5)]

    adapter = cli._production_multi_reference_adapter(requests)

    assert isinstance(adapter, ProductionReferenceBoardAdapter)


def test_production_runner_rejects_more_than_four_references_before_qc_load():
    reference_ids = tuple(f"identity-{index}" for index in range(5))
    requests = [_request(index, reference_ids) for index in range(5)]

    with pytest.raises(GPUProductionBenchmarkCLIError, match="at most four"):
        cli._production_multi_reference_adapter(requests)


def test_production_runner_fails_closed_without_default_gpu_evidence(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: object(),
    )
    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        lambda *args, **kwargs: _quality_receipt(gpu=False),
    )

    with pytest.raises(
        GPUProductionBenchmarkCLIError, match="without default production CUDA evidence"
    ):
        run_production_benchmark(
            "production-evidence",
            [_request(index) for index in range(5)],
            output_dir=tmp_path,
            reference_manifest="approved-references.json",
        )


def test_production_runner_fails_closed_without_quality_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: object(),
    )
    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        lambda *args, **kwargs: _quality_receipt(quality=False),
    )

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="without artifact-bound production QC evidence",
    ):
        run_production_benchmark(
            "production-evidence",
            [_request(index) for index in range(5)],
            output_dir=tmp_path,
            reference_manifest="approved-references.json",
        )


def test_production_runner_returns_verified_quality_gated_receipt(
    monkeypatch, tmp_path
):
    fake_receipt = _quality_receipt()
    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: object(),
    )
    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        lambda *args, **kwargs: fake_receipt,
    )

    receipt = run_production_benchmark(
        "production-evidence",
        [_request(index) for index in range(5)],
        output_dir=tmp_path,
        reference_manifest="approved-references.json",
    )

    assert receipt is fake_receipt
