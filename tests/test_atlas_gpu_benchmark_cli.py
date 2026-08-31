import json
from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_benchmark_cli import (
    GPUProductionBenchmarkCLIError,
    load_native_requests,
    run_production_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest


def _request(index: int) -> NativeShotRequest:
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
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=4000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
    )
    request.refresh_hash()
    return request


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


def test_production_runner_fails_closed_without_default_gpu_evidence(
    monkeypatch, tmp_path
):
    fake_receipt = SimpleNamespace(production_gpu_evidence=False)

    monkeypatch.setattr(
        "cineos.atlas.gpu_benchmark_cli.run_connected_gpu_benchmark",
        lambda *args, **kwargs: fake_receipt,
    )

    with pytest.raises(
        GPUProductionBenchmarkCLIError, match="without default production CUDA evidence"
    ):
        run_production_benchmark(
            "production-evidence",
            [_request(index) for index in range(5)],
            output_dir=tmp_path,
        )


def test_production_runner_returns_verified_default_gpu_receipt(monkeypatch, tmp_path):
    fake_receipt = SimpleNamespace(production_gpu_evidence=True)

    monkeypatch.setattr(
        "cineos.atlas.gpu_benchmark_cli.run_connected_gpu_benchmark",
        lambda *args, **kwargs: fake_receipt,
    )

    receipt = run_production_benchmark(
        "production-evidence",
        [_request(index) for index in range(5)],
        output_dir=tmp_path,
    )

    assert receipt is fake_receipt
