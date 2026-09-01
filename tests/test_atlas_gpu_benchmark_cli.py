import hashlib
import json
from types import SimpleNamespace

import pytest

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


def _install_fake_persistent_executor(monkeypatch):
    lifecycle = []

    class FakePersistentExecutor:
        def __init__(
            self,
            profile,
            *,
            output_dir,
            reference_loader=None,
            multi_reference_adapter=None,
        ):
            self.profile = profile
            self.output_dir = output_dir
            self.reference_loader = reference_loader
            self.multi_reference_adapter = multi_reference_adapter
            lifecycle.append(("init", profile, output_dir, self))

        def __enter__(self):
            lifecycle.append(("enter", self))
            return self

        def __exit__(self, exc_type, exc, traceback):
            lifecycle.append(("exit", exc_type, self))

    monkeypatch.setattr(
        "cineos.atlas.gpu_benchmark_cli.PersistentGPUFoundationExecutor",
        FakePersistentExecutor,
    )
    return lifecycle


def test_production_runner_requires_reference_manifest_before_gpu_session(
    monkeypatch, tmp_path
):
    lifecycle = _install_fake_persistent_executor(monkeypatch)

    with pytest.raises(GPUProductionBenchmarkCLIError, match="reference-manifest"):
        run_production_benchmark(
            "production-evidence",
            [_request(index) for index in range(5)],
            output_dir=tmp_path / "renders",
        )

    assert lifecycle == []


def test_production_runner_rejects_manifest_missing_requested_identity(
    monkeypatch, tmp_path
):
    lifecycle = _install_fake_persistent_executor(monkeypatch)
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

    assert lifecycle == []


def test_production_runner_uses_one_persistent_gpu_executor(monkeypatch, tmp_path):
    fake_receipt = SimpleNamespace(production_gpu_evidence=True)
    lifecycle = _install_fake_persistent_executor(monkeypatch)
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_receipt

    monkeypatch.setattr(
        "cineos.atlas.gpu_benchmark_cli.run_connected_gpu_benchmark",
        fake_run,
    )

    requests = [_request(index) for index in range(5)]
    receipt = run_production_benchmark(
        "production-evidence",
        requests,
        output_dir=tmp_path / "renders",
        reference_manifest=_reference_manifest(tmp_path),
    )

    assert receipt is fake_receipt
    assert [event[0] for event in lifecycle] == ["init", "enter", "exit"]
    executor = lifecycle[0][3]
    assert executor.reference_loader.reference_ids == ("lead-approved-reference",)
    assert executor.multi_reference_adapter is None
    assert captured["kwargs"]["shot_executor"] is executor
    assert captured["kwargs"]["output_dir"] == tmp_path / "renders"
    assert (tmp_path / "renders").is_dir()


def test_production_runner_enables_audited_multi_reference_adapter(
    monkeypatch, tmp_path
):
    fake_receipt = SimpleNamespace(production_gpu_evidence=True)
    lifecycle = _install_fake_persistent_executor(monkeypatch)
    monkeypatch.setattr(
        "cineos.atlas.gpu_benchmark_cli.run_connected_gpu_benchmark",
        lambda *args, **kwargs: fake_receipt,
    )
    reference_ids = ("lead-approved-reference", "partner-approved-reference")
    requests = [_request(index, reference_ids) for index in range(5)]

    receipt = run_production_benchmark(
        "production-multi-reference",
        requests,
        output_dir=tmp_path / "renders",
        reference_manifest=_reference_manifest(tmp_path, reference_ids),
    )

    assert receipt is fake_receipt
    executor = lifecycle[0][3]
    assert isinstance(executor.multi_reference_adapter, ProductionReferenceBoardAdapter)


def test_production_runner_rejects_more_than_four_references_before_gpu_session(
    monkeypatch, tmp_path
):
    lifecycle = _install_fake_persistent_executor(monkeypatch)
    reference_ids = tuple(f"identity-{index}" for index in range(5))
    requests = [_request(index, reference_ids) for index in range(5)]

    with pytest.raises(GPUProductionBenchmarkCLIError, match="at most four"):
        run_production_benchmark(
            "too-many-identities",
            requests,
            output_dir=tmp_path / "renders",
            reference_manifest=_reference_manifest(tmp_path, reference_ids),
        )

    assert lifecycle == []


def test_production_runner_fails_closed_without_default_gpu_evidence(
    monkeypatch, tmp_path
):
    fake_receipt = SimpleNamespace(production_gpu_evidence=False)
    _install_fake_persistent_executor(monkeypatch)

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
            reference_manifest=_reference_manifest(tmp_path),
        )


def test_production_runner_returns_verified_default_gpu_receipt(monkeypatch, tmp_path):
    fake_receipt = SimpleNamespace(production_gpu_evidence=True)
    _install_fake_persistent_executor(monkeypatch)

    monkeypatch.setattr(
        "cineos.atlas.gpu_benchmark_cli.run_connected_gpu_benchmark",
        lambda *args, **kwargs: fake_receipt,
    )

    receipt = run_production_benchmark(
        "production-evidence",
        [_request(index) for index in range(5)],
        output_dir=tmp_path,
        reference_manifest=_reference_manifest(tmp_path),
    )

    assert receipt is fake_receipt
