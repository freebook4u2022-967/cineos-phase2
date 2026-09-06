from types import SimpleNamespace

import pytest

from cineos.atlas import gpu_benchmark_cli as cli
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.siglip2_video_scorer import SigLIP2VideoScorerError


def _quality_gated_receipt(*, production_quality_evidence: bool = True):
    return SimpleNamespace(
        production_gpu_evidence=True,
        production_quality_evidence=production_quality_evidence,
        evidence_tier=(
            "production-gpu-quality-gated"
            if production_quality_evidence
            else "production-gpu"
        ),
    )


def _connected_requests() -> tuple[NativeShotRequest, ...]:
    requests = []
    for index in range(5):
        request = NativeShotRequest(
            shot_id=f"shot-{index}",
            scene_id="scene-quality-gate",
            camera={"movement": "tracking"},
            characters=[{"character_id": "lead"}],
            environment={"location": "street"},
            wardrobe=[],
            props=[],
            continuity={
                "previous_shot_id": None if index == 0 else f"shot-{index - 1}"
            },
            performance={"action": "walk"},
            approved_reference_ids=["lead-approved-reference"],
            deterministic_seed=7000 + index,
            renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        )
        request.refresh_hash()
        requests.append(request)
    return tuple(requests)


def test_production_cli_routes_real_render_through_quality_retry_gate(
    monkeypatch, tmp_path
):
    evaluator = object()
    observed = {}
    requests = _connected_requests()

    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: evaluator,
    )

    def fake_quality_retry(benchmark_id, requests, profile, **kwargs):
        observed["benchmark_id"] = benchmark_id
        observed["requests"] = requests
        observed["profile"] = profile
        observed.update(kwargs)
        return _quality_gated_receipt()

    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        fake_quality_retry,
    )

    receipt = cli.run_production_benchmark(
        "bench-quality",
        requests,
        output_dir=tmp_path,
        reference_manifest="approved-references.json",
    )

    assert receipt.production_quality_evidence is True
    assert observed["benchmark_id"] == "bench-quality"
    assert observed["requests"] == requests
    assert observed["quality_evaluator"] is evaluator
    assert observed["reference_manifest"] == "approved-references.json"
    assert observed["profile"] is cli.WAN22_TI2V_5B_PROFILE


def test_production_cli_rejects_receipt_without_production_quality_evidence(
    monkeypatch, tmp_path
):
    requests = _connected_requests()
    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        lambda requests, reference_manifest: object(),
    )
    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        lambda *args, **kwargs: _quality_gated_receipt(
            production_quality_evidence=False
        ),
    )

    with pytest.raises(
        cli.GPUProductionBenchmarkCLIError,
        match="artifact-bound production QC evidence",
    ):
        cli.run_production_benchmark(
            "bench-no-qc",
            requests,
            output_dir=tmp_path,
            reference_manifest="approved-references.json",
        )


def test_production_cli_rejects_direct_request_mutated_after_hash_before_qc(
    monkeypatch, tmp_path
):
    requests = list(_connected_requests())
    requests[2].performance["action"] = "run"
    qc_initialized = False

    def fail_if_qc_initialized(requests, reference_manifest):
        nonlocal qc_initialized
        qc_initialized = True
        raise AssertionError("QC/model acquisition must not start for stale requests")

    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        fail_if_qc_initialized,
    )

    with pytest.raises(
        cli.GPUProductionBenchmarkCLIError,
        match="content_hash is stale or does not match its live payload",
    ):
        cli.run_production_benchmark(
            "bench-stale-direct-request",
            tuple(requests),
            output_dir=tmp_path,
            reference_manifest="approved-references.json",
        )

    assert qc_initialized is False


def test_production_cli_rejects_unhashed_direct_request_before_qc(
    monkeypatch, tmp_path
):
    requests = list(_connected_requests())
    requests[1].content_hash = ""
    qc_initialized = False

    def fail_if_qc_initialized(requests, reference_manifest):
        nonlocal qc_initialized
        qc_initialized = True
        raise AssertionError("QC/model acquisition must not start for unhashed requests")

    monkeypatch.setattr(
        cli,
        "_production_quality_evaluator",
        fail_if_qc_initialized,
    )

    with pytest.raises(
        cli.GPUProductionBenchmarkCLIError,
        match="requires a canonical 64-character content_hash",
    ):
        cli.run_production_benchmark(
            "bench-unhashed-direct-request",
            tuple(requests),
            output_dir=tmp_path,
            reference_manifest="approved-references.json",
        )

    assert qc_initialized is False


def test_production_quality_evaluator_fails_closed_when_pinned_qc_unavailable(
    monkeypatch,
):
    loader = object()
    monkeypatch.setattr(
        cli,
        "_production_reference_loader",
        lambda requests, reference_manifest: loader,
    )
    monkeypatch.setattr(
        cli,
        "_production_multi_reference_adapter",
        lambda requests: None,
    )

    class MissingPinnedQC:
        def __init__(self, reference_loader, *, device):
            assert reference_loader is loader
            assert device == "cuda"
            raise SigLIP2VideoScorerError("pinned snapshot unavailable")

    monkeypatch.setattr(cli, "SigLIP2FeatureVideoScorer", MissingPinnedQC)

    with pytest.raises(
        cli.GPUProductionBenchmarkCLIError,
        match="cannot initialize pinned production visual QC",
    ):
        cli._production_quality_evaluator((), "approved-references.json")
