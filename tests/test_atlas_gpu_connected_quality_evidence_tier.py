from types import SimpleNamespace

from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkReceipt


def _gpu_receipt(index: int):
    output_sha256 = f"{index + 1:064x}"
    return SimpleNamespace(
        output_sha256=output_sha256,
        result=SimpleNamespace(
            scene_id="scene-quality-tier",
            shot_id=f"shot-{index}",
            request_hash=f"request-{index}",
        ),
        runtime_provenance={
            "schema": "cineos-gpu-runtime-provenance/0.1",
            "runtime_mode": "default",
            "production_default_runtime": True,
            "cuda_device": "cuda:0",
        },
    )


def _benchmark(*, quality_reports=()):
    return GPUConnectedBenchmarkReceipt(
        benchmark_id="quality-tier",
        profile_id="foundation-profile",
        origin="external_pretrained_foundation",
        shot_receipts=tuple(_gpu_receipt(index) for index in range(5)),
        chain_sha256="a" * 64,
        total_output_bytes=100,
        elapsed_seconds=1.0,
        manifest_path="quality-tier.gpu-benchmark.json",
        quality_reports=tuple(quality_reports),
    )


def _measured_report(index: int):
    output_sha256 = f"{index + 1:064x}"
    return {
        "accepted": True,
        "score": 0.95,
        "production_measurement_evidence": True,
        "output_sha256": output_sha256,
        "scene_id": "scene-quality-tier",
        "shot_id": f"shot-{index}",
        "effective_request_hash": f"request-{index}",
        "measurement": {
            "schema": "cineos-sequence-quality-measurement/0.1",
            "observer_id": "cineos-artifact-video-observer/0.1",
            "artifact_sha256": output_sha256,
        },
    }


def test_generic_quality_reports_do_not_claim_production_quality_evidence():
    report = {
        "accepted": True,
        "score": 0.95,
        "metrics": {
            "identity_similarity": 0.95,
            "temporal_consistency": 0.95,
            "artifact_integrity": 1.0,
            "motion_quality": 0.9,
        },
    }
    receipt = _benchmark(quality_reports=[report] * 5)

    assert receipt.production_gpu_evidence is True
    assert receipt.production_quality_evidence is False
    assert receipt.evidence_tier == "production-gpu-execution"
    assert receipt.to_dict()["production_quality_evidence"] is False


def test_every_exactly_bound_report_promotes_production_quality_evidence():
    reports = [_measured_report(index) for index in range(5)]
    receipt = _benchmark(quality_reports=reports)

    assert receipt.production_gpu_evidence is True
    assert receipt.production_quality_evidence is True
    assert receipt.evidence_tier == "production-gpu-quality-gated"
    assert receipt.to_dict()["production_quality_evidence"] is True


def test_one_unbound_report_fails_closed_for_entire_connected_benchmark():
    reports = [_measured_report(index) for index in range(5)]
    reports[2] = {"accepted": True, "score": 0.95}
    receipt = _benchmark(quality_reports=reports)

    assert receipt.production_quality_evidence is False
    assert receipt.evidence_tier == "production-gpu-execution"


def test_one_swapped_artifact_hash_fails_closed_for_entire_connected_benchmark():
    reports = [_measured_report(index) for index in range(5)]
    reports[3]["measurement"]["artifact_sha256"] = reports[2]["output_sha256"]
    receipt = _benchmark(quality_reports=reports)

    assert receipt.production_quality_evidence is False
    assert receipt.evidence_tier == "production-gpu-execution"
