from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from cineos.atlas.production_connected_evidence import (
    ProductionConnectedEvidenceError,
    production_connected_evidence,
    validate_production_connected_evidence,
)
from cineos.atlas.production_continuity_diffusers import VISUAL_CONTINUITY_SCHEMA


def _sha(index: int) -> str:
    return f"{index:064x}"


def _benchmark(*, shot_count: int = 5) -> GPUConnectedBenchmarkReceipt:
    receipts = []
    reports = []
    for index in range(shot_count):
        output_sha = _sha(index + 1)
        request_hash = f"request-{index + 1}"
        provenance = {
            "schema": VISUAL_CONTINUITY_SCHEMA,
            "scene_id": "scene-1",
            "shot_id": f"shot-{index + 1}",
            "current_artifact_sha256": output_sha,
            "current_request_hash": request_hash,
            "in_memory_terminal_frame": index > 0,
        }
        if index == 0:
            provenance.update(
                {
                    "mode": "approved_reference_root",
                    "previous_scene_id": None,
                    "previous_shot_id": None,
                    "predecessor_artifact_sha256": None,
                    "predecessor_request_hash": None,
                }
            )
        else:
            provenance.update(
                {
                    "mode": "predecessor_terminal_frame_lineage",
                    "previous_scene_id": "scene-1",
                    "previous_shot_id": f"shot-{index}",
                    "predecessor_artifact_sha256": _sha(index),
                    "predecessor_request_hash": f"request-{index}",
                }
            )

        result = SimpleNamespace(
            scene_id="scene-1",
            shot_id=f"shot-{index + 1}",
            request_hash=request_hash,
            conditioning_provenance=provenance,
        )
        receipts.append(
            SimpleNamespace(
                result=result,
                output_sha256=output_sha,
                runtime_provenance={
                    "schema": "cineos-gpu-runtime-provenance/0.1",
                    "runtime_mode": "default",
                    "production_default_runtime": True,
                    "cuda_device": "cuda:0",
                },
            )
        )
        reports.append(
            {
                "accepted": True,
                "production_measurement_evidence": True,
                "scene_id": "scene-1",
                "shot_id": f"shot-{index + 1}",
                "effective_request_hash": request_hash,
                "output_sha256": output_sha,
                "measurement": {
                    "schema": "cineos-sequence-quality-measurement/0.1",
                    "observer_id": "measured-video-qc-v1",
                    "artifact_sha256": output_sha,
                },
            }
        )

    return GPUConnectedBenchmarkReceipt(
        benchmark_id="connected-production-test",
        profile_id="wan2.2-ti2v-5b",
        origin="external_pretrained_foundation",
        shot_receipts=tuple(receipts),
        chain_sha256=_sha(99),
        total_output_bytes=1_000,
        elapsed_seconds=12.0,
        manifest_path="benchmark.json",
        quality_reports=tuple(reports),
    )


def test_accepts_only_unified_runtime_quality_and_continuity_evidence() -> None:
    benchmark = _benchmark()

    evidence = validate_production_connected_evidence(benchmark)

    assert evidence.accepted is True
    assert evidence.runtime_valid is True
    assert evidence.quality_valid is True
    assert evidence.continuity_valid is True
    assert evidence.shot_count == 5
    assert len(evidence.continuity_provenance) == 5
    assert evidence.to_dict()["accepted"] is True
    assert production_connected_evidence(benchmark) is True


@pytest.mark.parametrize("shot_count", [0, 1, 4, 11])
def test_rejects_sequences_outside_competitive_5_to_10_shot_range(
    shot_count: int,
) -> None:
    benchmark = _benchmark(shot_count=shot_count)

    with pytest.raises(ProductionConnectedEvidenceError, match="between 5 and 10"):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_non_default_gpu_runtime_provenance() -> None:
    benchmark = _benchmark()
    benchmark.shot_receipts[2].runtime_provenance["runtime_mode"] = "injected"

    with pytest.raises(ProductionConnectedEvidenceError, match="default CUDA runtime"):
        validate_production_connected_evidence(benchmark)


def test_rejects_quality_report_bound_to_different_artifact() -> None:
    benchmark = _benchmark()
    benchmark.quality_reports[3]["measurement"]["artifact_sha256"] = _sha(777)

    with pytest.raises(ProductionConnectedEvidenceError, match="artifact-bound QC"):
        validate_production_connected_evidence(benchmark)


def test_rejects_substituted_predecessor_continuity_artifact() -> None:
    benchmark = _benchmark()
    benchmark.shot_receipts[4].result.conditioning_provenance[
        "predecessor_artifact_sha256"
    ] = _sha(888)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="failed visual continuity validation",
    ):
        validate_production_connected_evidence(benchmark)


def test_rejects_non_benchmark_objects() -> None:
    with pytest.raises(TypeError, match="GPUConnectedBenchmarkReceipt"):
        validate_production_connected_evidence(SimpleNamespace())

    assert production_connected_evidence(SimpleNamespace()) is False
