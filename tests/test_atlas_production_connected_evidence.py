from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from cineos.atlas.production_connected_evidence import (
    ProductionConnectedEvidenceError,
    production_connected_evidence,
    validate_production_connected_evidence,
)
from cineos.atlas.production_continuity_diffusers import VISUAL_CONTINUITY_SCHEMA
from cineos.atlas.seedance_style_challenge import REQUIRED_CHALLENGES
from cineos.atlas.transition_quality import TRANSITION_QUALITY_SCHEMA


def _sha(index: int) -> str:
    return f"{index:064x}"


def _receipt_chain_sha256(receipts) -> str:
    digest = hashlib.sha256()
    for receipt in receipts:
        digest.update(receipt.result.request_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(receipt.output_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _challenge_contract(shot_count: int) -> dict[str, object]:
    usable_count = max(1, shot_count)
    payload: dict[str, object] = {
        "schema": "cineos-seedance-style-challenge-coverage/0.1",
        "required_challenges": list(REQUIRED_CHALLENGES),
        "complete": True,
        "missing": [],
        "challenge_to_shots": {
            challenge: [f"scene-1/shot-{(index % usable_count) + 1}"]
            for index, challenge in enumerate(REQUIRED_CHALLENGES)
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _resign_challenge_contract(contract: dict[str, object]) -> None:
    contract.pop("contract_sha256", None)
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["contract_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _benchmark(tmp_path: Path, *, shot_count: int = 5) -> GPUConnectedBenchmarkReceipt:
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

    transitions = [
        {
            "schema": TRANSITION_QUALITY_SCHEMA,
            "production_measurement_evidence": True,
            "accepted": True,
            "observer_id": "measured-transition-qc-v1",
            "measured_sample_count": 4,
            "previous_output_sha256": _sha(index + 1),
            "current_output_sha256": _sha(index + 2),
            "previous_scene_id": "scene-1",
            "previous_shot_id": f"shot-{index + 1}",
            "current_scene_id": "scene-1",
            "current_shot_id": f"shot-{index + 2}",
            "metrics": {
                "visual_seam_similarity": 0.91,
                "motion_boundary_consistency": 0.88,
            },
            "failed_metrics": [],
            "directives": [],
        }
        for index in range(max(0, shot_count - 1))
    ]
    chain_sha256 = _receipt_chain_sha256(receipts)
    manifest = tmp_path / f"benchmark-{shot_count}.json"
    manifest.write_text(
        json.dumps(
            {
                "chain_sha256": chain_sha256,
                "quality_retry_gate": {
                    "transition_gate_applied": True,
                    "accepted_transition_count": len(transitions),
                    "accepted_transitions": transitions,
                },
                "competitive_challenge_contract": _challenge_contract(shot_count),
            }
        ),
        encoding="utf-8",
    )

    return GPUConnectedBenchmarkReceipt(
        benchmark_id="connected-production-test",
        profile_id="wan2.2-ti2v-5b",
        origin="external_pretrained_foundation",
        shot_receipts=tuple(receipts),
        chain_sha256=chain_sha256,
        total_output_bytes=1_000,
        elapsed_seconds=12.0,
        manifest_path=str(manifest),
        quality_reports=tuple(reports),
    )


def _manifest_payload(benchmark: GPUConnectedBenchmarkReceipt) -> dict[str, object]:
    return json.loads(Path(benchmark.manifest_path).read_text(encoding="utf-8"))


def _write_manifest(
    benchmark: GPUConnectedBenchmarkReceipt, payload: dict[str, object]
) -> None:
    Path(benchmark.manifest_path).write_text(json.dumps(payload), encoding="utf-8")


def _transitions(payload: dict[str, object]) -> list[dict[str, object]]:
    gate = payload["quality_retry_gate"]
    assert isinstance(gate, dict)
    transitions = gate["accepted_transitions"]
    assert isinstance(transitions, list)
    return transitions


def _challenge(payload: dict[str, object]) -> dict[str, object]:
    contract = payload["competitive_challenge_contract"]
    assert isinstance(contract, dict)
    return contract


def test_accepts_only_unified_runtime_quality_continuity_and_challenge_evidence(
    tmp_path,
) -> None:
    benchmark = _benchmark(tmp_path)

    evidence = validate_production_connected_evidence(benchmark)

    assert evidence.accepted is True
    assert evidence.runtime_valid is True
    assert evidence.quality_valid is True
    assert evidence.continuity_valid is True
    assert evidence.transition_quality_valid is True
    assert evidence.challenge_coverage_valid is True
    assert evidence.shot_count == 5
    assert len(evidence.continuity_provenance) == 5
    assert evidence.to_dict()["schema"] == "cineos-production-connected-evidence/0.6"
    assert evidence.to_dict()["accepted"] is True
    assert evidence.to_dict()["transition_quality_valid"] is True
    assert evidence.to_dict()["challenge_coverage_valid"] is True
    assert production_connected_evidence(benchmark) is True


@pytest.mark.parametrize("shot_count", [0, 1, 4, 11])
def test_rejects_sequences_outside_competitive_5_to_10_shot_range(
    shot_count: int,
    tmp_path,
) -> None:
    benchmark = _benchmark(tmp_path, shot_count=shot_count)

    with pytest.raises(ProductionConnectedEvidenceError, match="between 5 and 10"):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_non_default_gpu_runtime_provenance(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    benchmark.shot_receipts[2].runtime_provenance["runtime_mode"] = "injected"

    with pytest.raises(ProductionConnectedEvidenceError, match="default CUDA runtime"):
        validate_production_connected_evidence(benchmark)


def test_rejects_quality_report_bound_to_different_artifact(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    benchmark.quality_reports[3]["measurement"]["artifact_sha256"] = _sha(777)

    with pytest.raises(ProductionConnectedEvidenceError, match="artifact-bound QC"):
        validate_production_connected_evidence(benchmark)


def test_rejects_substituted_predecessor_continuity_artifact(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    benchmark.shot_receipts[4].result.conditioning_provenance[
        "predecessor_artifact_sha256"
    ] = _sha(888)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="failed visual continuity validation",
    ):
        validate_production_connected_evidence(benchmark)


def test_rejects_forged_chain_even_when_manifest_matches_receipt(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    forged_chain = _sha(999)
    payload = _manifest_payload(benchmark)
    payload["chain_sha256"] = forged_chain
    _write_manifest(benchmark, payload)
    benchmark = replace(benchmark, chain_sha256=forged_chain)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="chain hash does not match render receipts",
    ):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_missing_measured_transition_gate(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    payload["quality_retry_gate"] = {"transition_gate_applied": False}
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="measured transition QC",
    ):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_incomplete_transition_boundary_coverage(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    gate = payload["quality_retry_gate"]
    assert isinstance(gate, dict)
    gate["accepted_transition_count"] = 3
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="cover every shot boundary",
    ):
        validate_production_connected_evidence(benchmark)


def test_rejects_transition_bound_to_different_artifact(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    _transitions(payload)[2]["current_output_sha256"] = _sha(777)
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="current artifact hash mismatch",
    ):
        validate_production_connected_evidence(benchmark)


def test_rejects_unmeasured_transition_evidence(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    _transitions(payload)[1]["production_measurement_evidence"] = False
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="not measured evidence",
    ):
        validate_production_connected_evidence(benchmark)


def test_rejects_transition_with_missing_measurement_schema(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    _transitions(payload)[0].pop("schema")
    _write_manifest(benchmark, payload)

    with pytest.raises(ProductionConnectedEvidenceError, match="unsupported schema"):
        validate_production_connected_evidence(benchmark)


def test_rejects_transition_with_missing_metrics(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    _transitions(payload)[0].pop("metrics")
    _write_manifest(benchmark, payload)

    with pytest.raises(ProductionConnectedEvidenceError, match="measured metrics"):
        validate_production_connected_evidence(benchmark)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1]
)
def test_rejects_non_finite_or_out_of_range_transition_metrics(value, tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    transition = _transitions(payload)[0]
    metrics = transition["metrics"]
    assert isinstance(metrics, dict)
    metrics["visual_seam_similarity"] = value
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="must be finite and between 0 and 1",
    ):
        validate_production_connected_evidence(benchmark)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("visual_seam_similarity", 0.779, "visual_seam_similarity"),
        ("motion_boundary_consistency", 0.719, "motion_boundary_consistency"),
    ],
)
def test_rejects_accepted_transition_below_production_floor(
    metric: str,
    value: float,
    message: str,
    tmp_path,
) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    transition = _transitions(payload)[0]
    metrics = transition["metrics"]
    assert isinstance(metrics, dict)
    metrics[metric] = value
    _write_manifest(benchmark, payload)

    with pytest.raises(ProductionConnectedEvidenceError, match=message):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_accepts_transition_exactly_at_production_floors(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    transition = _transitions(payload)[0]
    metrics = transition["metrics"]
    assert isinstance(metrics, dict)
    metrics["visual_seam_similarity"] = 0.78
    metrics["motion_boundary_consistency"] = 0.72
    _write_manifest(benchmark, payload)

    assert validate_production_connected_evidence(benchmark).accepted is True


def test_rejects_accepted_transition_that_still_lists_failed_metrics(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    _transitions(payload)[0]["failed_metrics"] = ["motion_boundary_consistency"]
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="accepted report contains failed metrics",
    ):
        validate_production_connected_evidence(benchmark)


def test_rejects_accepted_transition_with_unresolved_rerender_directives(
    tmp_path,
) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    _transitions(payload)[0]["directives"] = [
        "preserve physically coherent motion across the shot boundary"
    ]
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="unresolved rerender directives",
    ):
        validate_production_connected_evidence(benchmark)


def test_rejects_missing_competitive_challenge_contract(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    payload.pop("competitive_challenge_contract")
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="requires competitive challenge coverage",
    ):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_tampered_competitive_challenge_contract(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    contract = _challenge(payload)
    mapping = contract["challenge_to_shots"]
    assert isinstance(mapping, dict)
    mapping["dialogue"] = ["scene-1/shot-4"]
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="contract hash does not match its contents",
    ):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_resigned_challenge_contract_referencing_unrendered_shot(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    contract = _challenge(payload)
    mapping = contract["challenge_to_shots"]
    assert isinstance(mapping, dict)
    mapping["dialogue"] = ["scene-1/shot-999"]
    _resign_challenge_contract(contract)
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="references an unrendered shot",
    ):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_resigned_challenge_contract_with_uncovered_hard_case(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    payload = _manifest_payload(benchmark)
    contract = _challenge(payload)
    mapping = contract["challenge_to_shots"]
    assert isinstance(mapping, dict)
    mapping["physics"] = []
    _resign_challenge_contract(contract)
    _write_manifest(benchmark, payload)

    with pytest.raises(
        ProductionConnectedEvidenceError,
        match="physics has no covered shot",
    ):
        validate_production_connected_evidence(benchmark)

    assert production_connected_evidence(benchmark) is False


def test_rejects_non_benchmark_objects() -> None:
    with pytest.raises(TypeError, match="GPUConnectedBenchmarkReceipt"):
        validate_production_connected_evidence(SimpleNamespace())

    assert production_connected_evidence(SimpleNamespace()) is False
