from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from cineos.film.benchmark_film_assembly import build_production_shot_evidence
from cineos.film.exceptions import AssemblyError


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _score(metrics: dict[str, float]) -> float:
    core = (
        0.32 * metrics["identity_similarity"]
        + 0.30 * metrics["temporal_consistency"]
        + 0.20 * metrics["artifact_integrity"]
        + 0.18 * metrics["motion_quality"]
    )
    optional_names = (
        "multi_character_interaction_quality",
        "anatomy_quality",
        "locomotion_quality",
        "object_interaction_quality",
        "camera_motion_quality",
        "lighting_transition_quality",
        "physics_plausibility",
        "dialogue_lip_sync",
    )
    optional = [metrics[name] for name in optional_names if name in metrics]
    if not optional:
        return core
    return 0.85 * core + 0.15 * (sum(optional) / len(optional))


def _policy() -> dict[str, float | str]:
    return {
        "schema": "cineos-sequence-quality-policy/0.3",
        "identity_floor": 0.78,
        "temporal_floor": 0.76,
        "artifact_floor": 0.90,
        "motion_floor": 0.72,
        "overall_floor": 0.80,
        "multi_character_interaction_floor": 0.76,
        "anatomy_floor": 0.78,
        "locomotion_floor": 0.74,
        "object_interaction_floor": 0.76,
        "camera_motion_floor": 0.72,
        "lighting_transition_floor": 0.74,
        "physics_plausibility_floor": 0.74,
        "dialogue_lip_sync_floor": 0.74,
    }


def _benchmark() -> SimpleNamespace:
    receipts = []
    reports = []
    chain = hashlib.sha256()
    total_output_bytes = 0
    for index in range(5):
        shot_id = f"shot-{index}"
        request_sha = _sha(f"request-{index}")
        output_sha = _sha(f"output-{index}")
        output_bytes = 100 + index
        result = SimpleNamespace(
            shot_id=shot_id,
            output_path=f"/tmp/{shot_id}.mp4",
            request_hash=request_sha,
        )
        receipts.append(
            SimpleNamespace(
                profile_id="wan22-a14b",
                origin="external-open-pretrained:wan2.2",
                result=result,
                output_sha256=output_sha,
                output_bytes=output_bytes,
            )
        )
        chain.update(request_sha.encode("ascii"))
        chain.update(b"\0")
        chain.update(output_sha.encode("ascii"))
        chain.update(b"\n")
        total_output_bytes += output_bytes

        metrics = {
            "identity_similarity": 0.95,
            "temporal_consistency": 0.95,
            "artifact_integrity": 0.95,
            "motion_quality": 0.95,
        }
        reports.append(
            {
                "schema": "cineos-sequence-quality-report/0.3",
                "accepted": True,
                "decision": "accept",
                "score": _score(metrics),
                "metrics": metrics,
                "failed_metrics": [],
                "directives": [],
                "required_challenge_metrics": {},
                "policy": _policy(),
                "production_measurement_evidence": True,
                "shot_id": shot_id,
                "output_sha256": output_sha,
                "measurement": {
                    "schema": "cineos-sequence-quality-measurement/0.1",
                    "observer_id": "production-observer",
                    "artifact_sha256": output_sha,
                    "observer_attested": True,
                    "measurement_attested": True,
                },
            }
        )

    return SimpleNamespace(
        evidence_tier="production-gpu-quality-gated",
        production_gpu_evidence=True,
        production_quality_evidence=True,
        profile_id="wan22-a14b",
        origin="external-open-pretrained:wan2.2",
        shot_receipts=tuple(receipts),
        quality_reports=tuple(reports),
        chain_sha256=chain.hexdigest(),
        total_output_bytes=total_output_bytes,
    )


def test_release_accepts_policy_consistent_quality_reports() -> None:
    evidence = build_production_shot_evidence(_benchmark())
    assert len(evidence) == 5


def test_release_rejects_accept_label_when_core_metric_fails_policy() -> None:
    benchmark = _benchmark()
    report = benchmark.quality_reports[0]
    report["metrics"]["identity_similarity"] = 0.50
    report["score"] = _score(report["metrics"])

    with pytest.raises(AssemblyError, match="below its recorded policy floor"):
        build_production_shot_evidence(benchmark)


def test_release_rejects_declared_score_not_derived_from_metrics() -> None:
    benchmark = _benchmark()
    benchmark.quality_reports[0]["score"] = 0.99

    with pytest.raises(AssemblyError, match="score does not match measured metrics"):
        build_production_shot_evidence(benchmark)


def test_release_rejects_accept_label_when_challenge_metric_fails_policy() -> None:
    benchmark = _benchmark()
    report = benchmark.quality_reports[0]
    report["metrics"]["anatomy_quality"] = 0.50
    report["required_challenge_metrics"] = {"anatomy_quality": "hands_anatomy"}
    report["score"] = _score(report["metrics"])

    with pytest.raises(
        AssemblyError, match="anatomy_quality.*below its recorded policy floor"
    ):
        build_production_shot_evidence(benchmark)


def test_release_rejects_accept_label_when_overall_score_fails_policy() -> None:
    benchmark = _benchmark()
    benchmark.quality_reports[0]["policy"]["overall_floor"] = 0.99

    with pytest.raises(AssemblyError, match="accepted QC score is below"):
        build_production_shot_evidence(benchmark)


def test_release_rejects_accepted_report_with_rejection_evidence() -> None:
    benchmark = _benchmark()
    benchmark.quality_reports[0]["failed_metrics"] = ["identity_similarity"]

    with pytest.raises(AssemblyError, match="retains rejection evidence"):
        build_production_shot_evidence(benchmark)


def test_release_rejects_unversioned_or_substituted_policy() -> None:
    benchmark = _benchmark()
    benchmark.quality_reports[0]["policy"]["schema"] = "cineos-sequence-quality-policy/0.2"

    with pytest.raises(AssemblyError, match="unsupported QC policy schema"):
        build_production_shot_evidence(benchmark)
