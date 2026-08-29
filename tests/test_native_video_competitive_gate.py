from dataclasses import replace
from pathlib import Path

from cineos.native_video.competitive_benchmark import (
    CompetitiveBenchmarkReport,
    ShotBenchmarkResult,
    default_connected_cases,
)
from cineos.native_video.competitive_gate import evaluate_competitive_acceptance


def _passing_report(tmp_path: Path) -> CompetitiveBenchmarkReport:
    shots = []
    for case in default_connected_cases():
        artifact = tmp_path / f"{case.shot_id}.mp4"
        artifact.write_bytes(b"measured-real-artifact")
        shots.append(
            ShotBenchmarkResult(
                shot_id=case.shot_id,
                challenge_tags=case.challenge_tags,
                request_hash=f"hash-{case.shot_id}",
                output_path=str(artifact),
                artifact_bytes=artifact.stat().st_size,
                frame_count=32,
                execution_passed=True,
                quality_evaluated=True,
                quality_passed=True,
                quality_metrics={"visual_quality": 0.92},
                notes=("measured",),
            )
        )
    return CompetitiveBenchmarkReport(
        scene_id="connected-scene",
        foundation={
            "model_id": "open/foundation",
            "revision": "pinned-revision",
            "license_id": "declared-license",
            "provenance_declared": True,
        },
        shots=tuple(shots),
    )


def test_full_measured_connected_suite_can_pass_competitive_gate(tmp_path):
    verdict = evaluate_competitive_acceptance(_passing_report(tmp_path))

    assert verdict.passed is True
    assert verdict.reasons == ()
    assert verdict.missing_challenges == frozenset()
    assert verdict.evaluated_metric_names == frozenset({"visual_quality"})
    assert verdict.to_dict()["schema"] == "cineos-competitive-acceptance/0.1"


def test_reduced_suite_cannot_be_mislabeled_as_competitive(tmp_path):
    report = _passing_report(tmp_path)
    report = CompetitiveBenchmarkReport(
        scene_id=report.scene_id,
        foundation=report.foundation,
        shots=report.shots[:2],
    )

    verdict = evaluate_competitive_acceptance(report)

    assert verdict.passed is False
    assert any("at least 10 connected shots" in reason for reason in verdict.reasons)
    assert verdict.missing_challenges


def test_undeclared_or_unlicensed_foundation_fails_closed(tmp_path):
    report = _passing_report(tmp_path)
    report = CompetitiveBenchmarkReport(
        scene_id=report.scene_id,
        foundation={"model_id": "unknown", "provenance_declared": False},
        shots=report.shots,
    )

    verdict = evaluate_competitive_acceptance(report)

    assert verdict.passed is False
    assert "renderer foundation provenance is not declared" in verdict.reasons
    assert "renderer foundation model_id is missing or unknown" in verdict.reasons
    assert "renderer foundation revision is not declared" in verdict.reasons
    assert "renderer foundation license_id is not declared" in verdict.reasons


def test_empty_metric_evidence_fails_even_when_boolean_quality_passed(tmp_path):
    report = _passing_report(tmp_path)
    first = replace(report.shots[0], quality_metrics={})
    report = CompetitiveBenchmarkReport(
        scene_id=report.scene_id,
        foundation=report.foundation,
        shots=(first, *report.shots[1:]),
    )

    verdict = evaluate_competitive_acceptance(report)

    assert verdict.passed is False
    assert any("returned no numeric metrics" in reason for reason in verdict.reasons)
