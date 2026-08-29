from dataclasses import replace
from pathlib import Path

from cineos.native_video.competitive_benchmark import (
    CompetitiveBenchmarkReport,
    ShotBenchmarkResult,
    default_connected_cases,
)
from cineos.native_video.competitive_gate import (
    SEEDANCE_STYLE_CHALLENGE_METRICS,
    evaluate_competitive_acceptance,
)


def _metrics_for(challenge_tags: tuple[str, ...]) -> dict[str, float]:
    metrics = {"visual_quality": 0.92}
    for tag in challenge_tags:
        for metric_name in SEEDANCE_STYLE_CHALLENGE_METRICS.get(tag, ()):
            metrics[metric_name] = 0.91
    return metrics


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
                quality_metrics=_metrics_for(case.challenge_tags),
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
    assert verdict.missing_metric_evidence == frozenset()
    assert "identity_similarity" in verdict.evaluated_metric_names
    assert "long_range_continuity" in verdict.evaluated_metric_names
    assert verdict.to_dict()["schema"] == "cineos-competitive-acceptance/0.2"


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


def test_generic_visual_quality_cannot_substitute_for_challenge_measurement(tmp_path):
    report = _passing_report(tmp_path)
    generic_only = tuple(
        replace(shot, quality_metrics={"visual_quality": 0.99}) for shot in report.shots
    )
    report = CompetitiveBenchmarkReport(
        scene_id=report.scene_id,
        foundation=report.foundation,
        shots=generic_only,
    )

    verdict = evaluate_competitive_acceptance(report)

    assert verdict.passed is False
    assert verdict.missing_metric_evidence
    assert "identity_consistency" in verdict.missing_metric_evidence
    assert "dialogue" in verdict.missing_metric_evidence
    assert "physics" in verdict.missing_metric_evidence
    assert any(
        "missing challenge-specific metric evidence" in reason
        for reason in verdict.reasons
    )
