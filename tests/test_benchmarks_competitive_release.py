import pytest

from cineos.benchmarks.competitive_release import validate_seedance_competitive_release
from cineos.benchmarks.exceptions import BenchmarkError
from cineos.benchmarks.report import BenchmarkReport, CaseResult
from cineos.benchmarks.seedance_competitive import seedance_competitive_suite


def _report(*, metadata=None, results=None, suite_hash=None):
    suite = seedance_competitive_suite()
    if results is None:
        results = tuple(
            CaseResult(
                case_id=case.case_id,
                passed=True,
                metrics=(),
                outputs=case.expected_outputs,
                deterministic_hash=f"hash-{case.case_id}",
            )
            for case in suite.cases
            if case.mandatory
        )
    return BenchmarkReport(
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.content_hash if suite_hash is None else suite_hash,
        renderer_profile=suite.renderer_profile,
        hardware_profile="NVIDIA CUDA production runner",
        results=results,
        metadata={
            "production_gpu_evidence": True,
            "real_inference": True,
            "commit_sha": "abcdef1234567890",
            **(metadata or {}),
        },
    )


def _output_dirs(report):
    return {result.case_id: f"/evidence/{result.case_id}" for result in report.results}


def test_release_gate_validates_every_mandatory_case(monkeypatch):
    report = _report()
    calls = []

    def fake_validate(case, result, output_dir, *, foundation):
        calls.append(
            (case.case_id, result.case_id, str(output_dir), foundation["model_id"])
        )

    monkeypatch.setattr(
        "cineos.benchmarks.competitive_release.validate_real_inference_evidence",
        fake_validate,
    )

    validate_seedance_competitive_release(
        report,
        case_output_dirs=_output_dirs(report),
        foundation={
            "origin": "external_pretrained_foundation",
            "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        },
    )

    suite = seedance_competitive_suite()
    assert [item[0] for item in calls] == [
        case.case_id for case in suite.cases if case.mandatory
    ]
    assert all(case_id == result_id for case_id, result_id, *_ in calls)


def test_release_gate_rejects_stale_suite_hash(monkeypatch):
    report = _report(suite_hash="0" * 64)
    monkeypatch.setattr(
        "cineos.benchmarks.competitive_release.validate_real_inference_evidence",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(BenchmarkError, match="suite_hash"):
        validate_seedance_competitive_release(
            report,
            case_output_dirs=_output_dirs(report),
            foundation={
                "origin": "external_pretrained_foundation",
                "model_id": "model",
            },
        )


def test_release_gate_rejects_non_production_attestation(monkeypatch):
    report = _report(metadata={"production_gpu_evidence": False})
    monkeypatch.setattr(
        "cineos.benchmarks.competitive_release.validate_real_inference_evidence",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(BenchmarkError, match="production GPU evidence"):
        validate_seedance_competitive_release(
            report,
            case_output_dirs=_output_dirs(report),
            foundation={
                "origin": "external_pretrained_foundation",
                "model_id": "model",
            },
        )


def test_release_gate_rejects_duplicate_case_result(monkeypatch):
    base = _report()
    duplicate = base.results + (base.results[0],)
    report = _report(results=duplicate)
    monkeypatch.setattr(
        "cineos.benchmarks.competitive_release.validate_real_inference_evidence",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(BenchmarkError, match="duplicate"):
        validate_seedance_competitive_release(
            report,
            case_output_dirs=_output_dirs(report),
            foundation={
                "origin": "external_pretrained_foundation",
                "model_id": "model",
            },
        )


def test_release_gate_rejects_missing_case_output_directory(monkeypatch):
    report = _report()
    outputs = _output_dirs(report)
    outputs.pop(report.results[-1].case_id)
    monkeypatch.setattr(
        "cineos.benchmarks.competitive_release.validate_real_inference_evidence",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(BenchmarkError, match="missing output directory"):
        validate_seedance_competitive_release(
            report,
            case_output_dirs=outputs,
            foundation={
                "origin": "external_pretrained_foundation",
                "model_id": "model",
            },
        )
