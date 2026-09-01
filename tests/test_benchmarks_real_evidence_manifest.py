import json

import pytest

from cineos.benchmarks.exceptions import BenchmarkError
from cineos.benchmarks.metrics import Metric, MetricStatus
from cineos.benchmarks.real_evidence import (
    ARTIFACT_MANIFEST_NAME,
    validate_real_inference_evidence,
    write_real_inference_artifact_manifest,
)
from cineos.benchmarks.report import CaseResult
from cineos.benchmarks.seedance_competitive import seedance_competitive_suite


FOUNDATION = {
    "origin": "external_pretrained_foundation",
    "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}


def _case():
    return seedance_competitive_suite().cases[0]


def _write_expected_outputs(case, root):
    root.mkdir(parents=True, exist_ok=True)
    for name in case.expected_outputs:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps({"case_id": case.case_id}) + "\n", encoding="utf-8")
        elif path.suffix == ".mp4":
            path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2")
        else:
            path.write_bytes(b"evidence")


def _result(case):
    metrics = tuple(
        Metric(name=name, value=max(1.0, threshold), status=MetricStatus.MEASURED)
        for name, threshold in case.validation_thresholds.items()
    )
    return CaseResult(
        case_id=case.case_id,
        passed=True,
        metrics=metrics,
        outputs=case.expected_outputs,
        deterministic_hash="fixture-result-hash",
    )


def _validate(case, root):
    validate_real_inference_evidence(
        case,
        _result(case),
        root,
        foundation=FOUNDATION,
        require_artifact_manifest=True,
    )


def test_content_bound_manifest_accepts_unchanged_artifacts(tmp_path):
    case = _case()
    _write_expected_outputs(case, tmp_path)

    manifest_path = write_real_inference_artifact_manifest(case, tmp_path)

    assert manifest_path.name == ARTIFACT_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["case_id"] == case.case_id
    assert set(payload["artifacts"]) == set(case.expected_outputs)
    _validate(case, tmp_path)


def test_content_bound_manifest_rejects_artifact_modified_after_measurement(tmp_path):
    case = _case()
    _write_expected_outputs(case, tmp_path)
    write_real_inference_artifact_manifest(case, tmp_path)
    (tmp_path / "output.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00tampered-video"
    )

    with pytest.raises(BenchmarkError, match="manifest"):
        _validate(case, tmp_path)


def test_production_validation_rejects_missing_manifest(tmp_path):
    case = _case()
    _write_expected_outputs(case, tmp_path)

    with pytest.raises(BenchmarkError, match="artifact manifest"):
        _validate(case, tmp_path)


def test_content_bound_manifest_rejects_wrong_case_identity(tmp_path):
    case = _case()
    _write_expected_outputs(case, tmp_path)
    manifest_path = write_real_inference_artifact_manifest(case, tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["case_id"] = "competitive-other-case"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="case_id"):
        _validate(case, tmp_path)


def test_content_bound_manifest_rejects_unexpected_artifact_entry(tmp_path):
    case = _case()
    _write_expected_outputs(case, tmp_path)
    manifest_path = write_real_inference_artifact_manifest(case, tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"]["unapproved.bin"] = {
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="exactly match"):
        _validate(case, tmp_path)


def test_research_validation_remains_backwards_compatible_without_manifest(tmp_path):
    case = _case()
    _write_expected_outputs(case, tmp_path)

    validate_real_inference_evidence(
        case,
        _result(case),
        tmp_path,
        foundation=FOUNDATION,
    )
