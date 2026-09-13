from pathlib import Path

import pytest

from cineos.benchmarks.exceptions import BenchmarkError
from cineos.benchmarks.metrics import Metric, MetricStatus
from cineos.benchmarks.real_evidence import validate_real_inference_evidence
from cineos.benchmarks.report import CaseResult
from cineos.benchmarks.seedance_competitive import seedance_competitive_suite


def _case():
    return seedance_competitive_suite().cases[0]


def _write_expected_outputs(root: Path) -> None:
    (root / "report.json").write_text('{"passed": true}', encoding="utf-8")
    (root / "render_receipt.json").write_text('{"renderer": "gpu"}', encoding="utf-8")
    (root / "output.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41"
    )


def _passing_result():
    case = _case()
    metrics = tuple(
        Metric(name, threshold, MetricStatus.MEASURED)
        for name, threshold in case.validation_thresholds.items()
    )
    return CaseResult(
        case_id=case.case_id,
        passed=True,
        metrics=metrics,
        outputs=case.expected_outputs,
    )


def _foundation():
    return {
        "origin": "external_pretrained_foundation",
        "model_id": "declared/model",
    }


def test_real_inference_evidence_accepts_structured_artifacts_measured_metrics_and_provenance(
    tmp_path,
):
    _write_expected_outputs(tmp_path)

    validate_real_inference_evidence(
        _case(), _passing_result(), tmp_path, foundation=_foundation()
    )


def test_real_inference_evidence_rejects_missing_or_empty_media(tmp_path):
    _write_expected_outputs(tmp_path)
    (tmp_path / "output.mp4").write_bytes(b"")

    with pytest.raises(BenchmarkError, match="missing or empty benchmark artifact"):
        validate_real_inference_evidence(
            _case(), _passing_result(), tmp_path, foundation=_foundation()
        )


def test_real_inference_evidence_rejects_placeholder_non_mp4_media(tmp_path):
    _write_expected_outputs(tmp_path)
    (tmp_path / "output.mp4").write_bytes(b"not-a-real-video")

    with pytest.raises(BenchmarkError, match="not an MP4/ISO-BMFF container"):
        validate_real_inference_evidence(
            _case(), _passing_result(), tmp_path, foundation=_foundation()
        )


def test_real_inference_evidence_rejects_malformed_json_receipt(tmp_path):
    _write_expected_outputs(tmp_path)
    (tmp_path / "render_receipt.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="JSON artifact is malformed"):
        validate_real_inference_evidence(
            _case(), _passing_result(), tmp_path, foundation=_foundation()
        )


def test_real_inference_evidence_rejects_non_object_json_report(tmp_path):
    _write_expected_outputs(tmp_path)
    (tmp_path / "report.json").write_text("[]", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="must contain an object"):
        validate_real_inference_evidence(
            _case(), _passing_result(), tmp_path, foundation=_foundation()
        )


def test_real_inference_evidence_rejects_estimated_threshold_metric(tmp_path):
    _write_expected_outputs(tmp_path)
    result = _passing_result()
    first = result.metrics[0]
    estimated = Metric(first.name, first.value, MetricStatus.ESTIMATED)
    result = CaseResult(
        case_id=result.case_id,
        passed=True,
        metrics=(estimated, *result.metrics[1:]),
        outputs=result.outputs,
    )

    with pytest.raises(BenchmarkError, match="required metric is not measured"):
        validate_real_inference_evidence(
            _case(), result, tmp_path, foundation=_foundation()
        )


def test_real_inference_evidence_rejects_below_threshold_metric(tmp_path):
    _write_expected_outputs(tmp_path)
    result = _passing_result()
    first = result.metrics[0]
    below = Metric(first.name, float(first.value) - 0.01, MetricStatus.MEASURED)
    result = CaseResult(
        case_id=result.case_id,
        passed=True,
        metrics=(below, *result.metrics[1:]),
        outputs=result.outputs,
    )

    with pytest.raises(BenchmarkError, match="below threshold"):
        validate_real_inference_evidence(
            _case(), result, tmp_path, foundation=_foundation()
        )


def test_real_inference_evidence_rejects_ambiguous_foundation_origin(tmp_path):
    _write_expected_outputs(tmp_path)
    foundation = {"origin": "cineos_native", "model_id": "declared/model"}

    with pytest.raises(BenchmarkError, match="external_pretrained_foundation"):
        validate_real_inference_evidence(
            _case(), _passing_result(), tmp_path, foundation=foundation
        )
