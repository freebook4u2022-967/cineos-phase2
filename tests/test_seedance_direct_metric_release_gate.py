from pathlib import Path

import pytest

from cineos.benchmarks.exceptions import BenchmarkError
from cineos.benchmarks.metrics import Metric, MetricStatus
from cineos.benchmarks.real_evidence import validate_real_inference_evidence
from cineos.benchmarks.report import CaseResult
from cineos.benchmarks.seedance_competitive import seedance_competitive_suite


def _write_expected_outputs(root: Path) -> None:
    (root / "report.json").write_text('{"passed": true}', encoding="utf-8")
    (root / "render_receipt.json").write_text('{"renderer": "gpu"}', encoding="utf-8")
    (root / "output.mp4").write_bytes(
        b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41"
    )


def test_hands_object_case_rejects_proxy_only_quality_evidence(tmp_path):
    case = next(
        item
        for item in seedance_competitive_suite().cases
        if item.case_id == "competitive-hands-object"
    )
    result = CaseResult(
        case_id=case.case_id,
        passed=True,
        metrics=(
            Metric("prop_continuity_score", 0.99, MetricStatus.MEASURED),
            Metric("temporal_stability", 0.99, MetricStatus.MEASURED),
        ),
        outputs=case.expected_outputs,
    )
    _write_expected_outputs(tmp_path)

    with pytest.raises(BenchmarkError, match="missing required measured metric"):
        validate_real_inference_evidence(
            case,
            result,
            tmp_path,
            foundation={
                "origin": "external_pretrained_foundation",
                "model_id": "declared/model",
            },
        )
