from pathlib import Path

import pytest

from cineos.native_video.competitive_benchmark import (
    VisualEvaluation,
    default_connected_cases,
    run_competitive_benchmark,
)
from cineos.native_video.quality_retry import QualityRetryPolicy


class RetryRenderer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.requests = []

    def render(self, request):
        self.requests.append(request)
        path = self.output_dir / f"{request.shot_id}-{len(self.requests)}.mp4"
        path.write_bytes(f"artifact-{len(self.requests)}".encode())
        return path


def test_competitive_benchmark_rerenders_measured_failure_and_records_evidence(tmp_path):
    renderer = RetryRenderer(tmp_path)
    evaluations = iter(
        [
            VisualEvaluation(False, {"identity_similarity": 0.62}),
            VisualEvaluation(True, {"identity_similarity": 0.94}),
        ]
    )

    report = run_competitive_benchmark(
        renderer,
        approved_reference_ids=("approved-hero",),
        evaluator=lambda path, request: next(evaluations),
        cases=(default_connected_cases()[0],),
        quality_retry_policy=QualityRetryPolicy(max_attempts=3, seed_stride=101),
    )

    shot = report.shots[0]
    assert report.production_passed is True
    assert shot.attempt_count == 2
    assert shot.selected_attempt == 2
    assert shot.rerendered is True
    assert shot.quality_metrics["identity_similarity"] == pytest.approx(0.94)
    assert len(renderer.requests) == 2
    assert [request.deterministic_seed for request in renderer.requests] == [
        20260929,
        20261030,
    ]
    assert renderer.requests[0].camera == renderer.requests[1].camera
    assert renderer.requests[0].characters == renderer.requests[1].characters
    assert renderer.requests[0].continuity == renderer.requests[1].continuity
    assert renderer.requests[0].approved_reference_ids == renderer.requests[1].approved_reference_ids
    assert shot.request_hash == renderer.requests[1].content_hash
    assert "measured quality retry attempts=2, selected_attempt=2" in shot.notes


def test_competitive_benchmark_quality_retry_requires_measured_evaluator(tmp_path):
    with pytest.raises(ValueError, match="quality retries require a visual evaluator"):
        run_competitive_benchmark(
            RetryRenderer(tmp_path),
            approved_reference_ids=("approved-hero",),
            cases=(default_connected_cases()[0],),
            quality_retry_policy=QualityRetryPolicy(max_attempts=2),
        )
