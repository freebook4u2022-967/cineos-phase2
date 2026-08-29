from dataclasses import dataclass
from pathlib import Path

from cineos.native_video.competitive_benchmark import (
    VisualEvaluation,
    default_connected_cases,
    run_competitive_benchmark,
)


@dataclass(frozen=True)
class FakeFoundation:
    model_id: str = "open/foundation"

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "revision": "test-revision",
            "license_id": "test-license",
        }


@dataclass(frozen=True)
class FakeRenderResult:
    output_path: str
    frame_count: int


class ArtifactRenderer:
    foundation = FakeFoundation()

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.requests = []

    def render(self, request):
        self.requests.append(request)
        path = self.output_dir / f"{request.shot_id}.mp4"
        path.write_bytes(b"real-artifact-bytes")
        return FakeRenderResult(str(path), 32)


def test_default_suite_covers_connected_seedance_style_failures():
    cases = default_connected_cases()
    assert len(cases) == 10
    tags = {tag for case in cases for tag in case.challenge_tags}
    assert {
        "identity_consistency",
        "multi_character_interaction",
        "hands_anatomy",
        "walking",
        "running",
        "dialogue",
        "fast_camera_movement",
        "lighting_change",
        "physics",
        "long_range_continuity",
    } <= tags


def test_execution_evidence_cannot_be_mislabeled_as_competitive_quality(tmp_path):
    renderer = ArtifactRenderer(tmp_path)
    report = run_competitive_benchmark(
        renderer,
        approved_reference_ids=("approved-hero",),
        cases=(default_connected_cases()[0],),
    )

    assert report.execution_passed is True
    assert report.quality_validated is False
    assert report.quality_passed is False
    assert report.production_passed is False
    assert report.foundation["model_id"] == "open/foundation"
    assert renderer.requests[0].approved_reference_ids == ["approved-hero"]
    assert renderer.requests[0].content_hash


def test_measured_visual_pass_is_required_for_production_pass(tmp_path):
    renderer = ArtifactRenderer(tmp_path)

    def evaluator(path, request):
        assert path.is_file()
        assert request.metadata["benchmark_challenges"]
        return VisualEvaluation(
            passed=True,
            metrics={
                "identity_consistency": 0.94,
                "temporal_consistency": 0.91,
            },
            notes=("measured by test evaluator",),
        )

    report = run_competitive_benchmark(
        renderer,
        approved_reference_ids=("approved-hero",),
        evaluator=evaluator,
        cases=(default_connected_cases()[0],),
    )

    assert report.execution_passed is True
    assert report.quality_validated is True
    assert report.quality_passed is True
    assert report.production_passed is True
    assert report.shots[0].quality_metrics["identity_consistency"] == 0.94

    target = report.write_json(tmp_path / "benchmark.json")
    payload = target.read_text(encoding="utf-8")
    assert '"schema": "cineos-competitive-video-benchmark/0.1"' in payload
    assert '"production_passed": true' in payload
