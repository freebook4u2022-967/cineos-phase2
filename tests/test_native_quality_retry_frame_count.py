from pathlib import Path
from types import SimpleNamespace

from cineos.native_video.competitive_benchmark import (
    BenchmarkCase,
    VisualEvaluation,
    run_competitive_benchmark,
)
from cineos.native_video.quality_retry import QualityRetryPolicy


class _FrameCountRenderer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls = 0

    def render(self, request):
        self.calls += 1
        output = self.output_dir / f"attempt-{self.calls}.mp4"
        output.write_bytes(b"real-render-artifact")
        return SimpleNamespace(output_path=output, frame_count=24 + self.calls)


def test_competitive_retry_preserves_selected_render_frame_count(
    tmp_path: Path,
) -> None:
    renderer = _FrameCountRenderer(tmp_path)
    evaluations = 0

    def evaluator(path, request):
        nonlocal evaluations
        evaluations += 1
        passed = evaluations == 2
        return VisualEvaluation(
            passed=passed,
            metrics={"identity_consistency": 0.95 if passed else 0.45},
        )

    case = BenchmarkCase(
        shot_id="retry-frame-count",
        prompt="Hero remains visually identical through a measured rerender.",
        challenge_tags=("identity_consistency",),
        camera={"shot_size": "close-up", "movement": "locked"},
    )

    report = run_competitive_benchmark(
        renderer,
        approved_reference_ids=("approved-hero-reference",),
        evaluator=evaluator,
        cases=(case,),
        quality_retry_policy=QualityRetryPolicy(max_attempts=2),
    )

    shot = report.shots[0]
    assert report.production_passed is True
    assert shot.rerendered is True
    assert shot.attempt_count == 2
    assert shot.selected_attempt == 2
    assert shot.frame_count == 26
    assert shot.output_path is not None
    assert shot.output_path.endswith("attempt-2.mp4")
