from pathlib import Path
from types import SimpleNamespace

from cineos.atlas.native_request import NativeShotRequest
from cineos.native_video.competitive_benchmark import VisualEvaluation
from cineos.native_video.quality_retry import (
    QualityRetryPolicy,
    render_with_quality_retries,
)


class _FeedbackAwareRenderer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.requests: list[NativeShotRequest] = []

    def render(self, request: NativeShotRequest):
        self.requests.append(request)
        path = self.output_dir / f"attempt-{len(self.requests)}.mp4"
        path.write_bytes(b"real-render-artifact")
        return SimpleNamespace(output_path=path, frame_count=32)


def _request() -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="feedback-shot",
        scene_id="feedback-scene",
        camera={"shot_size": "full", "movement": "tracking"},
        characters=[
            {
                "character_id": "hero",
                "identity_invariants": ["same approved face", "natural anatomy"],
            }
        ],
        environment={"name": "corridor"},
        wardrobe=[{"character_id": "hero", "description": "dark coat"}],
        props=[{"prop_id": "key", "continuity": "same metal key"}],
        continuity={"previous_shot_id": "shot-03", "scene_anchor": "same corridor"},
        performance={"action": "run toward camera"},
        approved_reference_ids=["hero-approved-ref"],
        deterministic_seed=41,
        renderer_requirements={"benchmark": {"require_real_artifact": True}},
        metadata={"prompt": "Hero runs toward camera holding the same key."},
    )
    request.refresh_hash()
    return request


def test_failed_metrics_are_forwarded_as_targeted_retry_feedback(
    tmp_path: Path,
) -> None:
    renderer = _FeedbackAwareRenderer(tmp_path)

    def evaluator(path: Path, request: NativeShotRequest) -> VisualEvaluation:
        assert path.is_file()
        retry = request.metadata["qc_retry"]
        if retry["attempt"] == 1:
            assert "feedback" not in retry
            return VisualEvaluation(
                passed=False,
                metrics={
                    "identity_similarity": 0.81,
                    "anatomy": 0.44,
                    "temporal_consistency": 0.57,
                    "object_integrity": 0.63,
                },
                notes=("hands deform during fast motion",),
            )

        feedback = retry["feedback"]
        assert feedback["source_attempt"] == 1
        assert feedback["quality_passed"] is False
        assert feedback["weakest_metrics"] == [
            {"name": "anatomy", "score": 0.44},
            {"name": "temporal_consistency", "score": 0.57},
        ]
        assert feedback["evaluator_notes"] == ["hands deform during fast motion"]
        return VisualEvaluation(
            passed=True,
            metrics={
                "identity_similarity": 0.92,
                "anatomy": 0.88,
                "temporal_consistency": 0.86,
                "object_integrity": 0.9,
            },
        )

    result = render_with_quality_retries(
        renderer,
        evaluator,
        _request(),
        policy=QualityRetryPolicy(
            max_attempts=2,
            seed_stride=17,
            feedback_metric_limit=2,
        ),
    )

    assert result.accepted is True
    assert result.selected_attempt == 2
    assert result.attempt_count == 2
    assert renderer.requests[0].deterministic_seed == 41
    assert renderer.requests[1].deterministic_seed == 58
    assert result.attempts[0].conditioning_hash == result.attempts[1].conditioning_hash
    assert renderer.requests[0].camera == renderer.requests[1].camera
    assert renderer.requests[0].characters == renderer.requests[1].characters
    assert renderer.requests[0].continuity == renderer.requests[1].continuity
    assert (
        renderer.requests[0].approved_reference_ids
        == renderer.requests[1].approved_reference_ids
    )
