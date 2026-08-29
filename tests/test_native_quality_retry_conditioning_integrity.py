from pathlib import Path
from types import SimpleNamespace

from cineos.atlas.native_request import NativeShotRequest
from cineos.native_video.competitive_benchmark import VisualEvaluation
from cineos.native_video.quality_retry import QualityRetryPolicy, render_with_quality_retries


class _Renderer:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.requests: list[NativeShotRequest] = []

    def render(self, request: NativeShotRequest):
        self.requests.append(request)
        path = self.output_dir / f"attempt-{len(self.requests)}.mp4"
        path.write_bytes(b"rendered-video")
        return SimpleNamespace(output_path=path, frame_count=24)


def _request() -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="integrity-shot",
        scene_id="integrity-scene",
        camera={"shot_size": "medium", "movement": "tracking"},
        characters=[
            {
                "character_id": "hero",
                "identity_invariants": ["same face", "same body proportions"],
            }
        ],
        environment={"name": "corridor"},
        wardrobe=[{"character_id": "hero", "description": "dark coat"}],
        props=[{"prop_id": "key", "continuity": "same metal key"}],
        continuity={"previous_shot_id": "shot-01", "scene_anchor": "same corridor"},
        performance={"dialogue": "Run."},
        approved_reference_ids=["hero-approved-ref"],
        deterministic_seed=77,
        renderer_requirements={"benchmark": {"require_real_artifact": True}},
        metadata={"prompt": "Hero runs through the corridor."},
    )
    request.refresh_hash()
    return request


def test_quality_retries_preserve_conditioning_hash_while_varying_seed(tmp_path: Path) -> None:
    renderer = _Renderer(tmp_path)
    evaluations = 0

    def evaluator(path: Path, request: NativeShotRequest) -> VisualEvaluation:
        nonlocal evaluations
        evaluations += 1
        return VisualEvaluation(
            passed=evaluations == 2,
            metrics={"identity_similarity": 0.91 if evaluations == 2 else 0.62},
        )

    result = render_with_quality_retries(
        renderer,
        evaluator,
        _request(),
        policy=QualityRetryPolicy(max_attempts=2, seed_stride=101),
    )

    assert result.accepted is True
    assert result.selected_attempt == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].request_hash != result.attempts[1].request_hash
    assert result.attempts[0].deterministic_seed == 77
    assert result.attempts[1].deterministic_seed == 178
    assert result.attempts[0].conditioning_hash
    assert result.attempts[0].conditioning_hash == result.attempts[1].conditioning_hash

    first, second = renderer.requests
    assert first.camera == second.camera
    assert first.characters == second.characters
    assert first.environment == second.environment
    assert first.wardrobe == second.wardrobe
    assert first.props == second.props
    assert first.continuity == second.continuity
    assert first.performance == second.performance
    assert first.approved_reference_ids == second.approved_reference_ids
    assert first.renderer_requirements == second.renderer_requirements
