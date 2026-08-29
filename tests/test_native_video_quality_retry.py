from pathlib import Path

import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.native_video.competitive_benchmark import VisualEvaluation
from cineos.native_video.quality_retry import (
    QualityRetryPolicy,
    render_with_quality_retries,
)


class _Renderer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.requests: list[NativeShotRequest] = []

    def render(self, request: NativeShotRequest) -> Path:
        self.requests.append(request)
        target = self.root / f"attempt-{len(self.requests)}.mp4"
        target.write_bytes(b"real-video-artifact")
        return target


class _FlakyRenderer(_Renderer):
    def render(self, request: NativeShotRequest) -> Path:
        self.requests.append(request)
        target = self.root / f"attempt-{len(self.requests)}.mp4"
        if len(self.requests) > 1:
            target.write_bytes(b"recovered-video-artifact")
        return target


def _request() -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="shot-01",
        scene_id="scene-01",
        camera={"resolution": (832, 480), "fps": 16.0, "duration": 2.0},
        characters=[{"character_id": "hero"}],
        environment={"name": "corridor"},
        wardrobe=[{"character_id": "hero", "description": "dark coat"}],
        props=[{"prop_id": "key"}],
        continuity={"scene_anchor": "same corridor"},
        performance={},
        approved_reference_ids=["hero-approved-ref"],
        deterministic_seed=17,
        renderer_requirements={"benchmark": {"require_real_artifact": True}},
        metadata={"prompt": "hero walks through corridor"},
    )
    request.refresh_hash()
    return request


def test_failed_qc_is_rerendered_with_new_seed_and_same_constraints(tmp_path):
    renderer = _Renderer(tmp_path)
    base = _request()
    base_hash = base.content_hash
    evaluations = iter(
        [
            VisualEvaluation(False, {"identity_similarity": 0.61}),
            VisualEvaluation(True, {"identity_similarity": 0.93}),
        ]
    )

    result = render_with_quality_retries(
        renderer,
        lambda path, request: next(evaluations),
        base,
        policy=QualityRetryPolicy(max_attempts=3, seed_stride=100),
    )

    assert result.accepted is True
    assert result.rerendered is True
    assert result.attempt_count == 2
    assert result.selected_attempt == 2
    assert [item.deterministic_seed for item in result.attempts] == [17, 117]
    assert len({item.request_hash for item in result.attempts}) == 2
    assert base.content_hash == base_hash
    assert [request.camera for request in renderer.requests] == [base.camera, base.camera]
    assert [request.characters for request in renderer.requests] == [
        base.characters,
        base.characters,
    ]
    assert all(
        request.metadata["qc_retry"]["base_request_hash"] == base_hash
        for request in renderer.requests
    )


def test_execution_failure_retries_before_measured_acceptance(tmp_path):
    renderer = _FlakyRenderer(tmp_path)
    evaluation_calls = 0

    def evaluator(path: Path, request: NativeShotRequest) -> VisualEvaluation:
        nonlocal evaluation_calls
        evaluation_calls += 1
        return VisualEvaluation(True, {"motion_naturalness": 0.91})

    result = render_with_quality_retries(
        renderer,
        evaluator,
        _request(),
        policy=QualityRetryPolicy(max_attempts=2),
    )

    assert result.accepted is True
    assert result.selected_attempt == 2
    assert result.attempts[0].execution_passed is False
    assert result.attempts[0].quality_evaluated is False
    assert result.attempts[1].execution_passed is True
    assert result.attempts[1].quality_passed is True
    assert evaluation_calls == 1


def test_exhausted_retry_budget_fails_closed_and_keeps_best_measured_attempt(tmp_path):
    renderer = _Renderer(tmp_path)
    scores = iter([0.72, 0.84, 0.79])

    def evaluator(path: Path, request: NativeShotRequest) -> VisualEvaluation:
        score = next(scores)
        return VisualEvaluation(False, {"camera_geometry_stability": score})

    result = render_with_quality_retries(renderer, evaluator, _request())

    assert result.accepted is False
    assert result.attempt_count == 3
    assert result.selected_attempt == 2
    assert result.selected_output_path is not None
    assert result.attempts[1].metric_mean == pytest.approx(0.84)
    assert all(item.quality_passed is False for item in result.attempts)


def test_retry_policy_validates_limits():
    with pytest.raises(ValueError, match="max_attempts"):
        QualityRetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="seed_stride"):
        QualityRetryPolicy(seed_stride=0)
