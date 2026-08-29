from dataclasses import dataclass, field

import pytest

from cineos.atlas import AtlasRuntime, RuntimeState
from cineos.compiler import compile
from cineos.core import MovieProject, Scene, Shot, Timeline
from cineos.validation import ValidationReport, ValidationStatus


@dataclass
class RenderResult:
    attempt: int
    renderer_metadata: dict = field(default_factory=dict)


def make_package():
    shot = Shot("shot-1", duration=1)
    scene = Scene("scene-1", "Scene", shots=[shot], duration=1)
    timeline = Timeline([scene.scene_id], {scene.scene_id: [shot.shot_id]})
    return compile(MovieProject("Film", "Author", scenes=[scene], timeline=timeline))


def report(task, status):
    return ValidationReport(
        shot_id=task.shot_id,
        scene_id=task.scene_id,
        renderer_id="test-renderer",
        overall_status=status,
        overall_score=1.0 if status is ValidationStatus.PASS else 0.0,
        results=[],
    )


def test_runtime_rerenders_failed_shot_until_validation_passes() -> None:
    attempts = []
    statuses = iter([ValidationStatus.FAIL, ValidationStatus.PASS])

    def render(_task):
        result = RenderResult(attempt=len(attempts))
        attempts.append(result)
        return result

    def validate(task, _result):
        return report(task, next(statuses))

    job = AtlasRuntime().prepare(make_package())
    AtlasRuntime().run_with_validation(
        job,
        render,
        validate,
        max_rerenders=2,
    )

    assert job.state is RuntimeState.COMPLETED
    assert len(attempts) == 2
    final = job.results["shot-1"]
    assert final.attempt == 1
    assert final.renderer_metadata["rerender_attempts"] == 1
    assert final.renderer_metadata["mark_for_rerender"] is False
    assert len(final.renderer_metadata["validation_history"]) == 2
    assert final.renderer_metadata["validation_history"][0]["overall_status"] == "fail"
    assert final.renderer_metadata["validation_history"][1]["overall_status"] == "pass"


def test_runtime_can_adapt_rerender_from_previous_validation_report() -> None:
    first_results = []
    rerender_calls = []

    def render(_task):
        result = RenderResult(attempt=0)
        first_results.append(result)
        return result

    def rerender(task, attempt, previous_report):
        rerender_calls.append(
            (task.shot_id, attempt, previous_report.overall_status)
        )
        return RenderResult(attempt=attempt)

    def validate(task, result):
        status = ValidationStatus.FAIL if result.attempt == 0 else ValidationStatus.PASS
        return report(task, status)

    job = AtlasRuntime().prepare(make_package())
    AtlasRuntime().run_with_validation(
        job,
        render,
        validate,
        max_rerenders=2,
        rerender_handler=rerender,
    )

    assert len(first_results) == 1
    assert rerender_calls == [("shot-1", 1, ValidationStatus.FAIL)]
    final = job.results["shot-1"]
    assert final.attempt == 1
    assert final.renderer_metadata["rerender_attempts"] == 1
    assert final.renderer_metadata["mark_for_rerender"] is False


def test_runtime_stops_when_rerender_budget_is_exhausted() -> None:
    attempts = []

    def render(_task):
        result = RenderResult(attempt=len(attempts))
        attempts.append(result)
        return result

    def validate(task, _result):
        return report(task, ValidationStatus.FAIL)

    job = AtlasRuntime().prepare(make_package())
    AtlasRuntime().run_with_validation(
        job,
        render,
        validate,
        max_rerenders=1,
    )

    assert len(attempts) == 2
    final = job.results["shot-1"]
    assert final.renderer_metadata["rerender_attempts"] == 1
    assert final.renderer_metadata["mark_for_rerender"] is True
    assert len(final.renderer_metadata["validation_history"]) == 2


def test_runtime_validation_default_preserves_single_attempt_behavior() -> None:
    attempts = []

    def render(_task):
        result = RenderResult(attempt=len(attempts))
        attempts.append(result)
        return result

    def validate(task, _result):
        return report(task, ValidationStatus.FAIL)

    job = AtlasRuntime().prepare(make_package())
    AtlasRuntime().run_with_validation(job, render, validate)

    assert len(attempts) == 1
    assert job.results["shot-1"].renderer_metadata["mark_for_rerender"] is True


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_runtime_rejects_invalid_rerender_budget(value) -> None:
    job = AtlasRuntime().prepare(make_package())

    def render(_task):
        return RenderResult(attempt=0)

    def validate(task, _result):
        return report(task, ValidationStatus.PASS)

    expected = TypeError if value in {True, 1.5} else ValueError
    with pytest.raises(expected):
        AtlasRuntime().run_with_validation(
            job,
            render,
            validate,
            max_rerenders=value,
        )


def test_runtime_rejects_non_callable_rerender_handler() -> None:
    job = AtlasRuntime().prepare(make_package())

    with pytest.raises(TypeError, match="rerender_handler"):
        AtlasRuntime().run_with_validation(
            job,
            lambda _task: RenderResult(attempt=0),
            lambda task, _result: report(task, ValidationStatus.PASS),
            max_rerenders=1,
            rerender_handler="invalid",
        )
