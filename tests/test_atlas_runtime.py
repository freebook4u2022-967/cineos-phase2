import pytest

from cineos.atlas import AtlasRuntime, RuntimeState, RuntimeStateError
from cineos.compiler import compile
from cineos.core import MovieProject, Scene, Shot, Timeline


def make_package():
    shots = [Shot("shot-1", duration=1), Shot("shot-2", duration=2)]
    scene = Scene("scene-1", "Scene", shots=shots, duration=3)
    timeline = Timeline([scene.scene_id], {scene.scene_id: [s.shot_id for s in shots]})
    return compile(MovieProject("Film", "Author", scenes=[scene], timeline=timeline))


def test_runtime_uses_film_package_timeline_order() -> None:
    seen = []
    runtime = AtlasRuntime()
    job = runtime.execute(
        make_package(),
        lambda task: seen.append((task.scene_id, task.shot_id)) or task.shot_id,
        job_id="job-1",
    )

    assert seen == [("scene-1", "shot-1"), ("scene-1", "shot-2")]
    assert job.job_id == "job-1"
    assert job.state is RuntimeState.COMPLETED
    assert job.completed == ["shot-1", "shot-2"]
    assert job.results == {"shot-1": "shot-1", "shot-2": "shot-2"}
    assert job.progress == 1


def test_runtime_records_handler_failure_without_implementing_rendering() -> None:
    job = AtlasRuntime().prepare(make_package())

    def fail(_task):
        raise LookupError("integration failed")

    with pytest.raises(LookupError, match="integration failed"):
        AtlasRuntime().run(job, fail)

    assert job.state is RuntimeState.FAILED
    assert isinstance(job.error, LookupError)
    with pytest.raises(RuntimeStateError, match="failed"):
        AtlasRuntime().run(job, fail)


def test_pending_job_can_be_cancelled() -> None:
    job = AtlasRuntime().prepare(make_package())
    job.cancel()

    assert job.state is RuntimeState.CANCELLED
    with pytest.raises(RuntimeStateError, match="cancelled"):
        AtlasRuntime().run(job, lambda task: task)
