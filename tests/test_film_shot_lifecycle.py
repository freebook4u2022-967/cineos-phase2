from __future__ import annotations

from types import SimpleNamespace

from cineos.film.build import FilmBuild
from cineos.film.orchestrator import FilmOrchestrator
from cineos.film.shot_state import ShotState


class _Renderer:
    def render(self, planned, target):
        target.write_bytes(f"attempt:{target.name}".encode("utf-8"))
        return target


class _RejectThenApprove:
    def __init__(self):
        self.calls = 0

    def validate(self, path, planned):
        self.calls += 1
        return {"approved": self.calls >= 2}


def test_shot_attempt_lifecycle_is_transactional_across_recovery(tmp_path):
    events: list[tuple[str, int, int]] = []
    orchestrator = FilmOrchestrator(
        _Renderer(),
        validator=_RejectThenApprove(),
        max_recovery_attempts=1,
        shot_attempt_start=lambda planned, scene, attempt: events.append(
            ("start", scene, attempt)
        ),
        shot_attempt_accepted=lambda planned, scene, attempt: events.append(
            ("accepted", scene, attempt)
        ),
        shot_attempt_rejected=lambda planned, scene, attempt: events.append(
            ("rejected", scene, attempt)
        ),
    )
    planned = SimpleNamespace(
        shot_id="shot-1",
        scene_id="scene-a",
        index=0,
        duration=1.0,
        payload={},
    )
    state = ShotState("shot-1")
    build = FilmBuild("project", "package", "native")

    orchestrator._render_shot(
        planned,
        state,
        tmp_path,
        build,
        scene_index=3,
    )

    assert state.approved
    assert state.attempt_count == 2
    assert events == [
        ("start", 3, 1),
        ("rejected", 3, 1),
        ("start", 3, 2),
        ("accepted", 3, 2),
    ]


def test_accepted_hook_failure_is_rolled_back_and_retried(tmp_path):
    events: list[tuple[str, int]] = []
    accepted_calls = 0

    def accepted(planned, scene, attempt):
        nonlocal accepted_calls
        accepted_calls += 1
        events.append(("accepted", attempt))
        if accepted_calls == 1:
            raise RuntimeError("continuity commit failed")

    orchestrator = FilmOrchestrator(
        _Renderer(),
        validator=lambda: None,
        max_recovery_attempts=1,
        shot_attempt_start=lambda planned, scene, attempt: events.append(
            ("start", attempt)
        ),
        shot_attempt_accepted=accepted,
        shot_attempt_rejected=lambda planned, scene, attempt: events.append(
            ("rejected", attempt)
        ),
    )
    orchestrator.validator = SimpleNamespace(
        validate=lambda path, planned: {"approved": True}
    )
    planned = SimpleNamespace(
        shot_id="shot-2",
        scene_id="scene-b",
        index=1,
        duration=1.0,
        payload={},
    )
    state = ShotState("shot-2")
    build = FilmBuild("project", "package", "native")

    orchestrator._render_shot(
        planned,
        state,
        tmp_path,
        build,
        scene_index=4,
    )

    assert state.approved
    assert state.attempt_count == 2
    assert events == [
        ("start", 1),
        ("accepted", 1),
        ("rejected", 1),
        ("start", 2),
        ("accepted", 2),
    ]
