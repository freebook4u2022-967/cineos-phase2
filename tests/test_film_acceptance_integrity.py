from __future__ import annotations

from types import SimpleNamespace

import cineos.film.orchestrator as orchestrator_module
from cineos.film.build import FilmBuild
from cineos.film.orchestrator import FilmOrchestrator
from cineos.film.shot_state import ShotState


class _Renderer:
    def render(self, planned, target):
        target.write_bytes(b"native-shot")
        return target


def _planned_shot():
    return SimpleNamespace(
        shot_id="shot-integrity",
        scene_id="scene-integrity",
        index=0,
        duration=1.0,
        payload={},
    )


def test_artifact_hash_failure_does_not_advance_continuity(monkeypatch, tmp_path):
    events: list[tuple[str, int]] = []

    def fail_hash(path):
        raise OSError("artifact became unreadable before commit")

    monkeypatch.setattr(orchestrator_module, "file_hash", fail_hash)
    orchestrator = FilmOrchestrator(
        _Renderer(),
        validator=SimpleNamespace(validate=lambda path, planned: {"approved": True}),
        max_recovery_attempts=0,
        shot_attempt_start=lambda planned, scene, attempt: events.append(
            ("start", attempt)
        ),
        shot_attempt_accepted=lambda planned, scene, attempt: events.append(
            ("accepted", attempt)
        ),
        shot_attempt_rejected=lambda planned, scene, attempt: events.append(
            ("rejected", attempt)
        ),
    )
    state = ShotState("shot-integrity")
    build = FilmBuild("project", "package", "native")

    orchestrator._render_shot(
        _planned_shot(),
        state,
        tmp_path,
        build,
        scene_index=2,
    )

    assert not state.approved
    assert state.validation_status == "rejected"
    assert state.recovery_status == "exhausted"
    assert state.output_hash is None
    assert events == [("start", 1), ("rejected", 1)]
    assert len(state.attempt_history) == 1
    assert state.attempt_history[0]["approved"] is False
    assert "unreadable before commit" in state.attempt_history[0]["reason"]
