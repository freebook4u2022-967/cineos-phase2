from __future__ import annotations

import pytest

import cineos.film.orchestrator as orchestrator_module
from cineos.film.build import FilmBuild
from cineos.film.checkpoint import save_checkpoint
from cineos.film.exceptions import FilmBuildError
from cineos.film.orchestrator import FilmOrchestrator
from cineos.film.shot_state import ShotState
from cineos.film.validator import file_hash, validate_reusable_output
from cineos.native_video.film_bridge import NativeFilmContinuityBridge


def _approved(shot_id, path):
    state = ShotState(shot_id)
    state.validation_status = "approved"
    state.selected_output = str(path)
    state.output_hash = file_hash(path)
    return state


def test_validate_reusable_output_treats_missing_artifact_as_not_reusable(tmp_path):
    missing = tmp_path / "missing.mp4"
    assert not validate_reusable_output(missing, "0" * 64)


def test_native_film_bridge_exposes_resume_reset_hook():
    bridge = NativeFilmContinuityBridge.default()
    hooks = bridge.orchestrator_kwargs()

    assert hooks["checkpoint_state_resetter"] == bridge.reset
    bridge._active["sentinel"] = object()
    hooks["checkpoint_state_resetter"]()
    assert bridge._active == {}
    assert bridge.memory.latest() is None


def test_stateful_resume_run_fails_closed_without_runtime_restorer(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "film.checkpoint.json"
    build = FilmBuild("project", "package", "native")
    save_checkpoint(build, checkpoint, runtime_state={"kind": "test-runtime"})
    monkeypatch.setattr(orchestrator_module, "plan_shots", lambda package: [])
    orchestrator = FilmOrchestrator(renderer=object())

    with pytest.raises(FilmBuildError, match="no runtime state restorer"):
        orchestrator.run(
            object(),
            build,
            tmp_path / "output",
            resume=True,
            checkpoint_path=checkpoint,
        )


def test_stateful_resume_restores_only_contiguous_reusable_prefix(tmp_path):
    first = tmp_path / "first.mp4"
    first.write_bytes(b"first")
    build = FilmBuild("project", "package", "native")
    build.shot_states = [_approved("shot-1", first), ShotState("shot-2")]
    restored = []
    reset = []
    orchestrator = FilmOrchestrator(
        renderer=object(),
        checkpoint_state_restorer=restored.append,
        checkpoint_state_resetter=lambda: reset.append(True),
    )
    runtime = {"kind": "test-runtime"}

    orchestrator._restore_runtime_for_resume(build, runtime, dry_run=False)

    assert restored == [runtime]
    assert reset == []
    assert build.metadata["resume_integrity"] == {
        "action": "restored_contiguous_prefix",
        "approved_shots": 1,
        "reusable_prefix_shots": 1,
    }
    assert build.shot("shot-1").approved
    assert not build.shot("shot-2").approved


def test_stateful_resume_resets_whole_timeline_when_earlier_artifact_is_lost(tmp_path):
    first = tmp_path / "first.mp4"
    first.write_bytes(b"first")
    second = tmp_path / "second.mp4"
    second.write_bytes(b"second")
    first_state = _approved("shot-1", first)
    second_state = _approved("shot-2", second)
    first.unlink()

    build = FilmBuild("project", "package", "native")
    build.shot_states = [first_state, second_state]
    restored = []
    reset = []
    orchestrator = FilmOrchestrator(
        renderer=object(),
        checkpoint_state_restorer=restored.append,
        checkpoint_state_resetter=lambda: reset.append(True),
    )

    orchestrator._restore_runtime_for_resume(
        build, {"kind": "test-runtime"}, dry_run=False
    )

    assert restored == []
    assert reset == [True]
    assert all(not state.approved for state in build.shot_states)
    assert build.metadata["resume_integrity"] == {
        "action": "full_timeline_regeneration",
        "approved_shots": 2,
        "reusable_prefix_shots": 0,
    }
    assert any("continuity runtime was reset" in item for item in build.warnings)


def test_stateful_resume_fails_closed_without_runtime_reset_hook(tmp_path):
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"shot")
    state = _approved("shot-2", artifact)
    build = FilmBuild("project", "package", "native")
    build.shot_states = [ShotState("shot-1"), state]
    orchestrator = FilmOrchestrator(
        renderer=object(),
        checkpoint_state_restorer=lambda payload: None,
    )

    with pytest.raises(FilmBuildError, match="no runtime reset hook"):
        orchestrator._restore_runtime_for_resume(
            build, {"kind": "test-runtime"}, dry_run=False
        )
