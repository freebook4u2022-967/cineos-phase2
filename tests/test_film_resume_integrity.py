from __future__ import annotations

from types import SimpleNamespace

import pytest

import cineos.film.orchestrator as orchestrator_module
from cineos.film.build import FilmBuild
from cineos.film.checkpoint import save_checkpoint
from cineos.film.exceptions import FilmBuildError
from cineos.film.orchestrator import FilmOrchestrator
from cineos.film.planner import PlannedShot, shot_plan_fingerprint
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


def test_resume_run_restores_persisted_build_before_planning_reuse(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"approved-shot")
    saved = FilmBuild("project", "package", "native")
    saved.shot_states = [_approved("shot-1", artifact)]
    saved.metadata["resume_contract"] = {"schema": "test", "sha256": "same"}
    checkpoint = save_checkpoint(saved, tmp_path / "film.checkpoint.json")

    requested = FilmBuild("project", "package", "native")
    requested.metadata["resume_contract"] = {"schema": "test", "sha256": "same"}
    planned = SimpleNamespace(
        shot_id="shot-1", scene_id="scene-1", index=0, duration=1.0
    )
    monkeypatch.setattr(orchestrator_module, "plan_shots", lambda package: [planned])

    result = FilmOrchestrator(renderer=object()).run(
        object(),
        requested,
        tmp_path / "output",
        dry_run=True,
        resume=True,
        checkpoint_path=checkpoint,
    )

    assert result.build_id == saved.build_id
    assert result.shot("shot-1").approved
    assert result.shot("shot-1").selected_output == str(artifact)
    assert result.metadata["shot_plan_fingerprint"] == shot_plan_fingerprint([planned])


def test_resume_run_rejects_different_build_identity_before_reuse(tmp_path):
    saved = FilmBuild("project", "package", "native")
    checkpoint = save_checkpoint(saved, tmp_path / "film.checkpoint.json")
    requested = FilmBuild("other-project", "package", "native")

    with pytest.raises(FilmBuildError, match="project_id"):
        FilmOrchestrator(renderer=object()).run(
            object(),
            requested,
            tmp_path / "output",
            dry_run=True,
            resume=True,
            checkpoint_path=checkpoint,
        )


def test_resume_run_rejects_changed_creative_contract(tmp_path):
    saved = FilmBuild("project", "package", "native")
    saved.metadata["resume_contract"] = {"schema": "test", "sha256": "old"}
    checkpoint = save_checkpoint(saved, tmp_path / "film.checkpoint.json")
    requested = FilmBuild("project", "package", "native")
    requested.metadata["resume_contract"] = {"schema": "test", "sha256": "new"}

    with pytest.raises(FilmBuildError, match="resume_contract"):
        FilmOrchestrator(renderer=object()).run(
            object(),
            requested,
            tmp_path / "output",
            dry_run=True,
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


def test_resume_rejects_reordered_legacy_shot_timeline(tmp_path, monkeypatch):
    saved = FilmBuild("project", "package", "native")
    saved.shot_states = [ShotState("shot-1"), ShotState("shot-2")]
    checkpoint = save_checkpoint(saved, tmp_path / "film.checkpoint.json")
    requested = FilmBuild("project", "package", "native")
    plan = [
        PlannedShot("shot-2", "scene", 1.0, 0),
        PlannedShot("shot-1", "scene", 1.0, 1),
    ]
    monkeypatch.setattr(orchestrator_module, "plan_shots", lambda package: plan)

    with pytest.raises(FilmBuildError, match="legacy resume checkpoint shot order"):
        FilmOrchestrator(renderer=object()).run(
            object(),
            requested,
            tmp_path / "output",
            dry_run=True,
            resume=True,
            checkpoint_path=checkpoint,
        )


def test_resume_rejects_changed_renderer_facing_shot_payload(tmp_path, monkeypatch):
    original = [
        PlannedShot(
            "shot-1",
            "scene-1",
            1.0,
            0,
            {
                "shot_id": "shot-1",
                "scene_id": "scene-1",
                "duration": 1.0,
                "prompt": "old",
            },
        )
    ]
    saved = FilmBuild("project", "package", "native")
    saved.shot_states = [ShotState("shot-1")]
    saved.metadata["shot_plan_fingerprint"] = shot_plan_fingerprint(original)
    checkpoint = save_checkpoint(saved, tmp_path / "film.checkpoint.json")
    requested = FilmBuild("project", "package", "native")
    changed = [
        PlannedShot(
            "shot-1",
            "scene-1",
            1.0,
            0,
            {
                "shot_id": "shot-1",
                "scene_id": "scene-1",
                "duration": 1.0,
                "prompt": "new",
            },
        )
    ]
    monkeypatch.setattr(orchestrator_module, "plan_shots", lambda package: changed)

    with pytest.raises(FilmBuildError, match="shot plan differs"):
        FilmOrchestrator(renderer=object()).run(
            object(),
            requested,
            tmp_path / "output",
            dry_run=True,
            resume=True,
            checkpoint_path=checkpoint,
        )


def test_new_build_checkpoints_full_shot_plan_fingerprint(tmp_path, monkeypatch):
    build = FilmBuild("project", "package", "native")
    plan = [PlannedShot("shot-1", "scene-1", 1.5, 0, {"prompt": "frame"})]
    monkeypatch.setattr(orchestrator_module, "plan_shots", lambda package: plan)

    result = FilmOrchestrator(renderer=object()).run(
        object(),
        build,
        tmp_path / "output",
        dry_run=True,
        checkpoint_path=tmp_path / "film.checkpoint.json",
    )

    assert result.metadata["shot_plan_fingerprint"] == shot_plan_fingerprint(plan)
