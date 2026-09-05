from __future__ import annotations

from types import SimpleNamespace

from cineos.film.build import FilmBuild
from cineos.film.checkpoint import load_checkpoint_runtime_state, save_checkpoint
from cineos.film.orchestrator import FilmOrchestrator


def _build() -> FilmBuild:
    return FilmBuild(
        project_id="project-runtime",
        film_package_id="package-runtime",
        renderer_id="native-atlas",
    )


def _empty_package():
    return SimpleNamespace(shot_manifest=[], timeline_manifest={})


def test_orchestrator_restores_and_repersists_native_runtime_state(tmp_path):
    checkpoint = tmp_path / "build.json"
    previous = {
        "kind": "native-scene-continuity",
        "memory": {
            "schema": "cineos-scene-continuity-memory/0.1",
            "anchors": [{"scene_index": 0, "shot_id": "shot-previous"}],
        },
    }
    save_checkpoint(_build(), checkpoint, runtime_state=previous)

    restored = []
    current = {
        "kind": "native-scene-continuity",
        "memory": {
            "schema": "cineos-scene-continuity-memory/0.1",
            "anchors": [{"scene_index": 1, "shot_id": "shot-current"}],
        },
    }
    orchestrator = FilmOrchestrator(
        renderer=object(),
        checkpoint_state_provider=lambda: current,
        checkpoint_state_restorer=restored.append,
    )

    result = orchestrator.run(
        _empty_package(),
        _build(),
        tmp_path / "output",
        dry_run=True,
        resume=True,
        checkpoint_path=checkpoint,
    )

    assert restored == [previous]
    assert result.metadata["dry_run"]["shot_count"] == 0
    assert load_checkpoint_runtime_state(checkpoint) == current


def test_orchestrator_resume_accepts_legacy_checkpoint_without_runtime_state(tmp_path):
    checkpoint = tmp_path / "build.json"
    save_checkpoint(_build(), checkpoint)
    restored = []

    orchestrator = FilmOrchestrator(
        renderer=object(),
        checkpoint_state_provider=lambda: None,
        checkpoint_state_restorer=restored.append,
    )
    orchestrator.run(
        _empty_package(),
        _build(),
        tmp_path / "output",
        dry_run=True,
        resume=True,
        checkpoint_path=checkpoint,
    )

    assert restored == []
    assert load_checkpoint_runtime_state(checkpoint) is None
