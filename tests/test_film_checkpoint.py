from __future__ import annotations

import json

import pytest

from cineos.film.build import BuildStatus, FilmBuild
from cineos.film.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    load_checkpoint,
    load_checkpoint_runtime_state,
    save_checkpoint,
)
from cineos.film.shot_state import ShotState


def _build() -> FilmBuild:
    build = FilmBuild(
        project_id="project-1",
        film_package_id="film-package-1",
        renderer_id="native-atlas",
    )
    build.transition(BuildStatus.RENDERING)
    build.shot_states = [
        ShotState(
            shot_id="shot-1",
            attempt_count=2,
            output_path="shots/shot-1-a2.mp4",
            render_status="completed",
            validation_status="approved",
            recovery_status="recovered",
            selected_output="shots/shot-1-a2.mp4",
            output_hash="abc123",
            attempt_history=[{"attempt": 1, "approved": False}],
        )
    ]
    return build


def test_checkpoint_round_trip_preserves_build_state(tmp_path):
    build = _build()
    path = save_checkpoint(build, tmp_path / "state" / "build.json")

    restored = load_checkpoint(path)

    assert restored.build_id == build.build_id
    assert restored.status == BuildStatus.RENDERING
    assert restored.shot("shot-1").attempt_count == 2
    assert restored.shot("shot-1").recovery_status == "recovered"
    assert restored.content_hash == build.content_hash


def test_checkpoint_document_is_versioned(tmp_path):
    path = save_checkpoint(_build(), tmp_path / "build.json")
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert document["build"]["renderer_id"] == "native-atlas"
    assert document["content_hash"]


def test_checkpoint_round_trip_preserves_runtime_state(tmp_path):
    runtime_state = {
        "kind": "native-scene-continuity",
        "memory": {
            "schema": "cineos-scene-continuity-memory/0.1",
            "anchors": [{"shot_id": "shot-1", "scene_index": 0}],
        },
    }
    path = save_checkpoint(
        _build(), tmp_path / "build.json", runtime_state=runtime_state
    )

    assert load_checkpoint_runtime_state(path) == runtime_state
    assert load_checkpoint(path).content_hash == _build().content_hash


def test_old_checkpoint_without_runtime_state_remains_compatible(tmp_path):
    path = save_checkpoint(_build(), tmp_path / "build.json")

    assert load_checkpoint_runtime_state(path) is None


def test_checkpoint_detects_runtime_state_tampering(tmp_path):
    path = save_checkpoint(
        _build(),
        tmp_path / "build.json",
        runtime_state={"memory": {"anchors": [{"shot_id": "shot-1"}]}},
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["runtime_state"]["memory"]["anchors"][0]["shot_id"] = "poisoned"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="runtime_state hash mismatch"):
        load_checkpoint_runtime_state(path)


def test_checkpoint_rejects_partial_runtime_state_pair(tmp_path):
    path = save_checkpoint(
        _build(), tmp_path / "build.json", runtime_state={"memory": {"anchors": []}}
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("runtime_state_hash")
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="missing integrity hash"):
        load_checkpoint_runtime_state(path)


def test_checkpoint_rejects_unsupported_schema(tmp_path):
    path = save_checkpoint(_build(), tmp_path / "build.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="unsupported checkpoint schema"):
        load_checkpoint(path)


def test_checkpoint_detects_state_tampering(tmp_path):
    path = save_checkpoint(_build(), tmp_path / "build.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["build"]["renderer_id"] = "unexpected-renderer"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="content hash mismatch"):
        load_checkpoint(path)
