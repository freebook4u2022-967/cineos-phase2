import json

import pytest

from cineos.film import CheckpointError, FilmBuild, load_checkpoint, save_checkpoint


def _build() -> FilmBuild:
    return FilmBuild(
        project_id="project-1",
        film_package_id="package-1",
        renderer_id="native-renderer",
    )


def test_checkpoint_requires_integrity_hash(tmp_path):
    path = save_checkpoint(_build(), tmp_path / "film.checkpoint.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document.pop("content_hash")
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="missing content hash"):
        load_checkpoint(path)


def test_checkpoint_rejects_tampered_build_state(tmp_path):
    path = save_checkpoint(_build(), tmp_path / "film.checkpoint.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["build"]["renderer_id"] = "tampered-renderer"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="content hash mismatch"):
        load_checkpoint(path)


def test_checkpoint_rejects_non_object_shot_state_entries(tmp_path):
    path = save_checkpoint(_build(), tmp_path / "film.checkpoint.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["build"]["shot_states"] = ["invalid"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="shot_states entries must be objects"):
        load_checkpoint(path)
