import hashlib
from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import (
    DiffusersVideoError,
    DiffusersVideoRenderer,
    FoundationProvenance,
)
from cineos.atlas.native_request import NativeShotRequest


class Output:
    frames = [["frame-a", "frame-b"]]


class Pipeline:
    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames, generator=None):
        return Output()


def _request() -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="shot-artifact",
        scene_id="scene-artifact",
        camera={"resolution": (832, 480), "fps": 24, "duration": 1.0},
        characters=[],
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={"facial_targets": [], "gesture_tracks": []},
        approved_reference_ids=[],
        deterministic_seed=7,
        renderer_requirements={},
        metadata={"action": "A measured production render"},
    )
    request.refresh_hash()
    return request


def _renderer(tmp_path, exporter):
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: Pipeline(),
        video_exporter=exporter,
    )
    renderer.initialize()
    renderer.load_model(device="cpu")
    return renderer


def test_render_result_binds_fresh_artifact_size_and_sha256(tmp_path):
    payload = b"fresh-render-evidence"

    def exporter(_frames, path, *, fps):
        assert fps == 24.0
        Path(path).write_bytes(payload)

    result = _renderer(tmp_path, exporter).render(_request())

    assert result.artifact_size_bytes == len(payload)
    assert result.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert Path(result.output_path).read_bytes() == payload


def test_render_rejects_exporter_that_creates_no_artifact(tmp_path):
    renderer = _renderer(tmp_path, lambda *_args, **_kwargs: None)

    with pytest.raises(DiffusersVideoError, match="did not create"):
        renderer.render(_request())


def test_failed_rerender_cannot_reuse_stale_artifact(tmp_path):
    output_path = tmp_path / "scene-artifact-shot-artifact.mp4"
    output_path.write_bytes(b"stale-success-from-an-earlier-attempt")
    renderer = _renderer(tmp_path, lambda *_args, **_kwargs: None)

    with pytest.raises(DiffusersVideoError, match="did not create"):
        renderer.render(_request())

    assert not output_path.exists()


def test_render_rejects_empty_exported_artifact(tmp_path):
    def exporter(_frames, path, *, fps):
        Path(path).write_bytes(b"")

    renderer = _renderer(tmp_path, exporter)
    with pytest.raises(DiffusersVideoError, match="empty output artifact"):
        renderer.render(_request())
