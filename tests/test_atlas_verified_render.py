from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoRenderer, FoundationProvenance
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.verified_render import VerifiedRenderError, render_verified


class FakeOutput:
    def __init__(self):
        self.frames = [["f1", "f2", "f3"]]


class FakePipeline:
    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames, generator=None):
        return FakeOutput()


def _request():
    request = NativeShotRequest(
        shot_id="shot-verified",
        scene_id="scene-verified",
        camera={"resolution": (832, 480), "fps": 24, "duration": 1.0},
        characters=[],
        environment={"description": "night street"},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=[],
        deterministic_seed=77,
        renderer_requirements={},
        metadata={"action": "A character walks through rain"},
    )
    request.refresh_hash()
    return request


def _renderer(tmp_path, exporter):
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            revision="test-revision",
            license_id="Apache-2.0",
        ),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        video_exporter=exporter,
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")
    return renderer


def test_verified_render_writes_evidence_bound_to_actual_artifact(tmp_path):
    def exporter(_frames, path, *, fps):
        assert fps == 24.0
        Path(path).write_bytes(b"verified-video-bytes")

    renderer = _renderer(tmp_path, exporter)
    request = _request()
    result = render_verified(renderer, request)

    assert Path(result.render.output_path).is_file()
    assert Path(result.evidence_path).is_file()
    assert result.evidence.artifact_bytes == len(b"verified-video-bytes")
    assert result.evidence.request_hash == request.content_hash
    assert result.evidence.foundation_model_id == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert result.evidence.foundation_revision == "test-revision"
    assert result.evidence.foundation_license_id == "Apache-2.0"
    assert result.evidence.device == "cpu"
    assert result.evidence.dtype == "float32"
    assert result.evidence.memory_strategy == "resident"


def test_verified_render_fails_closed_when_exporter_creates_no_artifact(tmp_path):
    renderer = _renderer(tmp_path, lambda *_args, **_kwargs: None)

    with pytest.raises(VerifiedRenderError, match="artifact does not exist"):
        render_verified(renderer, _request())


def test_verified_render_fails_closed_when_exporter_creates_empty_artifact(tmp_path):
    def exporter(_frames, path, *, fps):
        Path(path).touch()

    renderer = _renderer(tmp_path, exporter)

    with pytest.raises(VerifiedRenderError, match="artifact is empty"):
        render_verified(renderer, _request())
