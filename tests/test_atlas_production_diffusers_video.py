from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError
from cineos.atlas.foundation_profiles import build_wan22_ti2v_5b_renderer
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_diffusers_video import ValidatedDiffusersVideoRenderer


class FakeOutput:
    def __init__(self):
        self.frames = [["frame-1", "frame-2"]]


class FakePipeline:
    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames, generator=None, image=None):
        del prompt, width, height, num_frames, generator, image
        return FakeOutput()


def _request() -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="shot-production-001",
        scene_id="scene-production-001",
        camera={
            "resolution": (1280, 704),
            "fps": 24,
            "duration": 1.0,
            "shot_size": "medium",
            "movement": "locked",
        },
        characters=[],
        environment={"description": "controlled benchmark stage"},
        wardrobe=[],
        props=[],
        continuity={},
        performance={"facial_targets": [], "gesture_tracks": []},
        approved_reference_ids=[],
        deterministic_seed=77,
        renderer_requirements={},
        metadata={"action": "A subject holds position for artifact validation"},
    )
    request.refresh_hash()
    return request


def _box(box_type: bytes, payload: bytes = b"") -> bytes:
    return (8 + len(payload)).to_bytes(4, "big") + box_type + payload


def _valid_mp4_bytes() -> bytes:
    return b"".join(
        (
            _box(b"ftyp", b"isom"),
            _box(b"moov"),
            _box(b"mdat", b"video-payload"),
        )
    )


def _renderer(tmp_path, exporter):
    return build_wan22_ti2v_5b_renderer(
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        video_exporter=exporter,
    )


def test_pinned_wan22_profile_uses_fail_closed_production_renderer(tmp_path):
    renderer = _renderer(
        tmp_path,
        lambda _frames, path, *, fps: Path(path).write_bytes(_valid_mp4_bytes()),
    )
    assert isinstance(renderer, ValidatedDiffusersVideoRenderer)

    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")
    result = renderer.render(_request())

    assert Path(result.output_path).read_bytes() == _valid_mp4_bytes()
    assert result.frame_count == 2


def test_production_renderer_rejects_non_mp4_exporter_output(tmp_path):
    renderer = _renderer(
        tmp_path,
        lambda _frames, path, *, fps: Path(path).write_bytes(b"not-an-mp4"),
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")

    with pytest.raises(DiffusersVideoError, match="MP4 integrity validation"):
        renderer.render(_request())


def test_production_renderer_rejects_exporter_that_writes_nothing(tmp_path):
    renderer = _renderer(tmp_path, lambda _frames, _path, *, fps: None)
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")

    with pytest.raises(DiffusersVideoError, match="MP4 integrity validation"):
        renderer.render(_request())
