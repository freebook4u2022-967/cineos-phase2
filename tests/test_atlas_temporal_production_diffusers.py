from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError
from cineos.atlas.foundation_profiles import (
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.temporal_production_diffusers import (
    TemporalLatticeProductionDiffusersVideoRenderer,
)


class Output:
    def __init__(self, frames):
        self.frames = [frames]


class Pipeline:
    def __init__(self, calls, *, returned_frame_delta=0):
        self.calls = calls
        self.returned_frame_delta = returned_frame_delta

    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames, generator=None, image=None):
        self.calls.append(num_frames)
        count = num_frames + self.returned_frame_delta
        return Output([f"frame-{index}" for index in range(count)])


def _request(*, fps=24.0, duration=5.0, references=()):
    request = NativeShotRequest(
        shot_id="shot-temporal",
        scene_id="scene-temporal",
        camera={"resolution": (1280, 704), "fps": fps, "duration": duration},
        characters=[],
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={"facial_targets": [], "gesture_tracks": []},
        approved_reference_ids=list(references),
        deterministic_seed=29,
        renderer_requirements={},
        metadata={"benchmark_challenges": ["fast_camera_movement"]},
    )
    request.refresh_hash()
    return request


def _renderer(tmp_path, profile, *, returned_frame_delta=0):
    calls = []
    pipeline = Pipeline(calls, returned_frame_delta=returned_frame_delta)
    exported = []

    def exporter(frames, path, *, fps):
        exported.append((list(frames), fps))
        Path(path).write_bytes(b"temporal-production-video")

    renderer = profile.renderer(
        output_dir=tmp_path,
        reference_loader=lambda reference_id: f"image:{reference_id}",
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=exporter,
    )
    renderer.initialize()
    renderer.load_model(device="cpu")
    return renderer, calls, exported


def test_wan_profiles_declare_native_temporal_compression_lattice():
    assert WAN22_TI2V_5B_PROFILE.temporal_compression_ratio == 4
    assert WAN22_I2V_A14B_PROFILE.temporal_compression_ratio == 4
    assert WAN22_TI2V_5B_PROFILE.snapshot()["temporal_compression_ratio"] == 4
    assert WAN22_I2V_A14B_PROFILE.snapshot()["temporal_compression_ratio"] == 4


def test_wan_5b_requests_lattice_frames_then_exports_exact_film_duration(tmp_path):
    renderer, calls, exported = _renderer(tmp_path, WAN22_TI2V_5B_PROFILE)
    request = _request(fps=24.0, duration=5.0)

    result = renderer.render(request)

    assert isinstance(renderer, TemporalLatticeProductionDiffusersVideoRenderer)
    assert calls == [121]
    assert len(exported) == 1
    frames, fps = exported[0]
    assert fps == 24.0
    assert len(frames) == 120
    assert frames[-1] == "frame-119"
    assert result.frame_count == 120
    assert result.request_hash == request.content_hash
    plan = result.conditioning_provenance["temporal_frame_plan"]
    assert plan["foundation_generated_frames"] == 121
    assert plan["export_frames"] == 120
    assert plan["trimmed_frames"] == 1
    assert renderer._terminal_frames[(request.scene_id, request.shot_id)] == "frame-119"


def test_wan_a14b_uses_81_native_frames_for_five_seconds_at_16fps(tmp_path):
    renderer, calls, exported = _renderer(tmp_path, WAN22_I2V_A14B_PROFILE)

    result = renderer.render(
        _request(fps=16.0, duration=5.0, references=("hero-front",))
    )

    assert calls == [81]
    assert len(exported[0][0]) == 80
    assert result.frame_count == 80


def test_foundation_frame_count_drift_fails_before_export(tmp_path):
    renderer, calls, exported = _renderer(
        tmp_path,
        WAN22_TI2V_5B_PROFILE,
        returned_frame_delta=-1,
    )

    with pytest.raises(DiffusersVideoError, match="expected 121, got 120"):
        renderer.render(_request(fps=24.0, duration=5.0))

    assert calls == [121]
    assert exported == []
