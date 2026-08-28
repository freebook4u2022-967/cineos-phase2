from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import (
    DiffusersVideoError,
    DiffusersVideoRenderer,
    FoundationProvenance,
)
from cineos.atlas.native_request import NativeShotRequest


class FakeOutput:
    def __init__(self):
        self.frames = [["f1", "f2", "f3"]]


class FakePipeline:
    def __init__(self):
        self.device = None
        self.calls = []
        self.progress_disabled = False
        self.memory_calls = []

    def to(self, device):
        self.device = device
        return self

    def enable_model_cpu_offload(self):
        self.memory_calls.append("model_cpu_offload")

    def enable_sequential_cpu_offload(self):
        self.memory_calls.append("sequential_cpu_offload")

    def enable_vae_tiling(self):
        self.memory_calls.append("vae_tiling")

    def enable_vae_slicing(self):
        self.memory_calls.append("vae_slicing")

    def enable_attention_slicing(self):
        self.memory_calls.append("attention_slicing")

    def set_progress_bar_config(self, *, disable):
        self.progress_disabled = disable

    def __call__(self, prompt, width, height, num_frames, generator=None, image=None):
        self.calls.append(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "generator": generator,
                "image": image,
            }
        )
        return FakeOutput()


def _request():
    request = NativeShotRequest(
        shot_id="shot-001",
        scene_id="scene-001",
        camera={
            "resolution": (1280, 720),
            "fps": 24,
            "duration": 2.0,
            "shot_size": "medium close-up",
            "movement": "slow dolly in",
        },
        characters=[
            {
                "approved_reference_ids": ["hero-front"],
                "identity_invariants": ["same face", "same black coat"],
            }
        ],
        environment={"description": "rainy neon street at night"},
        wardrobe=[],
        props=[],
        continuity={},
        performance={"facial_targets": ["determined"], "gesture_tracks": []},
        approved_reference_ids=["hero-front"],
        deterministic_seed=1234,
        renderer_requirements={},
        metadata={"action": "The hero turns toward camera"},
    )
    request.refresh_hash()
    return request


def test_diffusers_renderer_executes_injected_pipeline_and_exports_video(tmp_path):
    pipeline = FakePipeline()
    factory_calls = []
    exports = []

    def factory(model_id, **options):
        factory_calls.append((model_id, options))
        return pipeline

    def exporter(frames, path, *, fps):
        exports.append((frames, path, fps))
        Path(path).write_text("fake-video", encoding="utf-8")

    renderer = DiffusersVideoRenderer(
        FoundationProvenance(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            foundation_name="Wan2.2 TI2V 5B",
            license_id="Apache-2.0",
        ),
        output_dir=tmp_path,
        reference_loader=lambda reference_id: f"image:{reference_id}",
        pipeline_factory=factory,
        video_exporter=exporter,
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32", local_files_only=True)
    renderer.warmup()
    result = renderer.render(_request())

    assert factory_calls == [
        ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", {"local_files_only": True})
    ]
    assert pipeline.device == "cpu"
    assert pipeline.progress_disabled is True
    assert pipeline.calls[0]["width"] == 1280
    assert pipeline.calls[0]["height"] == 720
    assert pipeline.calls[0]["num_frames"] == 48
    assert pipeline.calls[0]["generator"] == 1234
    assert pipeline.calls[0]["image"] == "image:hero-front"
    assert "The hero turns toward camera" in pipeline.calls[0]["prompt"]
    assert "same face" in pipeline.calls[0]["prompt"]
    assert result.frame_count == 3
    assert result.seed == 1234
    assert result.foundation.license_id == "Apache-2.0"
    assert result.request_hash == _request().content_hash
    assert Path(result.output_path).exists()
    assert exports[0][2] == 24.0


def test_diffusers_renderer_applies_constrained_gpu_memory_policy(tmp_path):
    pipeline = FakePipeline()
    factory_calls = []

    def factory(model_id, **options):
        factory_calls.append((model_id, options))
        return pipeline

    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=factory,
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    renderer.load_model(
        device="cuda",
        dtype="float16",
        memory_strategy="model_cpu_offload",
        enable_vae_tiling=True,
        enable_vae_slicing=True,
        enable_attention_slicing=True,
        local_files_only=True,
    )

    assert factory_calls == [("declared/model", {"local_files_only": True})]
    assert pipeline.device is None
    assert pipeline.memory_calls == [
        "model_cpu_offload",
        "vae_tiling",
        "vae_slicing",
        "attention_slicing",
    ]


def test_diffusers_renderer_rejects_unknown_memory_strategy(tmp_path):
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    with pytest.raises(DiffusersVideoError, match="memory_strategy"):
        renderer.load_model(device="cuda", memory_strategy="magic")


def test_diffusers_renderer_rejects_cpu_offload_without_cuda(tmp_path):
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    with pytest.raises(DiffusersVideoError, match="CUDA"):
        renderer.load_model(device="cpu", memory_strategy="model_cpu_offload")


def test_diffusers_renderer_requires_requested_memory_feature(tmp_path):
    class NoOffloadPipeline:
        def to(self, device):
            return self

    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: NoOffloadPipeline(),
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    with pytest.raises(DiffusersVideoError, match="enable_model_cpu_offload"):
        renderer.load_model(device="cuda", memory_strategy="model_cpu_offload")


def test_diffusers_renderer_rejects_model_override_without_matching_provenance(
    tmp_path,
):
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    with pytest.raises(DiffusersVideoError, match="provenance"):
        renderer.load_model("different/model", device="cpu")


def test_diffusers_renderer_requires_native_shot_request(tmp_path):
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: FakePipeline(),
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    renderer.load_model(device="cpu")
    with pytest.raises(TypeError, match="NativeShotRequest"):
        renderer.render({"shot_id": "wrong"})