from pathlib import Path

import pytest

from cineos.atlas.wan22_execution import (
    Wan22ExecutionConfig,
    Wan22ExecutionError,
    aligned_wan22_frame_count,
    build_wan22_execution_request,
    run_wan22_gpu_validation,
)


class FakeOutput:
    def __init__(self, frames):
        self.frames = [frames]


class FakeWanPipeline:
    def __init__(self):
        self.calls = []
        self.to_calls = []
        self.progress_disabled = False
        self.vae_tiling_enabled = False

    def enable_model_cpu_offload(self, gpu_id=None):
        self.to_calls.append(("offload", gpu_id))

    def enable_vae_tiling(self):
        self.vae_tiling_enabled = True

    def set_progress_bar_config(self, disable=False):
        self.progress_disabled = disable

    def __call__(
        self,
        prompt,
        width,
        height,
        num_frames,
        generator=None,
        negative_prompt=None,
        guidance_scale=None,
        num_inference_steps=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "generator": generator,
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
        )
        return FakeOutput([f"frame-{index}" for index in range(num_frames)])


def test_wan22_frame_count_rounds_up_to_temporal_contract():
    assert aligned_wan22_frame_count(5.0, 24.0) == 121
    assert aligned_wan22_frame_count(1.0, 24.0) == 25
    assert aligned_wan22_frame_count(81 / 24, 24.0) == 81


def test_wan22_execution_request_preserves_requested_duration_and_audits_alignment():
    request = build_wan22_execution_request(
        Wan22ExecutionConfig(prompt="hero walks through a cinematic corridor")
    )

    assert request.camera["resolution"] == (1280, 704)
    assert request.camera["fps"] == 24.0
    assert request.metadata["requested_duration_seconds"] == 5.0
    assert request.metadata["execution_frame_count"] == 121
    assert request.camera["duration"] == 121 / 24
    assert request.metadata["foundation_origin"] == "external_pretrained_foundation"
    assert len(request.content_hash) == 64


def test_wan22_validation_routes_real_execution_controls_and_returns_receipt(tmp_path):
    pipeline = FakeWanPipeline()
    exported = {}

    def exporter(frames, output_path, fps):
        exported["frames"] = frames
        exported["output_path"] = output_path
        exported["fps"] = fps
        Path(output_path).write_bytes(b"fake-mp4-artifact")

    receipt = run_wan22_gpu_validation(
        Wan22ExecutionConfig(
            prompt="hero keeps the same face while walking toward camera",
            num_inference_steps=28,
            guidance_scale=4.5,
        ),
        output_dir=tmp_path,
        device="cuda",
        memory_strategy="model_cpu_offload",
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=exporter,
    )

    assert pipeline.to_calls == [("offload", 0)]
    assert pipeline.vae_tiling_enabled is True
    assert pipeline.progress_disabled is True
    assert pipeline.calls[0]["num_frames"] == 121
    assert pipeline.calls[0]["width"] == 1280
    assert pipeline.calls[0]["height"] == 704
    assert pipeline.calls[0]["guidance_scale"] == 4.5
    assert pipeline.calls[0]["num_inference_steps"] == 28
    assert "identity drift" in pipeline.calls[0]["negative_prompt"]
    assert receipt["actual_frame_count"] == 121
    assert receipt["execution_frame_count"] == 121
    assert receipt["artifact"]["exists"] is True
    assert receipt["artifact"]["size_bytes"] == len(b"fake-mp4-artifact")
    assert len(receipt["artifact"]["sha256"]) == 64
    assert receipt["execution_elapsed_seconds"] >= 0
    assert receipt["runtime"] == {
        "device": "cuda",
        "dtype": "bfloat16",
        "memory_strategy": "model_cpu_offload",
        "vae_tiling": True,
        "num_inference_steps": 28,
        "guidance_scale": 4.5,
    }
    assert receipt["foundation_profile"]["origin"] == "external_pretrained_foundation"
    assert receipt["foundation"]["model_id"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert exported["fps"] == 24.0


def test_wan22_validation_rejects_empty_export(tmp_path):
    pipeline = FakeWanPipeline()

    def empty_exporter(_frames, output_path, _fps):
        Path(output_path).touch()

    with pytest.raises(Wan22ExecutionError, match="empty artifact"):
        run_wan22_gpu_validation(
            Wan22ExecutionConfig(prompt="auditable GPU render"),
            output_dir=tmp_path,
            pipeline_factory=lambda *_args, **_kwargs: pipeline,
            video_exporter=empty_exporter,
        )
