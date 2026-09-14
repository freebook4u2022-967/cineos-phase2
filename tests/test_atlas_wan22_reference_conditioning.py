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
    def __init__(self, frame_count):
        self.frames = [[f"frame-{index}" for index in range(frame_count)]]


class ReferenceAwarePipeline:
    def __init__(self, frame_count):
        self.frame_count = frame_count
        self.image = None

    def to(self, _device):
        return self

    def enable_vae_tiling(self):
        return None

    def __call__(
        self,
        prompt,
        width,
        height,
        num_frames,
        generator=None,
        image=None,
        negative_prompt=None,
        num_inference_steps=None,
        guidance_scale=None,
    ):
        del (
            prompt,
            width,
            height,
            num_frames,
            generator,
            negative_prompt,
            num_inference_steps,
            guidance_scale,
        )
        self.image = image
        return FakeOutput(self.frame_count)


def test_wan22_request_binds_approved_reference_id():
    config = Wan22ExecutionConfig(
        prompt="same actor turns toward camera",
        approved_reference_id="hero-approved-front",
    )

    request = build_wan22_execution_request(config)

    assert request.approved_reference_ids == ["hero-approved-front"]
    assert request.metadata["reference_conditioned"] is True


def test_wan22_reference_conditioning_fails_closed_without_loader(tmp_path):
    config = Wan22ExecutionConfig(
        prompt="same actor turns toward camera",
        approved_reference_id="hero-approved-front",
    )

    with pytest.raises(Wan22ExecutionError, match="reference_loader"):
        run_wan22_gpu_validation(config, output_dir=tmp_path)


def test_wan22_reference_loader_reaches_diffusers_image_input(tmp_path):
    config = Wan22ExecutionConfig(
        prompt="same actor walks toward camera",
        requested_duration_seconds=1.0,
        approved_reference_id="hero-approved-front",
    )
    frame_count = aligned_wan22_frame_count(1.0, config.fps)
    pipeline = ReferenceAwarePipeline(frame_count)

    def exporter(_frames, output_path, *, fps):
        del fps
        Path(output_path).write_bytes(b"reference-conditioned-video")

    receipt = run_wan22_gpu_validation(
        config,
        output_dir=tmp_path,
        device="cpu",
        dtype="float32",
        memory_strategy="resident",
        reference_loader=lambda reference_id: f"loaded:{reference_id}",
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=exporter,
    )

    assert pipeline.image == "loaded:hero-approved-front"
    assert receipt["conditioning"] == {
        "reference_conditioned": True,
        "approved_reference_id": "hero-approved-front",
    }
    assert receipt["actual_frame_count"] == frame_count
