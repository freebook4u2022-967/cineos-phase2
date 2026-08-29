from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import (
    DiffusersVideoError,
    DiffusersVideoRenderer,
    FoundationProvenance,
)
from cineos.atlas.native_request import NativeShotRequest


class FakeOutput:
    frames = [["frame-1", "frame-2"]]


class QualityAwarePipeline:
    def __init__(self):
        self.calls = []

    def to(self, _device):
        return self

    def __call__(
        self,
        prompt,
        width,
        height,
        num_frames,
        generator=None,
        image=None,
        negative_prompt=None,
        guidance_scale=None,
        num_inference_steps=None,
        strength=None,
        max_sequence_length=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "generator": generator,
                "image": image,
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "strength": strength,
                "max_sequence_length": max_sequence_length,
            }
        )
        return FakeOutput()


def _request(*, metadata=None, renderer_requirements=None):
    request = NativeShotRequest(
        shot_id="shot-quality-001",
        scene_id="scene-quality-001",
        camera={"resolution": (1280, 704), "fps": 24, "duration": 1.0},
        characters=[{"identity_invariants": ["same face"]}],
        environment={"description": "cinematic night exterior"},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=["hero-approved"],
        deterministic_seed=99,
        renderer_requirements=renderer_requirements or {},
        metadata=metadata or {},
    )
    request.refresh_hash()
    return request


def test_renderer_forwards_only_supported_audited_quality_controls(tmp_path):
    pipeline = QualityAwarePipeline()
    exports = []
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=lambda frames, path, *, fps: (
            exports.append((frames, path, fps)),
            Path(path).write_text("video", encoding="utf-8"),
        ),
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")

    request = _request(
        renderer_requirements={
            "inference": {
                "guidance_scale": 4.5,
                "num_inference_steps": 28,
                "strength": 0.75,
                "unapproved_control": "must-not-leak",
            }
        },
        metadata={
            "action": "hero walks through rain",
            "negative_prompt": "deformed hands, identity drift, text, watermark",
            "inference": {
                "num_inference_steps": 36,
                "max_sequence_length": 512,
            },
        },
    )
    renderer.render(request)

    call = pipeline.calls[0]
    assert call["negative_prompt"] == (
        "deformed hands, identity drift, text, watermark"
    )
    assert call["guidance_scale"] == 4.5
    assert call["num_inference_steps"] == 36
    assert call["strength"] == 0.75
    assert call["max_sequence_length"] == 512
    assert exports[0][2] == 24.0


def test_renderer_does_not_forward_control_missing_from_pipeline_signature(tmp_path):
    calls = []

    class MinimalPipeline:
        def to(self, _device):
            return self

        def __call__(self, prompt, width, height, num_frames):
            calls.append((prompt, width, height, num_frames))
            return FakeOutput()

    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: MinimalPipeline(),
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")
    renderer.render(
        _request(
            metadata={
                "negative_prompt": "identity drift",
                "inference": {"num_inference_steps": 30},
            }
        )
    )

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("control", "value", "message"),
    [
        ("num_inference_steps", 0, "positive integer"),
        ("max_sequence_length", True, "positive integer"),
        ("guidance_scale", -0.1, "non-negative"),
        ("strength", 1.1, "between 0 and 1"),
        ("negative_prompt", ["bad hands"], "must be a string"),
    ],
)
def test_renderer_rejects_invalid_quality_controls(
    tmp_path, control, value, message
):
    pipeline = QualityAwarePipeline()
    renderer = DiffusersVideoRenderer(
        FoundationProvenance(model_id="declared/model"),
        output_dir=tmp_path,
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=lambda *_args, **_kwargs: None,
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")

    with pytest.raises(DiffusersVideoError, match=message):
        renderer.render(_request(metadata={"inference": {control: value}}))
