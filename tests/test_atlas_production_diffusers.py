from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError, FoundationProvenance
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_diffusers import ProductionDiffusersVideoRenderer


class FakeOutput:
    frames = [["f1", "f2"]]


class ImagePipeline:
    def __init__(self):
        self.calls = []

    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames, image=None):
        self.calls.append(image)
        return FakeOutput()


class TextOnlyPipeline:
    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames):
        return FakeOutput()


def _request(*, references=("hero-front",)):
    request = NativeShotRequest(
        shot_id="shot-001",
        scene_id="scene-001",
        camera={"resolution": (1280, 704), "fps": 24, "duration": 1.0},
        characters=[],
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=list(references),
        deterministic_seed=7,
        renderer_requirements={},
        metadata={"prompt": "hero walks toward camera"},
    )
    request.refresh_hash()
    return request


def _renderer(tmp_path, pipeline, *, reference_loader=None):
    renderer = ProductionDiffusersVideoRenderer(
        FoundationProvenance(model_id="external/foundation"),
        output_dir=tmp_path,
        reference_loader=reference_loader,
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=lambda _frames, path, *, fps: Path(path).write_bytes(b"video"),
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")
    return renderer


def test_pinned_foundation_uses_strict_production_renderer(tmp_path):
    renderer = WAN22_TI2V_5B_PROFILE.renderer(output_dir=tmp_path)
    assert isinstance(renderer, ProductionDiffusersVideoRenderer)


def test_production_reference_requires_loader(tmp_path):
    renderer = _renderer(tmp_path, ImagePipeline())

    with pytest.raises(DiffusersVideoError, match="no reference_loader"):
        renderer.render(_request())


def test_production_reference_requires_foundation_image_input(tmp_path):
    renderer = _renderer(
        tmp_path,
        TextOnlyPipeline(),
        reference_loader=lambda _reference_id: "resolved-image",
    )

    with pytest.raises(DiffusersVideoError, match="does not expose image conditioning"):
        renderer.render(_request())


def test_production_reference_must_resolve_to_an_image(tmp_path):
    renderer = _renderer(
        tmp_path,
        ImagePipeline(),
        reference_loader=lambda _reference_id: None,
    )

    with pytest.raises(DiffusersVideoError, match="could not be resolved"):
        renderer.render(_request())


def test_production_reference_reaches_foundation_pipeline(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: f"image:{reference_id}",
    )

    renderer.render(_request())

    assert pipeline.calls == ["image:hero-front"]


def test_text_only_production_shot_remains_supported_without_declared_reference(
    tmp_path,
):
    renderer = _renderer(tmp_path, TextOnlyPipeline())

    result = renderer.render(_request(references=()))

    assert Path(result.output_path).read_bytes() == b"video"
