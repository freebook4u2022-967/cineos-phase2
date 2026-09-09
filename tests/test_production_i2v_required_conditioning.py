from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError, FoundationProvenance
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
    def __init__(self):
        self.calls = 0

    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames):
        self.calls += 1
        return FakeOutput()


def _request(*, references=()):
    request = NativeShotRequest(
        shot_id="shot-i2v",
        scene_id="scene-i2v",
        camera={"resolution": (1280, 720), "fps": 16, "duration": 1.0},
        characters=[],
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=list(references),
        deterministic_seed=11,
        renderer_requirements={},
        metadata={"prompt": "hero crosses the room"},
    )
    request.refresh_hash()
    return request


def _renderer(
    tmp_path,
    pipeline,
    *,
    features,
    reference_loader=None,
):
    renderer = ProductionDiffusersVideoRenderer(
        FoundationProvenance(model_id="external/foundation"),
        output_dir=tmp_path,
        resolutions=((1280, 720),),
        fps=(16.0,),
        supported_features=frozenset(features),
        reference_loader=reference_loader,
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=lambda _frames, path, *, fps: Path(path).write_bytes(b"video"),
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")
    return renderer


def test_i2v_only_foundation_rejects_root_shot_without_image(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline, features={"image_to_video"})

    with pytest.raises(DiffusersVideoError, match="no resolved image conditioning source"):
        renderer.render(_request())

    assert pipeline.calls == []


def test_i2v_only_foundation_rejects_pipeline_without_image_input(tmp_path):
    pipeline = TextOnlyPipeline()
    renderer = _renderer(tmp_path, pipeline, features={"image_to_video"})

    with pytest.raises(DiffusersVideoError, match="does not expose image conditioning"):
        renderer.render(_request())

    assert pipeline.calls == 0


def test_i2v_only_foundation_consumes_resolved_approved_image(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(
        tmp_path,
        pipeline,
        features={"image_to_video"},
        reference_loader=lambda reference_id: f"image:{reference_id}",
    )

    result = renderer.render(_request(references=("hero-front",)))

    assert pipeline.calls == ["image:hero-front"]
    assert result.conditioning_provenance["mode"] == "single_reference"
    assert result.conditioning_provenance["consumed_reference_ids"] == ["hero-front"]


def test_hybrid_ti2v_foundation_preserves_text_only_root_compatibility(tmp_path):
    pipeline = TextOnlyPipeline()
    renderer = _renderer(
        tmp_path,
        pipeline,
        features={"text_to_video", "image_to_video"},
    )

    result = renderer.render(_request())

    assert pipeline.calls == 1
    assert Path(result.output_path).read_bytes() == b"video"
    assert result.conditioning_provenance is None
