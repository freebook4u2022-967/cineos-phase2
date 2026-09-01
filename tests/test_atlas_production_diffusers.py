from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError, FoundationProvenance
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_diffusers import (
    MultiReferenceConditioningResult,
    ProductionDiffusersVideoRenderer,
)


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


class KwargsPipeline:
    def __init__(self):
        self.calls = []

    def to(self, _device):
        return self

    def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
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


def _renderer(
    tmp_path,
    pipeline,
    *,
    reference_loader=None,
    multi_reference_adapter=None,
):
    renderer = ProductionDiffusersVideoRenderer(
        FoundationProvenance(model_id="external/foundation"),
        output_dir=tmp_path,
        reference_loader=reference_loader,
        multi_reference_adapter=multi_reference_adapter,
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


def test_kwargs_pipeline_receives_reference_and_execution_controls(tmp_path):
    pipeline = KwargsPipeline()
    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: f"image:{reference_id}",
    )

    renderer.render(_request())

    assert len(pipeline.calls) == 1
    call = pipeline.calls[0]
    assert call["image"] == "image:hero-front"
    assert call["width"] == 1280
    assert call["height"] == 704
    assert call["num_frames"] == 24
    assert call["generator"] == 7


def test_production_multiple_references_fail_closed_instead_of_using_only_first(
    tmp_path,
):
    pipeline = ImagePipeline()
    loaded_references = []
    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: loaded_references.append(reference_id)
        or f"image:{reference_id}",
    )

    with pytest.raises(DiffusersVideoError, match="multi_reference_adapter"):
        renderer.render(_request(references=("hero-front", "partner-front")))

    assert loaded_references == []
    assert pipeline.calls == []


def test_kwargs_pipeline_cannot_hide_partial_multi_reference_conditioning(tmp_path):
    pipeline = KwargsPipeline()
    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: f"image:{reference_id}",
    )

    with pytest.raises(DiffusersVideoError, match="multi_reference_adapter"):
        renderer.render(_request(references=("hero-front", "partner-front")))

    assert pipeline.calls == []


def test_audited_multi_reference_adapter_consumes_all_references_before_inference(
    tmp_path,
):
    pipeline = ImagePipeline()
    adapter_calls = []

    def adapter(request, references):
        adapter_calls.append((request.shot_id, references))
        return MultiReferenceConditioningResult(
            image="composed:hero+partner",
            consumed_reference_ids=tuple(request.approved_reference_ids),
            adapter_id="cineos.reference-compositor",
            adapter_version="1.0",
        )

    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: f"image:{reference_id}",
        multi_reference_adapter=adapter,
    )

    renderer.render(_request(references=("hero-front", "partner-front")))

    assert adapter_calls == [("shot-001", ("image:hero-front", "image:partner-front"))]
    assert pipeline.calls == ["composed:hero+partner"]


def test_multi_reference_adapter_must_attest_every_reference_in_request_order(tmp_path):
    pipeline = ImagePipeline()

    def adapter(_request, _references):
        return MultiReferenceConditioningResult(
            image="partial-composition",
            consumed_reference_ids=("hero-front",),
            adapter_id="cineos.reference-compositor",
            adapter_version="1.0",
        )

    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: f"image:{reference_id}",
        multi_reference_adapter=adapter,
    )

    with pytest.raises(DiffusersVideoError, match="every approved reference"):
        renderer.render(_request(references=("hero-front", "partner-front")))

    assert pipeline.calls == []


def test_multi_reference_adapter_rejects_unresolved_reference_before_inference(
    tmp_path,
):
    pipeline = ImagePipeline()
    adapter_called = False

    def adapter(request, references):
        nonlocal adapter_called
        adapter_called = True
        return MultiReferenceConditioningResult(
            image="composed",
            consumed_reference_ids=tuple(request.approved_reference_ids),
            adapter_id="cineos.reference-compositor",
            adapter_version="1.0",
        )

    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: (
            None if reference_id == "partner-front" else f"image:{reference_id}"
        ),
        multi_reference_adapter=adapter,
    )

    with pytest.raises(DiffusersVideoError, match="partner-front"):
        renderer.render(_request(references=("hero-front", "partner-front")))

    assert adapter_called is False
    assert pipeline.calls == []


def test_multi_reference_adapter_requires_nonempty_provenance(tmp_path):
    pipeline = ImagePipeline()

    def adapter(request, _references):
        return MultiReferenceConditioningResult(
            image="composed",
            consumed_reference_ids=tuple(request.approved_reference_ids),
            adapter_id="",
            adapter_version="1.0",
        )

    renderer = _renderer(
        tmp_path,
        pipeline,
        reference_loader=lambda reference_id: f"image:{reference_id}",
        multi_reference_adapter=adapter,
    )

    with pytest.raises(DiffusersVideoError, match="adapter_id"):
        renderer.render(_request(references=("hero-front", "partner-front")))

    assert pipeline.calls == []


def test_text_only_production_shot_remains_supported_without_declared_reference(
    tmp_path,
):
    renderer = _renderer(tmp_path, TextOnlyPipeline())

    result = renderer.render(_request(references=()))

    assert Path(result.output_path).read_bytes() == b"video"
