from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError, FoundationProvenance
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_continuity_diffusers import (
    ProductionContinuityDiffusersVideoRenderer,
    VISUAL_CONTINUITY_SCHEMA,
)


class FakeOutput:
    def __init__(self, frames):
        self.frames = [frames]


class ImagePipeline:
    def __init__(self):
        self.calls = []

    def to(self, _device):
        return self

    def __call__(self, prompt, width, height, num_frames, image=None):
        index = len(self.calls)
        self.calls.append(image)
        return FakeOutput([f"shot-{index}-first", f"shot-{index}-terminal"])


def _request(shot_id, *, previous_shot=None, references=("hero-front",)):
    request = NativeShotRequest(
        shot_id=shot_id,
        scene_id="scene-001",
        camera={"resolution": (1280, 704), "fps": 24, "duration": 1.0},
        characters=[{"character_id": "hero"}],
        environment={},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": previous_shot},
        performance={},
        approved_reference_ids=list(references),
        deterministic_seed=7,
        renderer_requirements={},
        metadata={"prompt": f"connected {shot_id}"},
    )
    request.refresh_hash()
    return request


def _renderer(tmp_path, pipeline):
    renderer = ProductionContinuityDiffusersVideoRenderer(
        FoundationProvenance(model_id="external/foundation"),
        output_dir=tmp_path,
        reference_loader=lambda reference_id: f"image:{reference_id}",
        pipeline_factory=lambda *_args, **_kwargs: pipeline,
        video_exporter=lambda _frames, path, *, fps: Path(path).write_bytes(b"video"),
    )
    renderer.initialize()
    renderer.load_model(device="cpu", dtype="float32")
    return renderer


def test_pinned_foundation_enables_visual_continuity_renderer(tmp_path):
    renderer = WAN22_TI2V_5B_PROFILE.renderer(output_dir=tmp_path)
    assert isinstance(renderer, ProductionContinuityDiffusersVideoRenderer)


def test_connected_shot_consumes_predecessor_terminal_frame(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)

    renderer.render(_request("shot-001"))
    first_provenance = renderer.last_conditioning_provenance
    renderer.render(_request("shot-002", previous_shot="shot-001"))
    second_provenance = renderer.last_conditioning_provenance

    assert pipeline.calls == ["image:hero-front", "shot-0-terminal"]
    assert first_provenance["schema"] == VISUAL_CONTINUITY_SCHEMA
    assert first_provenance["mode"] == "approved_reference_root"
    assert second_provenance["mode"] == "predecessor_terminal_frame_lineage"
    assert second_provenance["previous_shot_id"] == "shot-001"
    assert second_provenance["in_memory_terminal_frame"] is True


def test_continuation_fails_closed_without_predecessor_in_same_session(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)

    with pytest.raises(DiffusersVideoError, match="predecessor frame is unavailable"):
        renderer.render(_request("shot-002", previous_shot="shot-001"))

    assert pipeline.calls == []


def test_continuation_rejects_identity_reference_change(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    renderer.render(_request("shot-001"))

    with pytest.raises(DiffusersVideoError, match="differ from predecessor lineage"):
        renderer.render(
            _request(
                "shot-002",
                previous_shot="shot-001",
                references=("different-hero",),
            )
        )

    assert pipeline.calls == ["image:hero-front"]


def test_retry_of_same_shot_still_anchors_to_declared_predecessor(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    renderer.render(_request("shot-001"))
    renderer.render(_request("shot-002", previous_shot="shot-001"))
    renderer.render(_request("shot-002", previous_shot="shot-001"))

    assert pipeline.calls == [
        "image:hero-front",
        "shot-0-terminal",
        "shot-0-terminal",
    ]


def test_conflicting_legacy_and_canonical_predecessor_rejected(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    request = _request("shot-002", previous_shot="shot-001")
    request.continuity["previous_shot_id"] = "other-shot"
    request.refresh_hash()

    with pytest.raises(DiffusersVideoError, match="conflicting previous_shot"):
        renderer.render(request)

    assert pipeline.calls == []
