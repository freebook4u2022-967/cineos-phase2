from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError, FoundationProvenance
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_continuity_diffusers import (
    VISUAL_CONTINUITY_SCHEMA,
    ProductionContinuityDiffusersVideoRenderer,
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

    first_result = renderer.render(_request("shot-001"))
    first_provenance = renderer.last_conditioning_provenance
    second_result = renderer.render(_request("shot-002", previous_shot="shot-001"))
    second_provenance = renderer.last_conditioning_provenance

    assert pipeline.calls == ["image:hero-front", "shot-0-terminal"]
    assert first_provenance["schema"] == VISUAL_CONTINUITY_SCHEMA
    assert first_provenance["mode"] == "approved_reference_root"
    assert first_provenance["current_artifact_sha256"] == first_result.artifact_sha256
    assert first_provenance["current_request_hash"] == first_result.request_hash
    assert first_provenance["identity_conditioning"]["mode"] == "single_reference"
    assert second_provenance["mode"] == "predecessor_terminal_frame_lineage"
    assert second_provenance["previous_shot_id"] == "shot-001"
    assert second_provenance["predecessor_artifact_sha256"] == (
        first_result.artifact_sha256
    )
    assert second_provenance["predecessor_request_hash"] == first_result.request_hash
    assert second_provenance["current_artifact_sha256"] == second_result.artifact_sha256
    assert second_provenance["current_request_hash"] == second_result.request_hash
    assert second_provenance["in_memory_terminal_frame"] is True
    assert second_provenance["identity_conditioning"] == {
        "mode": "predecessor_terminal_frame_identity_lineage",
        "inherited_reference_ids": ["hero-front"],
        "identity_signal_source": "predecessor_terminal_frame",
        "fresh_reference_pixels_consumed": False,
    }


def test_returned_result_carries_artifact_bound_continuity_provenance(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)

    first_result = renderer.render(_request("shot-001"))
    second_result = renderer.render(_request("shot-002", previous_shot="shot-001"))

    provenance = second_result.conditioning_provenance
    assert provenance is not None
    assert provenance == renderer.last_conditioning_provenance
    assert provenance["schema"] == VISUAL_CONTINUITY_SCHEMA
    assert provenance["predecessor_artifact_sha256"] == first_result.artifact_sha256
    assert provenance["predecessor_request_hash"] == first_result.request_hash
    assert provenance["current_artifact_sha256"] == second_result.artifact_sha256
    assert provenance["current_request_hash"] == second_result.request_hash


def test_quality_rejected_result_is_evicted_from_continuity_state(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)

    rejected = renderer.render(_request("shot-001"))
    renderer.discard_quality_rejected_result(rejected)

    assert renderer.last_conditioning_provenance is None
    with pytest.raises(DiffusersVideoError, match="predecessor frame is unavailable"):
        renderer.render(_request("shot-002", previous_shot="shot-001"))
    assert pipeline.calls == ["image:hero-front"]


def test_quality_rejection_requires_exact_cached_artifact_binding(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    result = renderer.render(_request("shot-001"))

    tampered = result.__class__(
        shot_id=result.shot_id,
        scene_id=result.scene_id,
        output_path=result.output_path,
        frame_count=result.frame_count,
        seed=result.seed,
        foundation=result.foundation,
        request_hash=result.request_hash,
        artifact_sha256="0" * 64,
        artifact_size_bytes=result.artifact_size_bytes,
        conditioning_provenance=result.conditioning_provenance,
    )
    with pytest.raises(DiffusersVideoError, match="does not match"):
        renderer.discard_quality_rejected_result(tampered)

    renderer.render(_request("shot-002", previous_shot="shot-001"))
    assert pipeline.calls == ["image:hero-front", "shot-0-terminal"]


def test_continuation_fails_closed_without_predecessor_in_same_session(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)

    with pytest.raises(DiffusersVideoError, match="predecessor frame is unavailable"):
        renderer.render(_request("shot-002", previous_shot="shot-001"))

    assert pipeline.calls == []


def test_continuation_fails_closed_if_predecessor_render_binding_is_missing(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    renderer.render(_request("shot-001"))
    renderer._render_bindings.clear()

    with pytest.raises(DiffusersVideoError, match="render binding is unavailable"):
        renderer.render(_request("shot-002", previous_shot="shot-001"))

    assert pipeline.calls == ["image:hero-front"]


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


def test_continuation_rechecks_character_reference_authorization(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    root = _request("shot-001")
    root.characters = [
        {
            "character_uuid": "hero",
            "approved_reference_ids": ["hero-front"],
        }
    ]
    root.refresh_hash()
    renderer.render(root)

    continuation = _request("shot-002", previous_shot="shot-001")
    continuation.characters = [
        {
            "character_uuid": "hero",
            "approved_reference_ids": ["unapproved-side-profile"],
        }
    ]
    continuation.refresh_hash()

    with pytest.raises(DiffusersVideoError, match="not approved by the shot"):
        renderer.render(continuation)

    assert pipeline.calls == ["image:hero-front"]


def test_retry_of_same_shot_still_anchors_to_declared_predecessor(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    first_result = renderer.render(_request("shot-001"))
    renderer.render(_request("shot-002", previous_shot="shot-001"))
    retried = renderer.render(_request("shot-002", previous_shot="shot-001"))

    assert pipeline.calls == [
        "image:hero-front",
        "shot-0-terminal",
        "shot-0-terminal",
    ]
    provenance = retried.conditioning_provenance
    assert provenance is not None
    assert provenance["predecessor_artifact_sha256"] == first_result.artifact_sha256
    assert provenance["predecessor_request_hash"] == first_result.request_hash


def test_conflicting_legacy_and_canonical_predecessor_rejected(tmp_path):
    pipeline = ImagePipeline()
    renderer = _renderer(tmp_path, pipeline)
    request = _request("shot-002", previous_shot="shot-001")
    request.continuity["previous_shot_id"] = "other-shot"
    request.refresh_hash()

    with pytest.raises(DiffusersVideoError, match="conflicting previous_shot"):
        renderer.render(request)

    assert pipeline.calls == []
