from typing import Any

import pytest

from cineos.atlas import (
    BaseRenderer,
    Range,
    RendererCapabilities,
    RendererSession,
    Resolution,
)
from cineos.atlas.capabilities import CapabilityError
from cineos.atlas.native_ingest import NativeRequestError, ingest_native_request
from cineos.atlas.native_request import compile_native_shot_request
from cineos.conditioning import (
    CameraConditioning,
    CharacterConditioning,
    ConditioningPackage,
    ContinuityConditioning,
    RendererCapabilityRequirements,
)


class NativeStubRenderer(BaseRenderer):
    def __init__(self) -> None:
        self.requests = []

    @property
    def capabilities(self):
        return RendererCapabilities(
            supported_resolution=(Resolution(1920, 1080),),
            supported_duration=Range(1, 10),
            supported_fps=(24,),
            supported_features=frozenset(
                {"image-reference", "multi-reference", "face-identity"}
            ),
            maximum_character_count=2,
        )

    def initialize(self):
        pass

    def load_model(self, model=None, **options: Any):
        pass

    def warmup(self):
        pass

    def render(self, request):
        self.requests.append(request)
        return {"native": True, "shot_id": request.shot_id}

    def shutdown(self):
        pass


def _request(character_count=1):
    characters = [
        CharacterConditioning(
            character_uuid=f"char-{index}",
            cinedna_profile_id=f"char-{index}",
            cinedna_profile_version="1.0",
            approved_reference_ids=[f"ref-{index}-front", f"ref-{index}-full"],
            identity_invariants=["same face"],
        )
        for index in range(character_count)
    ]
    refs = [ref for character in characters for ref in character.approved_reference_ids]
    package = ConditioningPackage(
        shot_id="shot-001",
        scene_id="scene-001",
        character_conditioning=characters,
        environment_conditioning=None,
        wardrobe_conditioning=[],
        prop_conditioning=[],
        camera_conditioning=CameraConditioning(
            resolution=(1920, 1080), fps=24, duration=5.0
        ),
        continuity_constraints=ContinuityConditioning(),
        approved_reference_ids=refs,
        renderer_capability_requirements=RendererCapabilityRequirements(
            image_reference_support=True,
            multi_reference_support=True,
            face_identity_support=True,
            character_count=character_count,
            maximum_duration=5.0,
        ),
        deterministic_seed=99,
    )
    return compile_native_shot_request(package)


def test_atlas_ingests_valid_native_request():
    renderer = NativeStubRenderer()
    request = _request()
    with RendererSession(renderer) as session:
        receipt = ingest_native_request(session, request)
    assert receipt.shot_id == "shot-001"
    assert receipt.request_hash == request.content_hash
    assert receipt.result["native"] is True
    assert renderer.requests == [request]


def test_atlas_rejects_tampered_native_request():
    request = _request()
    request.camera["duration"] = 8.0
    with RendererSession(NativeStubRenderer()) as session:
        with pytest.raises(NativeRequestError, match="content hash mismatch"):
            ingest_native_request(session, request)


def test_atlas_rejects_native_request_over_character_limit():
    request = _request(character_count=3)
    with RendererSession(NativeStubRenderer()) as session:
        with pytest.raises(CapabilityError, match="character count"):
            ingest_native_request(session, request)
