import hashlib
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class FileRenderResult:
    output_path: str


class FileClaimingRenderer(NativeStubRenderer):
    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self.output_path = output_path

    def render(self, request):
        self.requests.append(request)
        return FileRenderResult(str(self.output_path))


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
    assert receipt.artifact_path is None
    assert receipt.artifact_bytes is None
    assert receipt.artifact_sha256 is None
    assert renderer.requests == [request]


def test_atlas_records_content_addressed_evidence_for_claimed_artifact(tmp_path):
    artifact = tmp_path / "shot-001.mp4"
    payload = b"real-render-evidence"
    artifact.write_bytes(payload)
    request = _request()

    with RendererSession(FileClaimingRenderer(artifact)) as session:
        receipt = ingest_native_request(session, request)

    assert receipt.artifact_path == str(artifact)
    assert receipt.artifact_bytes == len(payload)
    assert receipt.artifact_sha256 == hashlib.sha256(payload).hexdigest()


def test_atlas_rejects_claimed_artifact_that_does_not_exist(tmp_path):
    request = _request()
    missing = tmp_path / "missing.mp4"
    with RendererSession(FileClaimingRenderer(missing)) as session:
        with pytest.raises(NativeRequestError, match="missing output artifact"):
            ingest_native_request(session, request)


def test_atlas_rejects_claimed_artifact_that_is_empty(tmp_path):
    request = _request()
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with RendererSession(FileClaimingRenderer(empty)) as session:
        with pytest.raises(NativeRequestError, match="empty output artifact"):
            ingest_native_request(session, request)


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
