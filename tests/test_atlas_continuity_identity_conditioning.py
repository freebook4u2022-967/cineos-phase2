from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.atlas.diffusers_video import FoundationProvenance
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_continuity_identity import (
    CONTINUITY_IDENTITY_ADAPTER_ID,
    ProductionContinuityIdentityDiffusersVideoRenderer,
    compose_continuity_identity_board,
)

Image = pytest.importorskip("PIL.Image")


def _request(
    shot_id: str,
    *,
    previous_shot: str | None = None,
    reference_ids: list[str] | None = None,
) -> NativeShotRequest:
    continuity = {}
    if previous_shot is not None:
        continuity["previous_shot"] = previous_shot
    request = NativeShotRequest(
        shot_id=shot_id,
        scene_id="scene-1",
        camera={"resolution": (16, 8), "fps": 2.0, "duration": 1.0},
        characters=[],
        environment=None,
        wardrobe=[],
        props=[],
        continuity=continuity,
        performance={},
        approved_reference_ids=list(reference_ids or ["hero"]),
        deterministic_seed=123,
        renderer_requirements={},
        metadata={"prompt": "cinematic connected shot"},
    )
    request.refresh_hash()
    return request


def test_continuity_identity_board_preserves_predecessor_and_single_reference() -> None:
    predecessor = Image.new("RGB", (16, 8), (0, 0, 255))
    reference = Image.new("RGB", (8, 8), (255, 0, 0))

    result = compose_continuity_identity_board(
        _request("shot-2", previous_shot="shot-1"),
        predecessor,
        (reference,),
    )

    assert result.predecessor_frame_consumed is True
    assert result.consumed_reference_ids == ("hero",)
    assert result.adapter_id == CONTINUITY_IDENTITY_ADAPTER_ID
    assert result.image.size == (16, 8)
    assert result.image.getpixel((4, 4))[2] > 200
    assert result.image.getpixel((14, 4))[0] > 200


def test_continuity_identity_board_rejects_duplicate_multi_character_content() -> None:
    duplicate = Image.new("RGB", (8, 8), (200, 100, 50))
    request = _request(
        "shot-2",
        previous_shot="shot-1",
        reference_ids=["hero", "partner"],
    )

    with pytest.raises(Exception, match="distinct reference image content"):
        compose_continuity_identity_board(
            request, duplicate, (duplicate, duplicate.copy())
        )


class _Pipeline:
    def __init__(self) -> None:
        self.images: list[object] = []
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        image=None,
        **kwargs,
    ):
        self.images.append(image.copy() if hasattr(image, "copy") else image)
        self.calls += 1
        terminal = (
            Image.new("RGB", (width, height), (0, 0, 255))
            if self.calls == 1
            else Image.new("RGB", (width, height), (0, 255, 0))
        )
        return SimpleNamespace(frames=[[terminal, terminal.copy()]])


def test_connected_renderer_consumes_fresh_identity_pixels_with_predecessor(
    tmp_path,
) -> None:
    pipeline = _Pipeline()
    reference = Image.new("RGB", (8, 8), (255, 0, 0))
    load_calls: list[str] = []

    def reference_loader(reference_id: str):
        load_calls.append(reference_id)
        return reference.copy()

    def pipeline_factory(model_id: str, **kwargs):
        assert model_id == "test/video-foundation"
        return pipeline

    def exporter(frames, output_path: str, fps: float):
        with open(output_path, "wb") as artifact:
            artifact.write(b"fresh-video-artifact")

    renderer = ProductionContinuityIdentityDiffusersVideoRenderer(
        FoundationProvenance(model_id="test/video-foundation"),
        output_dir=tmp_path,
        resolutions=((16, 8),),
        duration_range=(1.0, 1.0),
        fps=(2.0,),
        reference_loader=reference_loader,
        continuity_identity_adapter=compose_continuity_identity_board,
        pipeline_factory=pipeline_factory,
        video_exporter=exporter,
    )
    renderer.initialize()
    renderer.load_model()

    root = renderer.render(_request("shot-1"))
    connected = renderer.render(_request("shot-2", previous_shot="shot-1"))

    assert root.artifact_sha256
    assert connected.artifact_sha256
    assert load_calls == ["hero", "hero"]
    assert len(pipeline.images) == 2

    combined = pipeline.images[1]
    assert combined.size == (16, 8)
    assert combined.getpixel((4, 4))[2] > 200
    assert combined.getpixel((14, 4))[0] > 200

    provenance = connected.conditioning_provenance
    assert provenance is not None
    identity = provenance["identity_conditioning"]
    assert identity["mode"] == "predecessor_terminal_frame_plus_fresh_references"
    assert identity["fresh_reference_pixels_consumed"] is True
    assert identity["predecessor_frame_consumed"] is True
    assert identity["consumed_reference_ids"] == ["hero"]
