from cineos.atlas.native_request import NativeShotRequest
from cineos.native_image import NativeImageResearchBackend, compile_native_image_plan


class StubNativeImageModel:
    def encode_identity(self, tokens):
        return {"identity_count": len(tokens), "primary": tokens[0]["primary_reference_id"]}

    def encode_scene(self, plan):
        return {"size": [plan.width, plan.height], "shot_id": plan.shot_id}

    def generate(self, *, identity_state, scene_state, seed):
        return {
            "pixels": "stub",
            "identity": identity_state,
            "scene": scene_state,
            "seed": seed,
        }


def _request():
    request = NativeShotRequest(
        shot_id="shot-001",
        scene_id="scene-001",
        camera={
            "resolution": (1920, 1080),
            "shot_type": "close-up",
            "lens": "50mm",
            "aspect_ratio": "16:9",
        },
        characters=[
            {
                "character_uuid": "char-001",
                "cinedna_profile_id": "char-001",
                "cinedna_profile_version": "1.1",
                "approved_reference_ids": ["ref-front", "ref-full"],
                "identity_invariants": ["same face", "same silhouette"],
                "face_constraints": {"identity": "locked"},
                "body_constraints": {"build": "locked"},
                "scene_specific_overrides": {
                    "primary_reference_id": "ref-front",
                    "reference_strategy": "ranked-multi-reference",
                },
            }
        ],
        environment={"description": "dark apartment"},
        wardrobe=[],
        props=[],
        continuity={"forbidden_changes": ["identity drift"]},
        performance={"facial_targets": [{"emotion": "unease"}]},
        approved_reference_ids=["ref-front", "ref-full"],
        deterministic_seed=123,
        renderer_requirements={"face_identity_support": True},
    )
    request.refresh_hash()
    return request


def test_native_image_plan_preserves_identity_camera_and_continuity():
    request = _request()
    plan = compile_native_image_plan(request)

    assert plan.width == 1920
    assert plan.height == 1080
    assert plan.identity_tokens[0]["primary_reference_id"] == "ref-front"
    assert plan.identity_tokens[0]["identity_invariants"] == [
        "same face",
        "same silhouette",
    ]
    assert plan.composition_tokens["lens"] == "50mm"
    assert plan.continuity_tokens["forbidden_changes"] == ["identity drift"]
    assert plan.metadata["source_native_request_hash"] == request.content_hash
    assert len(plan.content_hash) == 64


def test_native_image_research_backend_uses_injected_cineos_model_boundary():
    plan = compile_native_image_plan(_request())
    result = NativeImageResearchBackend(StubNativeImageModel()).render(plan)

    assert result.shot_id == "shot-001"
    assert result.plan_hash == plan.content_hash
    assert result.identity_state["primary"] == "ref-front"
    assert result.scene_state["size"] == [1920, 1080]
    assert result.image["seed"] == 123


def test_native_image_plan_rejects_character_without_identity_references():
    request = _request()
    request.characters[0]["approved_reference_ids"] = []
    request.refresh_hash()

    try:
        compile_native_image_plan(request)
    except ValueError as error:
        assert "approved references" in str(error)
    else:
        raise AssertionError("native image plan accepted an unconditioned character")
