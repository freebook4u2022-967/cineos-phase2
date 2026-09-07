import pytest

from cineos.atlas.native_request import (
    NATIVE_SHOT_SCHEMA,
    NativeShotRequest,
    compile_native_shot_request,
)
from cineos.conditioning import (
    CameraConditioning,
    CharacterConditioning,
    ConditioningPackage,
    ContinuityConditioning,
    RendererCapabilityRequirements,
)


def _package():
    return ConditioningPackage(
        shot_id="shot-001",
        scene_id="scene-001",
        character_conditioning=[
            CharacterConditioning(
                character_uuid="char-001",
                cinedna_profile_id="char-001",
                cinedna_profile_version="1.0",
                approved_reference_ids=["ref-front", "ref-full"],
                identity_invariants=["same face", "same silhouette"],
                face_constraints={"identity": "locked"},
                body_constraints={"build": "locked"},
            )
        ],
        environment_conditioning=None,
        wardrobe_conditioning=[],
        prop_conditioning=[],
        camera_conditioning=CameraConditioning(
            shot_type="close-up",
            lens="50mm",
            camera_movement="slow push-in",
            aspect_ratio="16:9",
            duration=5.0,
        ),
        continuity_constraints=ContinuityConditioning(
            forbidden_changes=["identity drift"],
            required_carry_over_references=["ref-front"],
        ),
        approved_reference_ids=["ref-front", "ref-full"],
        renderer_capability_requirements=RendererCapabilityRequirements(
            image_reference_support=True,
            multi_reference_support=True,
            face_identity_support=True,
            character_count=1,
            maximum_duration=5.0,
        ),
        deterministic_seed=1234,
        performance_package_id="performance-001",
        dialogue_timing=[{"start": 0.5, "end": 2.0, "text": "Where are you?"}],
        facial_targets=[{"character_id": "char-001", "emotion": "unease"}],
    )


def _dialogue_request(*, dialogue_timing, metadata=None):
    return NativeShotRequest(
        shot_id="shot-dialogue",
        scene_id="scene-dialogue",
        camera={},
        characters=[{"character_id": "char-001"}],
        environment=None,
        wardrobe=[],
        props=[],
        continuity={},
        performance={"dialogue_timing": dialogue_timing},
        approved_reference_ids=["ref-front"],
        deterministic_seed=7,
        renderer_requirements={"duration_seconds": 5.0, "fps": 24.0},
        metadata=metadata or {"benchmark_challenges": ["dialogue"]},
    )


def test_conditioning_compiles_to_cineos_native_shot_request():
    request = compile_native_shot_request(_package())
    payload = request.to_dict()

    assert payload["schema"] == NATIVE_SHOT_SCHEMA
    assert payload["shot_id"] == "shot-001"
    assert payload["camera"]["lens"] == "50mm"
    assert payload["characters"][0]["approved_reference_ids"] == [
        "ref-front",
        "ref-full",
    ]
    assert payload["continuity"]["forbidden_changes"] == ["identity drift"]
    assert payload["performance"]["performance_package_id"] == "performance-001"
    assert payload["renderer_requirements"]["multi_reference_support"] is True
    assert len(payload["content_hash"]) == 64


def test_native_shot_hash_is_deterministic():
    first = compile_native_shot_request(_package())
    second = compile_native_shot_request(_package())
    assert first.content_hash == second.content_hash


def test_native_shot_rejects_unconditioned_request():
    package = _package()
    package.character_conditioning = []
    package.approved_reference_ids = []

    try:
        compile_native_shot_request(package)
    except ValueError as error:
        assert "approved conditioning references" in str(error)
    else:
        raise AssertionError("native request accepted an unconditioned shot")


def test_seedance_dialogue_challenge_rejects_empty_dialogue_timing():
    request = _dialogue_request(dialogue_timing=[])

    with pytest.raises(
        ValueError, match="requires non-empty performance.dialogue_timing"
    ):
        request.validate_timing_integrity()


def test_seedance_dialogue_challenge_requires_grounded_speaker():
    request = _dialogue_request(
        dialogue_timing=[{"start": 0.5, "end": 2.0, "text": "Where are you?"}]
    )

    with pytest.raises(ValueError, match="requires speaker_id"):
        request.validate_timing_integrity()


def test_seedance_dialogue_challenge_rejects_unconditioned_speaker():
    request = _dialogue_request(
        dialogue_timing=[
            {
                "start": 0.5,
                "end": 2.0,
                "speaker_id": "char-other",
                "text": "Where are you?",
            }
        ]
    )

    with pytest.raises(ValueError, match="does not match a conditioned character_id"):
        request.validate_timing_integrity()


def test_seedance_dialogue_challenge_accepts_grounded_dialogue():
    request = _dialogue_request(
        dialogue_timing=[
            {
                "start": 0.5,
                "end": 2.0,
                "speaker_id": "char-001",
                "text": "Where are you?",
            }
        ]
    )

    request.validate_timing_integrity()


def test_legacy_dialogue_lipsync_challenge_still_requires_grounding():
    request = _dialogue_request(
        dialogue_timing=[],
        metadata={"competitive_challenges": ["dialogue_lip_sync"]},
    )

    with pytest.raises(
        ValueError, match="requires non-empty performance.dialogue_timing"
    ):
        request.validate_timing_integrity()
