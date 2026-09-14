from cineos.atlas.native_request import NativeShotRequest
from cineos.native_image import compile_native_image_plan
from cineos.native_image.temporal_identity import (
    IdentityObservation,
    IdentityVisualQCGate,
    TemporalIdentityMemory,
    apply_temporal_identity_memory,
)


def _plan(shot_id: str = "shot-002"):
    request = NativeShotRequest(
        shot_id=shot_id,
        scene_id="scene-001",
        camera={"resolution": (1920, 1080), "shot_type": "close-up"},
        characters=[
            {
                "character_uuid": "char-001",
                "cinedna_profile_id": "char-001",
                "cinedna_profile_version": "1.1",
                "approved_reference_ids": ["ref-front", "ref-full"],
                "identity_invariants": ["same face"],
                "face_constraints": {},
                "body_constraints": {},
                "scene_specific_overrides": {"primary_reference_id": "ref-front"},
            }
        ],
        environment=None,
        wardrobe=[],
        props=[],
        continuity={"forbidden_changes": ["identity drift"]},
        performance={},
        approved_reference_ids=["ref-front", "ref-full"],
        deterministic_seed=22,
        renderer_requirements={"face_identity_support": True},
    )
    request.refresh_hash()
    return compile_native_image_plan(request)


def test_first_identity_observation_becomes_baseline():
    memory = TemporalIdentityMemory()
    report = IdentityVisualQCGate().evaluate(
        memory,
        IdentityObservation(
            character_uuid="char-001",
            shot_id="shot-001",
            embedding=(1.0, 0.0, 0.0),
            approved_reference_ids=("ref-front",),
        ),
    )

    assert report.decision == "baseline"
    assert report.drift_score == 0.0
    assert memory.latest("char-001").shot_id == "shot-001"


def test_small_identity_drift_is_accepted_and_carried_forward():
    memory = TemporalIdentityMemory()
    gate = IdentityVisualQCGate()
    gate.evaluate(
        memory,
        IdentityObservation(
            character_uuid="char-001",
            shot_id="shot-001",
            embedding=(1.0, 0.0, 0.0),
            approved_reference_ids=("ref-front", "ref-full"),
        ),
    )
    report = gate.evaluate(
        memory,
        IdentityObservation(
            character_uuid="char-001",
            shot_id="shot-002",
            embedding=(0.99, 0.1, 0.0),
            approved_reference_ids=("ref-front", "ref-full"),
        ),
    )

    assert report.accepted is True
    assert report.should_rerender is False
    assert memory.latest("char-001").shot_id == "shot-002"

    plan = apply_temporal_identity_memory(_plan("shot-003"), memory)
    context = plan.metadata["temporal_identity_context"][0]
    assert context["previous_shot_id"] == "shot-002"
    assert context["previous_identity_embedding"] == [0.99, 0.1, 0.0]
    assert len(plan.content_hash) == 64


def test_large_identity_drift_is_rejected_without_poisoning_memory():
    memory = TemporalIdentityMemory()
    gate = IdentityVisualQCGate(reject_drift=0.30)
    gate.evaluate(
        memory,
        IdentityObservation(
            character_uuid="char-001",
            shot_id="shot-001",
            embedding=(1.0, 0.0, 0.0),
        ),
    )
    report = gate.evaluate(
        memory,
        IdentityObservation(
            character_uuid="char-001",
            shot_id="shot-002",
            embedding=(-1.0, 0.0, 0.0),
        ),
    )

    assert report.decision == "reject"
    assert report.should_rerender is True
    assert memory.latest("char-001").shot_id == "shot-001"
    assert len(memory.history("char-001")) == 1
