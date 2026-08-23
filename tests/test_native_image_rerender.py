from cineos.native_image.rerender import AutomaticRerenderController, correction_payload
from cineos.native_image.temporal_identity import IdentityObservation, TemporalIdentityMemory
from cineos.native_image.visual_qc import VisualContinuityObservation


def _identity(shot_id: str, embedding: tuple[float, ...]) -> IdentityObservation:
    return IdentityObservation(
        character_uuid="hero",
        shot_id=shot_id,
        embedding=embedding,
        approved_reference_ids=("front",),
    )


def _visual(shot_id: str, face: float = 0.95) -> VisualContinuityObservation:
    return VisualContinuityObservation(
        shot_id=shot_id,
        scores={
            "face_identity": face,
            "body_shape": 0.95,
            "wardrobe": 0.95,
            "hair": 0.95,
            "props": 0.95,
            "environment": 0.95,
            "lighting": 0.95,
            "screen_direction": 0.95,
        },
    )


def test_accepted_shot_commits_identity_state():
    memory = TemporalIdentityMemory()
    controller = AutomaticRerenderController()
    decision = controller.evaluate(
        memory,
        (_identity("shot-1", (1.0, 0.0)),),
        _visual("shot-1"),
    )
    assert decision.decision == "accept"
    assert memory.latest("hero").shot_id == "shot-1"


def test_rejected_shot_requests_rerender_without_polluting_memory():
    memory = TemporalIdentityMemory()
    memory.accept(_identity("shot-1", (1.0, 0.0)))
    controller = AutomaticRerenderController(max_attempts=3)
    decision = controller.evaluate(
        memory,
        (_identity("shot-2", (-1.0, 0.0)),),
        _visual("shot-2", face=0.40),
        attempt=1,
    )
    assert decision.should_rerender is True
    assert memory.latest("hero").shot_id == "shot-1"
    payload = correction_payload(decision)
    assert payload["preserve_last_accepted_identity"] is True
    assert payload["do_not_commit_rejected_state"] is True
    assert "restore approved identity for hero" in payload["directives"]


def test_rerender_budget_exhaustion_escalates_without_committing_state():
    memory = TemporalIdentityMemory()
    memory.accept(_identity("shot-1", (1.0, 0.0)))
    controller = AutomaticRerenderController(max_attempts=2)
    decision = controller.evaluate(
        memory,
        (_identity("shot-2", (-1.0, 0.0)),),
        _visual("shot-2", face=0.30),
        attempt=2,
    )
    assert decision.decision == "reject"
    assert decision.should_rerender is False
    assert memory.latest("hero").shot_id == "shot-1"
    assert any("budget exhausted" in item for item in decision.directives)
