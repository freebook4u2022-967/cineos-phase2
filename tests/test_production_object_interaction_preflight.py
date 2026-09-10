import pytest

from cineos.atlas.gpu_benchmark_cli import COMPETITIVE_CHALLENGE_METADATA_KEY
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_input_preflight import (
    ProductionInputPreflightError,
    _validate_object_interaction_grounding,
)


def _request() -> NativeShotRequest:
    return NativeShotRequest(
        shot_id="object-shot",
        scene_id="object-scene",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "warehouse"},
        wardrobe=[],
        props=[{"prop_id": "case"}],
        continuity={"previous_shot": None},
        performance={"action": "lead picks up case"},
        approved_reference_ids=["lead-ref"],
        deterministic_seed=8100,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={COMPETITIVE_CHALLENGE_METADATA_KEY: ["object_interaction"]},
    )


def test_object_interaction_accepts_performance_grounded_to_conditioned_prop():
    _validate_object_interaction_grounding([_request()])


def test_object_interaction_rejects_static_prop_presence_without_manipulation():
    request = _request()
    request.performance["action"] = "lead waits beside the case"

    with pytest.raises(ProductionInputPreflightError, match="no explicit performance"):
        _validate_object_interaction_grounding([request])


def test_object_interaction_rejects_manipulation_of_unconditioned_object():
    request = _request()
    request.performance["action"] = "lead throws a bottle"

    with pytest.raises(
        ProductionInputPreflightError, match="conditioned prop identity"
    ):
        _validate_object_interaction_grounding([request])


def test_object_interaction_rejects_prop_without_stable_identity():
    request = _request()
    request.props = [{"description": "metal case"}]

    with pytest.raises(ProductionInputPreflightError, match="stable prop identity"):
        _validate_object_interaction_grounding([request])


def test_object_interaction_accepts_structured_character_prop_cue():
    request = _request()
    request.performance = {
        "action": "lead waits",
        "object_interaction_cues": [
            {
                "character_id": "lead",
                "prop_id": "case",
                "action": "lead opens the case",
            }
        ],
    }

    _validate_object_interaction_grounding([request])


def test_object_interaction_rejects_cue_for_unconditioned_prop():
    request = _request()
    request.performance = {
        "action": "lead waits",
        "object_interaction_cues": [
            {
                "character_id": "lead",
                "prop_id": "bottle",
                "action": "lead opens the bottle",
            }
        ],
    }

    with pytest.raises(
        ProductionInputPreflightError, match="conditioned prop identity"
    ):
        _validate_object_interaction_grounding([request])


def test_object_interaction_rejects_cue_for_unconditioned_character():
    request = _request()
    request.performance = {
        "action": "lead waits",
        "object_interaction_cues": [
            {
                "character_id": "intruder",
                "prop_id": "case",
                "action": "intruder opens the case",
            }
        ],
    }

    with pytest.raises(ProductionInputPreflightError, match="unconditioned character"):
        _validate_object_interaction_grounding([request])


def test_object_interaction_rejects_conflicting_prop_aliases():
    request = _request()
    request.props = [{"prop_uuid": "case", "prop_id": "different-case"}]

    with pytest.raises(
        ProductionInputPreflightError, match="conflicting prop identity"
    ):
        _validate_object_interaction_grounding([request])


def test_non_object_interaction_shot_remains_backward_compatible():
    request = _request()
    request.metadata[COMPETITIVE_CHALLENGE_METADATA_KEY] = ["physics"]
    request.props = []
    request.performance = {}

    _validate_object_interaction_grounding([request])
