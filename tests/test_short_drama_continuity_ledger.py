import pytest

from cineos.short_drama.continuity_ledger import ContinuityLedger
from cineos.short_drama.models import SceneState


def _scene(
    index: int,
    *,
    wardrobe: str = "black coat",
    physical_state: str = "uninjured",
    props=None,
    environment=None,
):
    return SceneState(
        scene_index=index,
        location="warehouse",
        time_of_day="night",
        weather="rain",
        characters={
            "arif": {
                "wardrobe": wardrobe,
                "physical_state": physical_state,
                "props": ["phone"] if props is None else props,
            }
        },
        environment={"vehicle": "black sedan"} if environment is None else environment,
    )


def test_continuity_ledger_accepts_stable_state_and_round_trips():
    ledger = ContinuityLedger()
    ledger.append(_scene(1))
    ledger.append(_scene(2))

    restored = ContinuityLedger.from_dict(ledger.to_dict())

    assert restored.to_dict() == ledger.to_dict()
    assert restored.latest().scene_index == 2


def test_continuity_ledger_rejects_silent_character_change():
    ledger = ContinuityLedger([_scene(1)])

    violations = ledger.validate(_scene(2, wardrobe="white shirt"))

    assert len(violations) == 1
    assert violations[0].scope == "character:arif"
    assert violations[0].key == "wardrobe"
    with pytest.raises(ValueError, match="continuity validation failed"):
        ledger.append(_scene(2, wardrobe="white shirt"))


def test_continuity_ledger_allows_scripted_character_transition():
    ledger = ContinuityLedger([_scene(1)])

    ledger.append(
        _scene(2, physical_state="injured"),
        allowed_character_changes={"arif.physical_state"},
    )

    assert ledger.latest().characters["arif"]["physical_state"] == "injured"


def test_continuity_ledger_rejects_silent_environment_change():
    ledger = ContinuityLedger([_scene(1)])
    next_scene = _scene(2, environment={"vehicle": "red van"})

    violations = ledger.validate(next_scene)

    assert len(violations) == 1
    assert violations[0].scope == "environment"
    assert violations[0].key == "vehicle"


def test_continuity_ledger_requires_monotonic_scene_indices():
    ledger = ContinuityLedger([_scene(2)])

    with pytest.raises(ValueError, match="increase monotonically"):
        ledger.validate(_scene(2))


def test_continuity_ledger_rejects_unknown_checkpoint_version():
    with pytest.raises(ValueError, match="unsupported continuity ledger version"):
        ContinuityLedger.from_dict({"version": 2, "scenes": []})
