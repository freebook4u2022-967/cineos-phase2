from types import SimpleNamespace

import pytest

from cineos.atlas.production_diffusers import (
    DiffusersVideoError,
    MultiReferenceConditioningResult,
    ProductionDiffusersVideoRenderer,
)


def _request(*, multi_character: bool = True):
    characters = [
        {
            "character_uuid": "alice",
            "approved_reference_ids": ["alice-ref"],
        }
    ]
    approved = ("alice-ref", "alice-alt")
    if multi_character:
        characters.append(
            {
                "character_uuid": "bob",
                "approved_reference_ids": ["bob-ref"],
            }
        )
        approved = ("alice-ref", "bob-ref")
    else:
        characters[0]["approved_reference_ids"] = list(approved)
    return SimpleNamespace(
        approved_reference_ids=approved,
        characters=characters,
        shot_id="identity-binding-shot",
    )


def _renderer(adapter):
    renderer = object.__new__(ProductionDiffusersVideoRenderer)
    renderer.multi_reference_adapter = adapter
    renderer.reference_loader = lambda reference_id: f"decoded:{reference_id}"
    renderer._conditioning_provenance = None
    return renderer


def _result(bindings):
    return MultiReferenceConditioningResult(
        image="composed-conditioning",
        consumed_reference_ids=("alice-ref", "bob-ref"),
        adapter_id="cineos-test-compositor",
        adapter_version="1",
        consumed_character_reference_ids=bindings,
    )


def test_multi_character_adapter_attests_exact_identity_slots():
    expected = (("alice", ("alice-ref",)), ("bob", ("bob-ref",)))
    renderer = _renderer(lambda request, refs: _result(expected))

    image = renderer._prepare_multi_reference_image(_request())

    assert image == "composed-conditioning"
    assert renderer._conditioning_provenance["consumed_character_reference_ids"] == [
        {"character_uuid": "alice", "reference_ids": ["alice-ref"]},
        {"character_uuid": "bob", "reference_ids": ["bob-ref"]},
    ]


def test_multi_character_adapter_rejects_swapped_identity_slots():
    swapped = (("alice", ("bob-ref",)), ("bob", ("alice-ref",)))
    renderer = _renderer(lambda request, refs: _result(swapped))

    with pytest.raises(DiffusersVideoError, match="character identity binding"):
        renderer._prepare_multi_reference_image(_request())


def test_multi_character_adapter_rejects_missing_identity_slot_attestation():
    renderer = _renderer(lambda request, refs: _result(None))

    with pytest.raises(DiffusersVideoError, match="must attest exact"):
        renderer._prepare_multi_reference_image(_request())


def test_single_character_legacy_adapter_remains_compatible():
    def adapter(request, refs):
        return MultiReferenceConditioningResult(
            image="composed-conditioning",
            consumed_reference_ids=("alice-ref", "alice-alt"),
            adapter_id="legacy-single-character-compositor",
            adapter_version="1",
        )

    renderer = _renderer(adapter)

    assert renderer._prepare_multi_reference_image(_request(multi_character=False)) == (
        "composed-conditioning"
    )
