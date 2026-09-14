import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_multi_reference import (
    ProductionMultiReferenceError,
    ProductionReferenceBoardAdapter,
)


def _request(reference_ids, characters, *, refresh_hash=True):
    request = NativeShotRequest(
        shot_id="shot-lineage",
        scene_id="scene-lineage",
        camera={"resolution": (1280, 704), "fps": 24, "duration": 1.0},
        characters=list(characters),
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=list(reference_ids),
        deterministic_seed=17,
        renderer_requirements={},
    )
    if refresh_hash:
        request.refresh_hash()
    return request


class _ExplodingImage:
    def convert(self, _mode):
        raise AssertionError("identity lineage must fail before image processing")

    def resize(self, _size, _resample):
        raise AssertionError("identity lineage must fail before image processing")


def _images(count):
    return tuple(_ExplodingImage() for _ in range(count))


def test_adapter_rejects_unowned_approved_reference_before_composition():
    adapter = ProductionReferenceBoardAdapter()
    request = _request(
        ("alice-ref", "bob-ref"),
        (
            {
                "character_uuid": "alice",
                "approved_reference_ids": ["alice-ref"],
            },
        ),
    )

    with pytest.raises(ProductionMultiReferenceError, match="unowned: bob-ref"):
        adapter(request, _images(2))


def test_adapter_rejects_reference_escaping_shot_approval_before_composition():
    adapter = ProductionReferenceBoardAdapter()
    request = _request(
        ("alice-ref", "bob-ref"),
        (
            {
                "character_uuid": "alice",
                "approved_reference_ids": ["alice-ref", "not-approved"],
            },
            {
                "character_uuid": "bob",
                "approved_reference_ids": ["bob-ref"],
            },
        ),
        refresh_hash=False,
    )

    with pytest.raises(ProductionMultiReferenceError, match="not approved by the shot"):
        adapter(request, _images(2))


def test_adapter_rejects_ambiguous_character_reference_owner_before_composition():
    adapter = ProductionReferenceBoardAdapter()
    request = _request(
        ("shared-ref", "bob-ref"),
        (
            {
                "character_uuid": "alice",
                "approved_reference_ids": ["shared-ref"],
            },
            {
                "character_uuid": "bob",
                "approved_reference_ids": ["shared-ref", "bob-ref"],
            },
        ),
        refresh_hash=False,
    )

    with pytest.raises(ProductionMultiReferenceError, match="ambiguously assigned"):
        adapter(request, _images(2))


def test_adapter_rejects_multi_character_missing_stable_identity_before_composition():
    adapter = ProductionReferenceBoardAdapter()
    request = _request(
        ("alice-ref", "bob-ref"),
        (
            {"approved_reference_ids": ["alice-ref"]},
            {
                "character_uuid": "bob",
                "approved_reference_ids": ["bob-ref"],
            },
        ),
    )

    with pytest.raises(ProductionMultiReferenceError, match="non-empty character_uuid"):
        adapter(request, _images(2))
