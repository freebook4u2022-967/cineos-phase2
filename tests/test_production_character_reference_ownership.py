from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError
from cineos.atlas.production_diffusers import ProductionDiffusersVideoRenderer


def _request(
    *,
    approved_reference_ids: tuple[str, ...],
    characters: tuple[dict, ...],
):
    return SimpleNamespace(
        approved_reference_ids=approved_reference_ids,
        characters=characters,
    )


def test_multi_character_reference_ownership_accepts_disjoint_approved_refs() -> None:
    request = _request(
        approved_reference_ids=("alice-ref", "bob-ref"),
        characters=(
            {"character_uuid": "alice", "approved_reference_ids": ["alice-ref"]},
            {"character_uuid": "bob", "approved_reference_ids": ["bob-ref"]},
        ),
    )

    ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_multi_character_reference_ownership_rejects_shared_identity_ref() -> None:
    request = _request(
        approved_reference_ids=("alice-ref", "bob-ref"),
        characters=(
            {"character_uuid": "alice", "approved_reference_ids": ["alice-ref"]},
            {"character_uuid": "bob", "approved_reference_ids": ["alice-ref"]},
        ),
    )

    with pytest.raises(
        DiffusersVideoError,
        match="ambiguously assigned to multiple characters",
    ):
        ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_multi_character_reference_ownership_requires_each_character_ref() -> None:
    request = _request(
        approved_reference_ids=("alice-ref",),
        characters=(
            {"character_uuid": "alice", "approved_reference_ids": ["alice-ref"]},
            {"character_uuid": "bob", "approved_reference_ids": []},
        ),
    )

    with pytest.raises(
        DiffusersVideoError,
        match="requires at least one approved reference per character",
    ):
        ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_multi_character_reference_ownership_rejects_duplicate_character_uuid() -> None:
    request = _request(
        approved_reference_ids=("alice-a", "alice-b"),
        characters=(
            {"character_uuid": "alice", "approved_reference_ids": ["alice-a"]},
            {"character_uuid": " alice ", "approved_reference_ids": ["alice-b"]},
        ),
    )

    with pytest.raises(
        DiffusersVideoError,
        match="requires unique character_uuid values",
    ):
        ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_multi_character_reference_ownership_requires_character_uuid() -> None:
    request = _request(
        approved_reference_ids=("alice-ref", "bob-ref"),
        characters=(
            {"character_uuid": "alice", "approved_reference_ids": ["alice-ref"]},
            {"approved_reference_ids": ["bob-ref"]},
        ),
    )

    with pytest.raises(
        DiffusersVideoError,
        match="requires a non-empty character_uuid for every character",
    ):
        ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_single_character_without_approved_identity_reference_remains_compatible() -> None:
    request = _request(
        approved_reference_ids=(),
        characters=({"character_uuid": "alice", "approved_reference_ids": []},),
    )

    ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)
