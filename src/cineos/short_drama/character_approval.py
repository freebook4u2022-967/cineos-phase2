"""Character reference approval workflow for Short Drama Agent projects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from cineos.assets import AssetRegistry
from cineos.assets import Character as CanonicalCharacter
from cineos.assets.storage import load as load_asset_registry
from cineos.assets.storage import save as save_asset_registry
from cineos.cinedna import CharacterDNA, CineDNABuilder, CineDNARegistry
from cineos.cinedna.exceptions import ProfileNotFoundError

_REFERENCE_WEIGHTS = {
    "front": 100,
    "three-quarter": 95,
    "close-up": 90,
    "left-profile": 80,
    "right-profile": 80,
    "full-body": 75,
    "expression": 65,
    "wardrobe": 55,
    "rear": 40,
}


def _resolve_character(registry: AssetRegistry, identifier: str) -> CanonicalCharacter:
    """Resolve a canonical character by UUID or case-insensitive display name."""
    try:
        asset = registry.retrieve(UUID(identifier))
        if not isinstance(asset, CanonicalCharacter):
            raise ValueError(f"asset {identifier!r} is not a character")
        return asset
    except (ValueError, TypeError):
        pass

    matches = [
        asset
        for asset in registry.list(kind="character")
        if asset.name.casefold() == identifier.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(f"character {identifier!r} was not uniquely resolved")
    return matches[0]  # type: ignore[return-value]


def _rank_approved_references(character: CanonicalCharacter) -> list[dict[str, Any]]:
    approved = [
        reference
        for reference in character.references
        if reference.approval_status == "approved"
    ]
    ranked = sorted(
        approved,
        key=lambda reference: (
            -_REFERENCE_WEIGHTS.get(reference.view_type, 0),
            -reference.priority,
            str(reference.reference_id),
        ),
    )
    return [
        {
            "reference_id": str(reference.reference_id),
            "file_path": reference.file_path,
            "view_type": reference.view_type,
            "rank": index + 1,
            "weight": _REFERENCE_WEIGHTS.get(reference.view_type, 0),
        }
        for index, reference in enumerate(ranked)
    ]


def _identity_lock_package(
    character: CanonicalCharacter, identity: dict[str, Any]
) -> dict[str, Any]:
    ranked = _rank_approved_references(character)
    constraints = identity.get("constraints", {})
    return {
        "schema": "cineos-character-identity-lock/0.1",
        "character_id": str(character.asset_id),
        "display_name": identity.get("display_name", character.name),
        "reference_strategy": "ranked-multi-reference",
        "references": ranked,
        "primary_reference_id": ranked[0]["reference_id"] if ranked else None,
        "face": identity["face"],
        "body": identity["body"],
        "constraints": constraints if isinstance(constraints, dict) else {},
        "forbidden_changes": (
            constraints.get("forbidden_changes", [])
            if isinstance(constraints, dict)
            else []
        ),
    }


def _next_profile_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) == 1 and parts[0].isdigit():
        return f"{parts[0]}.1"
    if parts and parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    return f"{version}.1"


def approve_character_reference(
    registry: AssetRegistry,
    character_identifier: str,
    reference_path: str,
    identity: dict[str, Any],
    *,
    view_type: str = "front",
    notes: str = "approved identity reference",
) -> tuple[CanonicalCharacter, CharacterDNA]:
    """Approve a reference and rebuild the validated multi-reference CineDNA profile."""
    if not reference_path.strip():
        raise ValueError("reference_path must not be empty")
    if not isinstance(identity.get("face"), dict):
        raise ValueError("identity requires an explicit face object")
    if not isinstance(identity.get("body"), dict):
        raise ValueError("identity requires an explicit body object")

    character = _resolve_character(registry, character_identifier)
    duplicate = next(
        (
            reference
            for reference in character.references
            if reference.file_path == reference_path
            and reference.view_type == view_type
            and reference.approval_status == "approved"
        ),
        None,
    )
    if duplicate is None:
        reference = character.add_reference(
            reference_path,
            view_type=view_type,
            approval_status="approved",
            notes=notes,
            priority=_REFERENCE_WEIGHTS.get(view_type, 0),
            source="short-drama-character-approval",
        )
    else:
        reference = duplicate

    character.metadata["cinedna"] = {
        **identity,
        "status": "approved",
        "display_name": identity.get("display_name", character.name),
    }
    character.metadata["identity_approval"] = {
        "status": "approved",
        "reference_id": str(reference.reference_id),
        "view_type": view_type,
        "approved_reference_count": len(
            [
                item
                for item in character.references
                if item.approval_status == "approved"
            ]
        ),
    }
    character.metadata["identity_lock"] = _identity_lock_package(character, identity)
    character.touch()
    profile = CineDNABuilder().build(character)
    return character, profile


def approve_character_files(
    asset_registry_path: str | Path,
    character_identifier: str,
    reference_path: str,
    identity_path: str | Path,
    *,
    profiles_path: str | Path,
    view_type: str = "front",
) -> dict[str, str]:
    """Persist reference approval and version CineDNA only when identity changes."""
    assets_path = Path(asset_registry_path)
    identity_file = Path(identity_path)
    identity = json.loads(identity_file.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise ValueError("identity JSON must contain an object")

    registry = load_asset_registry(assets_path)
    existing_character = _resolve_character(registry, character_identifier)
    duplicate = next(
        (
            reference
            for reference in existing_character.references
            if reference.file_path == reference_path
            and reference.view_type == view_type
            and reference.approval_status == "approved"
        ),
        None,
    )

    profile_file = Path(profiles_path)
    profiles = (
        CineDNARegistry.load(profile_file)
        if profile_file.exists()
        else CineDNARegistry()
    )
    previous_profile = None
    try:
        previous_profile = profiles.retrieve(existing_character.asset_id)
    except ProfileNotFoundError:
        pass

    if duplicate is not None and previous_profile is not None:
        lock = existing_character.metadata.get("identity_lock", {})
        return {
            "character_id": str(existing_character.asset_id),
            "character_name": existing_character.name,
            "reference_id": str(duplicate.reference_id),
            "approved_reference_count": str(
                len(
                    [
                        item
                        for item in existing_character.references
                        if item.approval_status == "approved"
                    ]
                )
            ),
            "primary_reference_id": str(lock.get("primary_reference_id", "")),
            "cinedna_profile": str(profile_file),
            "asset_registry": str(assets_path),
        }

    if previous_profile is not None:
        identity = dict(identity)
        identity["profile_version"] = _next_profile_version(
            previous_profile.profile_version
        )

    character, profile = approve_character_reference(
        registry,
        character_identifier,
        reference_path,
        identity,
        view_type=view_type,
    )
    save_asset_registry(registry, assets_path)
    profiles.register(profile)
    profiles.save(profile_file)
    return {
        "character_id": str(character.asset_id),
        "character_name": character.name,
        "reference_id": character.metadata["identity_approval"]["reference_id"],
        "approved_reference_count": str(
            character.metadata["identity_approval"]["approved_reference_count"]
        ),
        "primary_reference_id": character.metadata["identity_lock"][
            "primary_reference_id"
        ],
        "cinedna_profile": str(profile_file),
        "asset_registry": str(assets_path),
    }
