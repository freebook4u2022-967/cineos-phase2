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


def approve_character_reference(
    registry: AssetRegistry,
    character_identifier: str,
    reference_path: str,
    identity: dict[str, Any],
    *,
    view_type: str = "front",
    notes: str = "approved identity reference",
) -> tuple[CanonicalCharacter, CharacterDNA]:
    """Approve one reference and build a validated CineDNA profile.

    Identity descriptors are explicit user/project data. The workflow never
    infers face or body traits from the image path and therefore cannot silently
    fabricate CineDNA.
    """
    if not reference_path.strip():
        raise ValueError("reference_path must not be empty")
    if not isinstance(identity.get("face"), dict):
        raise ValueError("identity requires an explicit face object")
    if not isinstance(identity.get("body"), dict):
        raise ValueError("identity requires an explicit body object")

    character = _resolve_character(registry, character_identifier)
    reference = character.add_reference(
        reference_path,
        view_type=view_type,
        approval_status="approved",
        notes=notes,
        source="short-drama-character-approval",
    )
    character.metadata["cinedna"] = {
        **identity,
        "status": "approved",
        "display_name": identity.get("display_name", character.name),
    }
    character.metadata["identity_approval"] = {
        "status": "approved",
        "reference_id": str(reference.reference_id),
        "view_type": view_type,
    }
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
    """Persist an approved reference, canonical asset update and CineDNA profile."""
    assets_path = Path(asset_registry_path)
    identity_file = Path(identity_path)
    identity = json.loads(identity_file.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise ValueError("identity JSON must contain an object")

    registry = load_asset_registry(assets_path)
    character, profile = approve_character_reference(
        registry,
        character_identifier,
        reference_path,
        identity,
        view_type=view_type,
    )
    save_asset_registry(registry, assets_path)

    profile_file = Path(profiles_path)
    profiles = (
        CineDNARegistry.load(profile_file)
        if profile_file.exists()
        else CineDNARegistry()
    )
    profiles.register(profile)
    profiles.save(profile_file)
    return {
        "character_id": str(character.asset_id),
        "character_name": character.name,
        "reference_id": character.metadata["identity_approval"]["reference_id"],
        "cinedna_profile": str(profile_file),
        "asset_registry": str(assets_path),
    }
