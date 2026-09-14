"""Provider-neutral character consistency contracts for Atlas Native Renderer."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from cineos.conditioning import CharacterConditioning


class CharacterConsistencyError(ValueError):
    """Raised when a character cannot be safely conditioned for native rendering."""


def build_character_consistency_conditioning(
    character: Any, profile: Any
) -> CharacterConditioning:
    """Compile an approved identity lock into Atlas CharacterConditioning.

    This is intentionally renderer-neutral. It contains identity references and
    invariants but no Kling/Seedance/Veo prompt or API fields.
    """
    lock = character.metadata.get("identity_lock")
    if not isinstance(lock, dict):
        raise CharacterConsistencyError(
            f"character {character.name!r} has no approved identity lock"
        )
    ranked = lock.get("references")
    if not isinstance(ranked, list) or not ranked:
        raise CharacterConsistencyError(
            f"character {character.name!r} has no ranked approved references"
        )
    reference_ids = [str(item["reference_id"]) for item in ranked]
    profile_ids = {str(item) for item in profile.approved_reference_ids}
    if not set(reference_ids).issubset(profile_ids):
        raise CharacterConsistencyError(
            "identity lock contains references not approved by CineDNA"
        )

    face = lock.get("face", {})
    body = lock.get("body", {})
    constraints = lock.get("constraints", {})
    invariants = list(face.get("invariants", []))
    invariants.extend(constraints.get("immutable_facial_traits", []))
    invariants.extend(constraints.get("immutable_body_traits", []))
    invariants.extend(f"forbid:{item}" for item in lock.get("forbidden_changes", []))

    return CharacterConditioning(
        character_uuid=str(character.asset_id),
        cinedna_profile_id=str(profile.character_uuid),
        cinedna_profile_version=profile.profile_version,
        approved_reference_ids=reference_ids,
        identity_invariants=list(dict.fromkeys(invariants)),
        face_constraints=face,
        body_constraints=body,
        scene_specific_overrides={
            "identity_lock_schema": lock.get("schema"),
            "primary_reference_id": lock.get("primary_reference_id"),
            "reference_strategy": lock.get("reference_strategy"),
            "ranked_references": ranked,
        },
    )


def character_conditioning_payload(
    conditioning: CharacterConditioning,
) -> dict[str, Any]:
    """Return a JSON-safe native-renderer payload."""
    return asdict(conditioning)
