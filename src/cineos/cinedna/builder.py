"""Build deterministic CineDNA solely from approved canonical metadata."""

from __future__ import annotations

from copy import deepcopy

from cineos.assets import Character

from .body import BodyProfile
from .constraints import ContinuityConstraints
from .exceptions import ConflictingIdentityDataError, MissingIdentityDataError
from .expression import STANDARD_EXPRESSIONS, ExpressionProfile
from .face import FaceProfile
from .motion import MotionProfile
from .profile import CharacterDNA
from .validator import CineDNAValidator
from .voice import VoiceProfile
from .wardrobe import WardrobeProfile


class CineDNABuilder:
    """Translate explicit character metadata without inferring visual traits."""

    def build(self, character: Character) -> CharacterDNA:
        if not isinstance(character, Character):
            raise TypeError("character must be a CharacterAsset")
        approved = sorted(
            (ref for ref in character.references if ref.approval_status == "approved"),
            key=lambda ref: str(ref.reference_id),
        )
        if not approved:
            raise MissingIdentityDataError("character requires an approved reference")
        identity = character.metadata.get("cinedna", character.metadata.get("identity"))
        if not isinstance(identity, dict):
            raise MissingIdentityDataError(
                "character metadata requires a cinedna identity object"
            )
        required = ("face", "body")
        missing = [
            name for name in required if not isinstance(identity.get(name), dict)
        ]
        if missing:
            raise MissingIdentityDataError(
                "missing identity data: " + ", ".join(missing)
            )

        approved_ids = [str(ref.reference_id) for ref in approved]
        face_data = deepcopy(identity["face"])
        face_data.setdefault("reference_asset_ids", approved_ids)
        expressions_data = identity.get("expressions", {})
        if not isinstance(expressions_data, dict):
            raise MissingIdentityDataError("expressions must be an object")
        expressions: dict[str, ExpressionProfile] = {}
        for name in STANDARD_EXPRESSIONS:
            item = expressions_data.get(name, {})
            if item is None:
                item = {}
            if isinstance(item, str):
                item = {"description": item}
            expressions[name] = ExpressionProfile(name=name, **deepcopy(item))
        for name, item in expressions_data.items():
            if name not in expressions:
                expressions[name] = ExpressionProfile(name=name, **deepcopy(item))

        wardrobe = [
            WardrobeProfile(**deepcopy(item)) for item in identity.get("wardrobe", [])
        ]
        self._reject_wardrobe_conflicts(wardrobe)
        profile = CharacterDNA(
            character_uuid=character.asset_id,
            profile_version=str(identity.get("profile_version", "1.0")),
            display_name=str(identity.get("display_name", character.name)),
            approved_reference_ids=approved_ids,
            face_profile=FaceProfile(**face_data),
            body_profile=BodyProfile(**deepcopy(identity["body"])),
            wardrobe_profiles=wardrobe,
            voice_profile=VoiceProfile(**deepcopy(identity.get("voice", {}))),
            motion_profile=MotionProfile(**deepcopy(identity.get("motion", {}))),
            expression_profiles=expressions,
            continuity_constraints=ContinuityConstraints(
                **deepcopy(identity.get("constraints", {}))
            ),
            metadata=deepcopy(identity.get("metadata", {})),
        )
        profile.refresh_content_hash()
        CineDNAValidator().raise_for_errors(profile)
        return profile

    build_from_character_asset = build

    @staticmethod
    def _reject_wardrobe_conflicts(items: list[WardrobeProfile]) -> None:
        locks: dict[str, str] = {}
        for item in items:
            if not item.continuity_lock:
                continue
            for scene in item.scene_applicability:
                if scene in locks and locks[scene] != item.wardrobe_asset_id:
                    raise ConflictingIdentityDataError(
                        f"conflicting wardrobe locks for scene {scene!r}"
                    )
                locks[scene] = item.wardrobe_asset_id
