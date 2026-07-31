"""Structural and continuity validation for CineDNA."""

from .expression import STANDARD_EXPRESSIONS
from .profile import CharacterDNA
from .serializer import calculate_content_hash


class CineDNAValidator:
    def validate(self, profile: CharacterDNA) -> list[str]:
        errors: list[str] = []
        if not profile.display_name.strip():
            errors.append("display name cannot be empty")
        if not profile.approved_reference_ids:
            errors.append("at least one approved reference is required")
        if len(profile.approved_reference_ids) != len(
            set(profile.approved_reference_ids)
        ):
            errors.append("approved reference IDs must be unique")
        if not profile.face_profile.reference_asset_ids:
            errors.append("face profile requires an approved reference")
        unknown = set(profile.face_profile.reference_asset_ids) - set(
            profile.approved_reference_ids
        )
        if unknown:
            errors.append("face profile contains an unapproved reference")
        names = set(profile.expression_profiles)
        missing = set(STANDARD_EXPRESSIONS) - names
        if missing:
            errors.append("missing standard expressions: " + ", ".join(sorted(missing)))
        locked: dict[str, str] = {}
        for wardrobe in profile.wardrobe_profiles:
            if wardrobe.continuity_lock:
                for scene in wardrobe.scene_applicability:
                    previous = locked.get(scene)
                    if previous and previous != wardrobe.wardrobe_asset_id:
                        errors.append(f"conflicting wardrobe locks for scene {scene!r}")
                    locked[scene] = wardrobe.wardrobe_asset_id
        if profile.content_hash and profile.content_hash != calculate_content_hash(
            profile
        ):
            errors.append("content hash does not match profile content")
        return errors

    def raise_for_errors(self, profile: CharacterDNA) -> None:
        errors = self.validate(profile)
        if errors:
            from .exceptions import CineDNAError

            raise CineDNAError("; ".join(errors))


def validate(profile: CharacterDNA) -> list[str]:
    return CineDNAValidator().validate(profile)
