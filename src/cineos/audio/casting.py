"""Stable, rights-aware voice casting."""

from dataclasses import dataclass, field
from typing import Any

from .exceptions import VoiceCastingError
from .voice import VoiceProfile


@dataclass(slots=True)
class VoiceAssignment:
    character_id: str
    voice_profile_id: str
    scene_id: str | None = None
    override_approved: bool = False


@dataclass(slots=True)
class VoiceCasting:
    profiles: dict[str, VoiceProfile] = field(default_factory=dict)
    assignments: dict[str, str] = field(default_factory=dict)
    scene_overrides: dict[tuple[str, str], str] = field(default_factory=dict)

    def add_profile(self, profile: VoiceProfile) -> None:
        self.profiles[profile.voice_id] = profile

    def resolve_character_dna(
        self, character_id: str, character_dna: Any
    ) -> VoiceProfile:
        voice = getattr(character_dna, "voice", character_dna)
        references = list(getattr(voice, "approved_voice_reference_ids", []))
        if not references:
            raise VoiceCastingError(
                f"character {character_id} has no approved voice identity"
            )
        profile = VoiceProfile(
            character_id=character_id,
            language=getattr(voice, "language", ""),
            accent=getattr(voice, "accent", ""),
            vocal_age_range=getattr(voice, "vocal_age", ""),
            pitch_range=getattr(voice, "pitch_range", ""),
            cadence=getattr(voice, "cadence", ""),
            emotional_range=list(getattr(voice, "emotional_range", [])),
            approved_reference_ids=references,
            rights_approved=True,
        )
        self.add_profile(profile)
        return profile

    def assign(
        self,
        character_id: str,
        voice_profile_id: str,
        *,
        scene_id: str | None = None,
        approved_override: bool = False,
    ) -> VoiceAssignment:
        profile = self.profiles.get(voice_profile_id)
        if profile is None or profile.character_id != character_id:
            raise VoiceCastingError(
                "voice profile is missing or belongs to another character"
            )
        if not profile.rights_approved or not profile.approved_reference_ids:
            raise VoiceCastingError(
                "voice profile has no explicit approved rights/reference"
            )
        if scene_id is not None:
            if not approved_override:
                raise VoiceCastingError("scene-level voice overrides require approval")
            self.scene_overrides[(scene_id, character_id)] = voice_profile_id
            return VoiceAssignment(character_id, voice_profile_id, scene_id, True)
        existing = self.assignments.get(character_id)
        if existing and existing != voice_profile_id:
            raise VoiceCastingError(f"conflicting voice assignment for {character_id}")
        self.assignments[character_id] = voice_profile_id
        return VoiceAssignment(character_id, voice_profile_id)

    def voice_for(self, character_id: str, scene_id: str | None = None) -> VoiceProfile:
        identifier = (
            self.scene_overrides.get((scene_id, character_id)) if scene_id else None
        )
        identifier = identifier or self.assignments.get(character_id)
        if not identifier:
            raise VoiceCastingError(f"no approved voice assigned to {character_id}")
        return self.profiles[identifier]
