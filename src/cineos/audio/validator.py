"""Audio preflight and dry-run validation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .project import AudioProject
from .provider import TextToSpeechProvider
from .timeline import AudioTimeline


@dataclass(slots=True)
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    ffmpeg_available: bool = False

    @property
    def valid(self) -> bool:
        return not self.errors


class AudioValidator:
    def validate(
        self,
        project: AudioProject,
        *,
        provider: TextToSpeechProvider | None = None,
        timeline: AudioTimeline | None = None,
        check_ffmpeg: bool = False,
    ) -> ValidationReport:
        result = ValidationReport(ffmpeg_available=shutil.which("ffmpeg") is not None)
        assigned = project.voice_assignments
        profiles = {item.voice_id: item for item in project.voice_profiles}
        for cue in project.dialogue_tracks:
            identifier = cue.approved_voice_profile_id or assigned.get(cue.character_id)
            if identifier not in profiles:
                result.errors.append(f"missing approved voice for cue {cue.cue_id}")
                continue
            voice = profiles[identifier]
            if not voice.rights_approved or not voice.approved_reference_ids:
                result.errors.append(
                    f"voice {identifier} lacks approved rights/reference"
                )
            if provider and not provider.capabilities.supports(cue.language):
                result.errors.append(
                    f"provider {provider.provider_id} does not support {cue.language}"
                )
        for cue in [
            *project.ambience_tracks,
            *project.effects_tracks,
            *project.music_tracks,
        ]:
            source = getattr(cue, "source_type", None)
            reference = getattr(cue, "asset_reference", None) or getattr(
                cue, "approved_asset_reference", None
            )
            if source and str(source) != "none" and not reference:
                result.errors.append(f"missing approved asset for cue {cue.cue_id}")
            if reference and not Path(reference).exists():
                result.warnings.append(f"asset is not locally available: {reference}")
        if timeline:
            result.errors += [
                f"overlap conflict: {left}, {right}"
                for left, right in timeline.overlap_conflicts()
            ]
        if check_ffmpeg and not result.ffmpeg_available:
            result.warnings.append(
                "FFmpeg unavailable; WAV silence fallback remains available"
            )
        result.expected_outputs = project.output_targets or [
            "mixed.wav",
            "cue-sheet.json",
            "lip-sync.json",
            "report.json",
            "checksums.json",
        ]
        return result
