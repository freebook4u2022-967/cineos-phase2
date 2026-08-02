"""Environment and weather derived ambience planning."""

from dataclasses import dataclass, field

from .cue import AudioCue, CueType


@dataclass(slots=True)
class AmbiencePlanner:
    """Plan descriptions and approved assets without inventing recordings."""

    manual_overrides: dict[str, list[AudioCue]] = field(default_factory=dict)

    def plan(
        self,
        scene_id: str,
        start: float,
        duration: float,
        *,
        environment: str = "",
        weather: str = "",
        approved_asset_reference: str | None = None,
    ) -> list[AudioCue]:
        if scene_id in self.manual_overrides:
            return list(self.manual_overrides[scene_id])
        description = ", ".join(value for value in (environment, weather) if value)
        return [
            AudioCue(
                scene_id,
                start,
                duration,
                CueType.AMBIENCE,
                asset_reference=approved_asset_reference,
                description=description,
            )
        ]

    def override(self, scene_id: str, cues: list[AudioCue]) -> None:
        self.manual_overrides[scene_id] = list(cues)
