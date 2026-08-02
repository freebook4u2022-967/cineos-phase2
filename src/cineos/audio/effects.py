"""Action, prop, vehicle, and transition effect planning."""

from dataclasses import dataclass, field

from .cue import AudioCue, CueType


@dataclass(slots=True)
class EffectsPlanner:
    manual_overrides: dict[str, list[AudioCue]] = field(default_factory=dict)

    def plan(
        self,
        scene_id: str,
        shot_id: str,
        start: float,
        duration: float,
        *,
        actions: list[str] | None = None,
        props: list[str] | None = None,
        vehicles: list[str] | None = None,
        transition: str = "",
    ) -> list[AudioCue]:
        if shot_id in self.manual_overrides:
            return list(self.manual_overrides[shot_id])
        cues: list[AudioCue] = []
        for kind, values in (
            (CueType.FOLEY, actions),
            (CueType.SOUND_EFFECT, props),
            (CueType.SOUND_EFFECT, vehicles),
        ):
            for value in values or []:
                cues.append(
                    AudioCue(
                        scene_id,
                        start,
                        duration,
                        kind,
                        shot_id=shot_id,
                        description=value,
                    )
                )
        if transition:
            cues.append(
                AudioCue(
                    scene_id,
                    start + duration,
                    0,
                    CueType.TRANSITION,
                    shot_id=shot_id,
                    description=transition,
                )
            )
        return cues

    def override(self, shot_id: str, cues: list[AudioCue]) -> None:
        self.manual_overrides[shot_id] = list(cues)
