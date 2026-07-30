"""Scene model."""

from dataclasses import dataclass, field

from .shot import Shot


@dataclass(slots=True)
class Scene:
    """A scene and its shots, asset references, and declared duration."""

    scene_id: str
    title: str
    description: str = ""
    shots: list[Shot] = field(default_factory=list)
    location: str | None = None
    characters: list[str] = field(default_factory=list)
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError("scene duration cannot be negative")

    @property
    def shot_duration(self) -> float:
        """Return the duration of all shots in the scene."""

        return sum(shot.duration for shot in self.shots)
