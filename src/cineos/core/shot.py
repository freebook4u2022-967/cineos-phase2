"""Shot model."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class Shot:
    """The smallest ordered unit in the CINEOS project model."""

    shot_id: str
    camera: str = ""
    lens: str = ""
    movement: str = ""
    lighting: str = ""
    action: str = ""
    dialogue: str = ""
    duration: float = 0.0
    references: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError("shot duration cannot be negative")
