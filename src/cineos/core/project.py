"""Top-level CINEOS movie project model."""

from dataclasses import dataclass, field
from uuid import UUID

from cineos.assets import AssetRegistry as ProductionAssetRegistry

from .asset import Character, Environment, Prop
from .scene import Scene
from .timeline import Timeline


@dataclass(slots=True)
class MovieProject:
    """Complete, renderer-independent description of a movie project."""

    title: str
    author: str
    version: str = "1.0"
    fps: float = 24.0
    resolution: tuple[int, int] = (1920, 1080)
    aspect_ratio: str = "16:9"
    characters: list[Character] = field(default_factory=list)
    locations: list[Environment] = field(default_factory=list)
    props: list[Prop] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    timeline: Timeline = field(default_factory=Timeline)
    asset_registry: ProductionAssetRegistry = field(
        default_factory=ProductionAssetRegistry
    )
    asset_ids: list[UUID] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.resolution) != 2 or any(value <= 0 for value in self.resolution):
            raise ValueError("resolution must contain two positive dimensions")
        self.asset_ids = [UUID(str(value)) for value in self.asset_ids]
