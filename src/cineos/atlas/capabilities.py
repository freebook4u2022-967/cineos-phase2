"""Renderer capability declarations and negotiation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


class CapabilityError(ValueError):
    """Raised when a render request cannot be satisfied by a renderer."""


@dataclass(frozen=True, slots=True)
class Resolution:
    """A pixel resolution."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("resolution dimensions must be positive")


@dataclass(frozen=True, slots=True)
class Range:
    """An inclusive numeric range used for duration and frame-rate limits."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError("range must be non-negative and ordered")

    def supports(self, value: float) -> bool:
        """Return whether *value* is within the inclusive range."""

        return self.minimum <= value <= self.maximum


@dataclass(frozen=True, slots=True)
class RendererCapabilities:
    """The limits and optional features advertised by a renderer."""

    supported_resolution: tuple[Resolution, ...]
    supported_duration: Range
    supported_fps: tuple[float, ...]
    supported_features: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        resolutions = tuple(self.supported_resolution)
        fps_values = tuple(self.supported_fps)
        features = frozenset(self.supported_features)
        if not resolutions:
            raise ValueError("at least one supported resolution is required")
        if not fps_values or any(fps <= 0 for fps in fps_values):
            raise ValueError("supported fps values must be non-empty and positive")
        if any(not feature for feature in features):
            raise ValueError("feature names must not be empty")
        object.__setattr__(self, "supported_resolution", resolutions)
        object.__setattr__(self, "supported_fps", fps_values)
        object.__setattr__(self, "supported_features", features)

    def negotiate(
        self,
        *,
        resolution: Resolution | tuple[int, int],
        duration: float,
        fps: float,
        features: Iterable[str] = (),
    ) -> NegotiatedCapabilities:
        """Validate a requested configuration and return its normalized form."""

        requested_resolution = (
            resolution
            if isinstance(resolution, Resolution)
            else Resolution(*resolution)
        )
        requested_features = frozenset(features)
        problems: list[str] = []
        if requested_resolution not in self.supported_resolution:
            problems.append(
                f"resolution {requested_resolution.width}x{requested_resolution.height}"
            )
        if not self.supported_duration.supports(duration):
            problems.append(f"duration {duration}")
        if fps not in self.supported_fps:
            problems.append(f"fps {fps}")
        missing_features = requested_features - self.supported_features
        if missing_features:
            problems.append(f"features {', '.join(sorted(missing_features))}")
        if problems:
            raise CapabilityError("unsupported " + "; ".join(problems))
        return NegotiatedCapabilities(
            resolution=requested_resolution,
            duration=duration,
            fps=fps,
            features=requested_features,
        )


@dataclass(frozen=True, slots=True)
class NegotiatedCapabilities:
    """A renderer configuration accepted during capability negotiation."""

    resolution: Resolution
    duration: float
    fps: float
    features: frozenset[str]
